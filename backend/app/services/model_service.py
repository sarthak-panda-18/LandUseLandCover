"""
Model Service for LULC Classification
Provides lazy loading of trained Random Forest pixel model, scaler, and metadata,
using ModelManager for single model residency and streaming windowed raster execution.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.windows import Window

from app.services.model_manager import get_model_manager

logger = logging.getLogger("lulc.service.model")

CLASS_METADATA = {
    0: {"name": "Water",        "color_hex": "#1E88E5", "rgba": (30, 136, 229, 255)},
    1: {"name": "Trees/Forest", "color_hex": "#2E7D32", "rgba": (46, 125, 50, 255)},
    2: {"name": "Crops",        "color_hex": "#81C784", "rgba": (129, 199, 132, 255)},
    3: {"name": "Grassland",    "color_hex": "#DCE775", "rgba": (220, 231, 117, 255)},
    4: {"name": "Built-up",     "color_hex": "#E53935", "rgba": (229, 57, 53, 255)},
    5: {"name": "Bare land",    "color_hex": "#8D6E63", "rgba": (141, 110, 99, 255)},
    255: {"name": "Masked",     "color_hex": "#000000", "rgba": (0, 0, 0, 0)}
}


class ModelService:
    _instance: Optional["ModelService"] = None

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[2]  # backend/
        self.models_dir = self.base_dir / "ml" / "models"
        self.metrics_dir = self.base_dir / "ml" / "outputs" / "metrics"
        
        self.metadata_path = self.models_dir / "model_metadata.json"
        self.metrics_path = self.metrics_dir / "test_metrics.json"

        self.metadata: Dict[str, Any] = {}
        self.test_metrics: Dict[str, Any] = {}
        # Artifacts are loaded lazily on first inference request via ModelManager

    def _load_json_metadata(self):
        """Loads lightweight JSON metadata without requiring model binary loading."""
        if not self.metadata and self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata JSON: {e}")

        if not self.metadata:
            self.metadata = {
                "model_type": "RandomForestClassifier",
                "class_mapping": {k: v["name"] for k, v in CLASS_METADATA.items() if k != 255}
            }

        if not self.test_metrics and self.metrics_path.exists():
            try:
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    self.test_metrics = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load test metrics JSON: {e}")

    def classify_tile_streaming(
        self,
        raster_path: Path,
        strip_height: int = 512,
        batch_size: int = 250_000
    ) -> Dict[str, Any]:
        """
        Classifies a Sentinel-2 GeoTIFF raster using windowed streaming reads.
        Loads only 512 rows into memory at a time, keeping RAM bounded to minimal footprint.
        """
        model_mgr = get_model_manager()
        scaler, model = model_mgr.get_pixel_rf_model()

        with rasterio.open(raster_path) as src:
            height = src.height
            width = src.width
            band_count = src.count
            total_pixels = height * width

            if band_count < 4:
                raise ValueError(f"Uploaded GeoTIFF has {band_count} band(s). Sentinel-2 requires >= 4 bands.")

            # Preallocate output classified array in uint8
            classified_array = np.full((height, width), 255, dtype=np.uint8)

            for r_start in range(0, height, strip_height):
                r_end = min(r_start + strip_height, height)
                h_strip = r_end - r_start
                win = Window(col_off=0, row_off=r_start, width=width, height=h_strip)

                if band_count >= 5:
                    strip_raw = src.read([1, 2, 3, 4, 5], window=win).astype(np.float32)
                    strip_5band = strip_raw[:5]
                else:
                    strip_4band = src.read([1, 2, 3, 4], window=win).astype(np.float32)
                    b4_red = strip_4band[2]
                    b8_nir = strip_4band[3]
                    denom = b8_nir + b4_red
                    ndvi = np.where(denom != 0, (b8_nir - b4_red) / (denom + 1e-6), 0.0)
                    strip_5band = np.concatenate([strip_4band, ndvi[np.newaxis, ...]], axis=0)

                # Transpose to (H_strip, W, 5) and flatten to (N_strip, 5)
                strip_hwc = np.transpose(strip_5band, (1, 2, 0))
                flat_features = strip_hwc.reshape(-1, 5)

                valid_mask = (
                    ~np.isnan(flat_features).any(axis=1) &
                    ~np.isinf(flat_features).any(axis=1) &
                    ~((flat_features == 0).all(axis=1)) &
                    ~((flat_features == -9999).any(axis=1))
                )

                valid_count_strip = int(valid_mask.sum())
                if valid_count_strip > 0:
                    X_valid = flat_features[valid_mask]
                    preds_chunks = []
                    for i in range(0, len(X_valid), batch_size):
                        chunk_raw = X_valid[i:i + batch_size]
                        chunk_scaled = scaler.transform(chunk_raw).astype(np.float32)
                        chunk_pred = model.predict(chunk_scaled)
                        preds_chunks.append(chunk_pred)

                    strip_classified = np.full(flat_features.shape[0], 255, dtype=np.uint8)
                    strip_classified[valid_mask] = np.concatenate(preds_chunks).astype(np.uint8)
                    classified_array[r_start:r_end, :] = strip_classified.reshape(h_strip, width)

                del strip_5band, strip_hwc, flat_features

        # Compute per-class counts from final uint8 array
        flat_classified = classified_array.ravel()
        valid_pixel_mask = (flat_classified != 255)
        valid_count = int(valid_pixel_mask.sum())
        invalid_count = total_pixels - valid_count

        counts = np.bincount(flat_classified, minlength=256)

        class_distribution: List[Dict[str, Any]] = []
        for class_id in range(6):
            meta = CLASS_METADATA[class_id]
            pixel_count = int(counts[class_id])
            pct_of_valid = round((pixel_count / valid_count * 100.0), 2) if valid_count > 0 else 0.0
            pct_of_total = round((pixel_count / total_pixels * 100.0), 2)

            class_distribution.append({
                "class_id": class_id,
                "class_name": meta["name"],
                "color_hex": meta["color_hex"],
                "pixel_count": pixel_count,
                "percentage_valid": pct_of_valid,
                "percentage_total": pct_of_total
            })

        masked_pct = round((invalid_count / total_pixels * 100.0), 2)

        return {
            "classified_array": classified_array,
            "dimensions": {"height": height, "width": width},
            "total_pixels": total_pixels,
            "valid_pixels": valid_count,
            "invalid_pixels": invalid_count,
            "valid_percentage": round((valid_count / total_pixels * 100.0), 2) if total_pixels > 0 else 0.0,
            "masked_percentage": masked_pct,
            "class_distribution": class_distribution
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Returns structured information about the trained model, hyperparameters, and test metrics."""
        self._load_json_metadata()
        return {
            "model_type": self.metadata.get("model_type", "RandomForestClassifier"),
            "training_date": self.metadata.get("training_date"),
            "training_samples": self.metadata.get("training_samples_count"),
            "validation_samples": self.metadata.get("validation_samples_count"),
            "hyperparameters": self.metadata.get("hyperparameters", {}),
            "feature_importances": self.metadata.get("feature_importances", {}),
            "test_metrics": self.test_metrics.get("summary_metrics", {}),
            "per_class_metrics": self.test_metrics.get("per_class_metrics", {}),
            "class_mapping": CLASS_METADATA
        }


# Global singleton accessor
_model_service_instance: Optional[ModelService] = None


def get_model_service() -> ModelService:
    global _model_service_instance
    if _model_service_instance is None:
        _model_service_instance = ModelService()
    return _model_service_instance
