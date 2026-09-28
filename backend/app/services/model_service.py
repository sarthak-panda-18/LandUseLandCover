"""
Model Service for LULC Classification
Provides lazy loading of trained Random Forest pixel model, scaler, and metadata,
using ModelManager for single model residency and streaming windowed raster execution.
"""

import gc
import json
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import psutil
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


def get_current_rss_mb() -> float:
    """Returns current process and child processes RSS in Megabytes."""
    try:
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return rss / (1024 * 1024)
    except Exception:
        return 0.0


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
        strip_height: int = 256,
        batch_size: int = 50_000
    ) -> Dict[str, Any]:
        """
        Classifies a Sentinel-2 GeoTIFF raster using ultra-compact windowed streaming reads.
        Loads only 256 rows into memory at a time and batches prediction in 50,000 pixel chunks
        with explicit garbage collection to keep peak RAM well below container limits.
        """
        start_time = time.time()
        try:
            # Checkpoint 1: Before model load
            rss_before_load = get_current_rss_mb()
            logger.info(
                f"[MEM-CHECKPOINT] [{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] "
                f"[1/4 Before Model Load] Process RSS: {rss_before_load:.2f} MB"
            )

            model_mgr = get_model_manager()
            t_load_start = time.time()
            scaler, model = model_mgr.get_pixel_rf_model()
            t_load = (time.time() - t_load_start) * 1000

            # Checkpoint 2: Immediately after model load completes
            rss_after_load = get_current_rss_mb()
            load_delta = rss_after_load - rss_before_load
            logger.info(
                f"[MEM-CHECKPOINT] [{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] "
                f"[2/4 After Model Load] Process RSS: {rss_after_load:.2f} MB (Delta: {load_delta:+.2f} MB in {t_load:.1f}ms)"
            )

            with rasterio.open(raster_path) as src:
                height = src.height
                width = src.width
                band_count = src.count
                total_pixels = height * width

                if band_count < 4:
                    raise ValueError(f"Uploaded GeoTIFF has {band_count} band(s). Sentinel-2 requires >= 4 bands.")

                # Preallocate output classified array in uint8 (e.g. 2048*2048 = 4 MB)
                classified_array = np.full((height, width), 255, dtype=np.uint8)

                chunk_counter = 0
                peak_inference_rss = rss_after_load

                for r_start in range(0, height, strip_height):
                    r_end = min(r_start + strip_height, height)
                    h_strip = r_end - r_start
                    win = Window(col_off=0, row_off=r_start, width=width, height=h_strip)

                    if band_count >= 5:
                        strip_raw = src.read([1, 2, 3, 4, 5], window=win).astype(np.float32)
                        strip_5band = strip_raw[:5]
                        del strip_raw
                    else:
                        strip_4band = src.read([1, 2, 3, 4], window=win).astype(np.float32)
                        b4_red = strip_4band[2]
                        b8_nir = strip_4band[3]
                        denom = b8_nir + b4_red
                        ndvi = np.where(denom != 0, (b8_nir - b4_red) / (denom + 1e-6), 0.0)
                        strip_5band = np.concatenate([strip_4band, ndvi[np.newaxis, ...]], axis=0)
                        del strip_4band, b4_red, b8_nir, denom, ndvi

                    # Transpose to (H_strip, W, 5) and flatten to (N_strip, 5)
                    strip_hwc = np.transpose(strip_5band, (1, 2, 0))
                    flat_features = strip_hwc.reshape(-1, 5)
                    del strip_5band, strip_hwc

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

                            del chunk_raw, chunk_scaled, chunk_pred
                            chunk_counter += 1

                            # Track peak RSS
                            current_chunk_rss = get_current_rss_mb()
                            if current_chunk_rss > peak_inference_rss:
                                peak_inference_rss = current_chunk_rss

                            # Periodic garbage collection every 5 chunks
                            if chunk_counter % 5 == 0:
                                gc.collect()

                            # Checkpoint 3: Log every ~10 chunks
                            if chunk_counter % 10 == 0:
                                logger.info(
                                    f"[MEM-CHECKPOINT] [{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] "
                                    f"[3/4 Processing Chunk #{chunk_counter}] Process RSS: {current_chunk_rss:.2f} MB | Rows {r_start}-{r_end}/{height}"
                                )

                        strip_classified = np.full(flat_features.shape[0], 255, dtype=np.uint8)
                        strip_classified[valid_mask] = np.concatenate(preds_chunks).astype(np.uint8)
                        classified_array[r_start:r_end, :] = strip_classified.reshape(h_strip, width)

                        del X_valid, preds_chunks, strip_classified

                    del flat_features, valid_mask
                    gc.collect()

            # Compute per-class counts from final uint8 array
            flat_classified = classified_array.ravel()
            valid_pixel_mask = (flat_classified != 255)
            valid_count = int(valid_pixel_mask.sum())
            invalid_count = total_pixels - valid_count
            del valid_pixel_mask

            counts = np.bincount(flat_classified, minlength=256)
            del flat_classified

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

            del counts
            gc.collect()

            masked_pct = round((invalid_count / total_pixels * 100.0), 2)

            # Checkpoint 4: Right before returning response
            rss_before_return = get_current_rss_mb()
            elapsed_total = time.time() - start_time
            logger.info(
                f"[MEM-CHECKPOINT] [{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] "
                f"[4/4 Before Response Return] Process RSS: {rss_before_return:.2f} MB (Peak during inference: {peak_inference_rss:.2f} MB, Total Time: {elapsed_total:.2f}s)"
            )

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

        except Exception as e:
            err_trace = traceback.format_exc()
            logger.error(f"[ERROR in classify_tile_streaming (pixel_rf)] {e}\n{err_trace}")
            raise

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
