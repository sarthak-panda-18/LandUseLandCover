"""
Model Service for LULC Classification
Provides lazy loading of trained Random Forest pixel model, scaler, and metadata,
with memory-bounded chunked tile inference (250,000 pixels per batch).
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np

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
        
        self.model_path = self.models_dir / "lulc_rf_model.joblib"
        self.scaler_path = self.models_dir / "scaler.joblib"
        self.metadata_path = self.models_dir / "model_metadata.json"
        self.metrics_path = self.metrics_dir / "test_metrics.json"

        self.model = None
        self.scaler = None
        self.metadata: Dict[str, Any] = {}
        self.test_metrics: Dict[str, Any] = {}
        self.is_loaded = False
        # Artifacts are loaded lazily on first inference request, NOT at startup.

    def ensure_loaded(self):
        """Lazily loads model, scaler, and metadata JSON files on demand."""
        if self.is_loaded:
            return

        t_start = time.time()
        logger.info(f"[Lazy Loading] Loading LULC pixel RF model & scaler from {self.models_dir}...")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model artifact not found at {self.model_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler artifact not found at {self.scaler_path}")

        # 1. Load Scaler & Model
        self.scaler = joblib.load(self.scaler_path)
        self.model = joblib.load(self.model_path)

        # 2. Load Model Metadata
        self._load_json_metadata()

        self.is_loaded = True
        duration = time.time() - t_start
        logger.info(f"LULC Pixel Model Service loaded in {duration:.2f}s!")

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

    def classify_tile(self, input_array: np.ndarray, batch_size: int = 250_000) -> Dict[str, Any]:
        """
        Classifies a multi-band Sentinel-2 tile array with chunked memory-bounded execution.
        
        Args:
            input_array: numpy array of shape (5, H, W) or (H, W, 5) with bands (B2, B3, B4, B8, NDVI)
            batch_size: batch chunk size (default 250,000 pixels) for bounded RAM usage
            
        Returns:
            Dictionary containing classified 2D array, per-class counts, percentages, and metrics.
        """
        self.ensure_loaded()

        # Ensure shape (H, W, 5)
        if input_array.ndim != 3:
            raise ValueError(f"Input array must be 3-dimensional (got shape {input_array.shape})")

        if input_array.shape[0] == 5 and input_array.shape[2] != 5:
            # Transpose from (5, H, W) to (H, W, 5)
            input_array = np.transpose(input_array, (1, 2, 0))

        if input_array.shape[2] != 5:
            raise ValueError(f"Expected 5 bands [B2, B3, B4, B8, NDVI], but got {input_array.shape[2]} channels.")

        height, width, _ = input_array.shape
        total_pixels = height * width

        # Flatten to (N, 5)
        flat_features = input_array.reshape(-1, 5).astype(np.float32)

        # Detect valid pixels (exclude NaNs, Infs, NoData, all-zero background)
        valid_mask = (
            ~np.isnan(flat_features).any(axis=1) &
            ~np.isinf(flat_features).any(axis=1) &
            ~((flat_features == 0).all(axis=1)) &
            ~((flat_features == -9999).any(axis=1))
        )

        valid_count = int(valid_mask.sum())

        # Create output buffer initialized to 255 (masked/invalid)
        classified_flat = np.full(total_pixels, 255, dtype=np.uint8)

        if valid_count > 0:
            X_valid = flat_features[valid_mask]
            
            # Predict in bounded chunks of 250,000 pixels to restrict peak RAM
            preds_chunks = []
            for i in range(0, len(X_valid), batch_size):
                chunk_raw = X_valid[i:i + batch_size]
                chunk_scaled = self.scaler.transform(chunk_raw).astype(np.float32)
                chunk_pred = self.model.predict(chunk_scaled)
                preds_chunks.append(chunk_pred)

            classified_flat[valid_mask] = np.concatenate(preds_chunks).astype(np.uint8)

        classified_2d = classified_flat.reshape(height, width)

        # Compute per-class distribution
        class_distribution: List[Dict[str, Any]] = []
        for class_id in range(6):
            meta = CLASS_METADATA[class_id]
            class_mask = (classified_flat == class_id)
            pixel_count = int(class_mask.sum())
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

        # Add masked pixels summary
        masked_count = int((classified_flat == 255).sum())
        masked_pct = round((masked_count / total_pixels * 100.0), 2)

        return {
            "classified_array": classified_2d,
            "dimensions": {"height": height, "width": width},
            "total_pixels": total_pixels,
            "valid_pixels": valid_count,
            "invalid_pixels": masked_count,
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
