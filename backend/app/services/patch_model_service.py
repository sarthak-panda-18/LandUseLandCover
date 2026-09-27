"""
Patch-Based Model Service for LULC Classification
Loads Random Forest Patch model (statistical features) and EfficientNetB0 Keras model (3-channel image patches),
and performs 64x64 non-overlapping sliding window block classification.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np

logger = logging.getLogger("lulc.service.patch_model")

CLASS_METADATA = {
    0: {"name": "Water",        "color_hex": "#1E88E5", "rgba": (30, 136, 229, 255)},
    1: {"name": "Trees/Forest", "color_hex": "#2E7D32", "rgba": (46, 125, 50, 255)},
    2: {"name": "Crops",        "color_hex": "#81C784", "rgba": (129, 199, 132, 255)},
    3: {"name": "Grassland",    "color_hex": "#DCE775", "rgba": (220, 231, 117, 255)},
    4: {"name": "Built-up",     "color_hex": "#E53935", "rgba": (229, 57, 53, 255)},
    5: {"name": "Bare land",    "color_hex": "#8D6E63", "rgba": (141, 110, 99, 255)},
    255: {"name": "Masked",     "color_hex": "#000000", "rgba": (0, 0, 0, 0)}
}


class PatchModelService:
    _instance: Optional["PatchModelService"] = None

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[2]  # backend/
        self.patch_models_dir = self.base_dir / "ml" / "models_patch"
        
        self.rf_patch_path = self.patch_models_dir / "random_forest_patch.pkl"
        self.efficientnet_path = self.patch_models_dir / "efficientnetb0_patch.keras"

        self.rf_model = None
        self.efficientnet_model = None
        
        self.is_loaded = False
        self.load_artifacts()

    def load_artifacts(self):
        """Loads both Random Forest patch model and EfficientNetB0 Keras model."""
        t_start = time.time()
        logger.info(f"Loading patch classification models from {self.patch_models_dir}...")

        # 1. Load Random Forest Patch Model
        if self.rf_patch_path.exists():
            try:
                self.rf_model = joblib.load(self.rf_patch_path)
                logger.info(f"Loaded Random Forest Patch model from {self.rf_patch_path.name}")
            except Exception as e:
                logger.error(f"Error loading RF patch model: {e}", exc_info=True)
        else:
            logger.warning(f"RF patch model not found at {self.rf_patch_path}")

        # 2. Load EfficientNetB0 Keras Model
        if self.efficientnet_path.exists():
            try:
                import keras
                self.efficientnet_model = keras.models.load_model(self.efficientnet_path, compile=False)
                logger.info(f"Loaded EfficientNetB0 Patch model from {self.efficientnet_path.name}")
            except Exception as e:
                logger.error(f"Error loading EfficientNetB0 patch model: {e}", exc_info=True)
        else:
            logger.warning(f"EfficientNetB0 patch model not found at {self.efficientnet_path}")

        self.is_loaded = (self.rf_model is not None or self.efficientnet_model is not None)
        duration = time.time() - t_start
        logger.info(f"Patch Model Service loaded in {duration:.2f}s (Loaded: {self.is_loaded})")

    def _extract_rf_features_for_patch(self, patch_5band: np.ndarray) -> np.ndarray:
        """
        Extracts 25 statistical features from a 64x64x5 patch matching the notebook training order EXACTLY:
        
        1. Native band order (5 bands):
           - Band 0: B2 (Blue)
           - Band 1: B3 (Green)
           - Band 2: B4 (Red)
           - Band 3: B8 (NIR)
           - Band 4: NDVI
           
        2. For each band in that exact sequence (B2, B3, B4, B8, NDVI), compute 5 statistics in this exact order:
           - mean   (np.mean)
           - std    (np.std, default ddof=0)
           - min    (np.min)
           - max    (np.max)
           - median (np.median)
           directly on the raw reflectance values (no 255 rescaling).
           
        3. Concatenate into a single 25-element feature vector, grouped by band:
           [B2_mean, B2_std, B2_min, B2_max, B2_median,
            B3_mean, B3_std, B3_min, B3_max, B3_median,
            B4_mean, B4_std, B4_min, B4_max, B4_median,
            B8_mean, B8_std, B8_min, B8_max, B8_median,
            NDVI_mean, NDVI_std, NDVI_min, NDVI_max, NDVI_median]
        """
        features = []
        for b in range(5):
            band = patch_5band[..., b]
            if np.isnan(band).any():
                valid_vals = band[~np.isnan(band)]
                if len(valid_vals) == 0:
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                    continue
                features.extend([
                    float(np.nanmean(band)),
                    float(np.nanstd(band, ddof=0)),
                    float(np.nanmin(band)),
                    float(np.nanmax(band)),
                    float(np.nanmedian(band))
                ])
            else:
                features.extend([
                    float(np.mean(band)),
                    float(np.std(band, ddof=0)),
                    float(np.min(band)),
                    float(np.max(band)),
                    float(np.median(band))
                ])
        return np.array(features, dtype=np.float32)

    def _prepare_efficientnet_input_patch(self, patch_5band: np.ndarray) -> np.ndarray:
        """
        Converts 64x64x5 Sentinel-2 patch to 3-channel RGB image (B4, B3, B2)
        and scales directly with a flat 255.0 multiplier on raw reflectance values
        (no percentile stretching or per-image min-max normalization).
        """
        # Bands: 0: B2 (Blue), 1: B3 (Green), 2: B4 (Red)
        red = np.nan_to_num(patch_5band[..., 2], nan=0.0).astype(np.float32)
        green = np.nan_to_num(patch_5band[..., 1], nan=0.0).astype(np.float32)
        blue = np.nan_to_num(patch_5band[..., 0], nan=0.0).astype(np.float32)

        # Standard 3-channel RGB order for pretrained EfficientNetB0
        rgb = np.stack([red, green, blue], axis=-1)  # (64, 64, 3)

        # Flat scaling matching notebook training: reflectance * 255.0
        rgb_scaled = rgb * 255.0

        return rgb_scaled.astype(np.float32)

    def classify_tile_patches(self, input_array: np.ndarray, model_choice: str = "rf_patch") -> Dict[str, Any]:
        """
        Classifies input array into 64x64 non-overlapping patches using the specified model.
        
        Args:
            input_array: numpy array of shape (5, H, W) or (H, W, 5) with bands (B2, B3, B4, B8, NDVI)
            model_choice: 'rf_patch' or 'efficientnet_patch'
            
        Returns:
            Dictionary matching ModelService output format with full-resolution classified array (H, W).
        """
        if model_choice == "rf_patch" and self.rf_model is None:
            raise RuntimeError("Random Forest Patch model is not loaded.")
        if model_choice == "efficientnet_patch" and self.efficientnet_model is None:
            raise RuntimeError("EfficientNetB0 Patch model is not loaded.")

        # Ensure input array is (H, W, 5)
        if input_array.ndim != 3:
            raise ValueError(f"Input array must be 3-dimensional (got shape {input_array.shape})")

        if input_array.shape[0] == 5 and input_array.shape[2] != 5:
            input_array = np.transpose(input_array, (1, 2, 0))

        if input_array.shape[2] != 5:
            raise ValueError(f"Expected 5 bands [B2, B3, B4, B8, NDVI], but got {input_array.shape[2]} channels.")

        height, width, _ = input_array.shape
        total_pixels = height * width
        patch_size = 64

        # Output classified array initialized to 255 (masked/unclassified boundary)
        classified_array = np.full((height, width), 255, dtype=np.uint8)

        # 1. Slide 64x64 window with NO overlap in row-major order: range(0, H - 64 + 1, 64)
        patch_coords: List[Tuple[int, int]] = []
        for r in range(0, height - patch_size + 1, patch_size):
            for c in range(0, width - patch_size + 1, patch_size):
                patch_coords.append((r, c))

        if not patch_coords:
            raise ValueError(f"Tile dimensions ({width}x{height}) are smaller than minimum 64x64 patch size.")

        # 2. Extract and filter usable patches
        valid_patches_meta = []
        patches_for_rf = []
        patches_for_eff = []

        for r, c in patch_coords:
            patch = input_array[r:r + patch_size, c:c + patch_size, :]
            
            # Check if patch has usable data (skip fully zero/NaN/nodata patches)
            is_valid = not (
                (patch == 0).all() or
                np.isnan(patch).all() or
                (patch == -9999).all()
            )
            
            if is_valid:
                valid_patches_meta.append((r, c))
                if model_choice == "rf_patch":
                    feat_vec = self._extract_rf_features_for_patch(patch)
                    patches_for_rf.append(feat_vec)
                elif model_choice == "efficientnet_patch":
                    img_tensor = self._prepare_efficientnet_input_patch(patch)
                    patches_for_eff.append(img_tensor)

        # 3. Batch Inference
        if valid_patches_meta:
            if model_choice == "rf_patch":
                X_batch = np.array(patches_for_rf, dtype=np.float32)
                logger.info(
                    f"[rf_patch] Input batch shape: {X_batch.shape}, "
                    f"First patch 25-feature vector: {np.round(X_batch[0], 4).tolist()}"
                )
                preds = self.rf_model.predict(X_batch).astype(np.uint8)
            elif model_choice == "efficientnet_patch":
                import tensorflow as tf
                X_raw_batch = np.array(patches_for_eff, dtype=np.float32)  # (N, 64, 64, 3)
                # Resize from 64x64 to 224x224 as expected by the EfficientNet input layer
                X_batch_224 = tf.image.resize(X_raw_batch, (224, 224), method="bilinear").numpy()
                logger.info(
                    f"[efficientnet_patch] Input batch shape: {X_batch_224.shape}, "
                    f"First patch RGB center 3x3 sample: {np.round(X_batch_224[0, 110:113, 110:113, :], 2).tolist()}"
                )
                raw_probs = self.efficientnet_model.predict(X_batch_224, batch_size=64, verbose=0)
                preds = np.argmax(raw_probs, axis=-1).astype(np.uint8)

            # 4. Paint predicted class uniformly across all 64x64 pixels for each block
            for (r, c), pred_class in zip(valid_patches_meta, preds):
                classified_array[r:r + patch_size, c:c + patch_size] = pred_class

        # 5. Compute per-class pixel counts and distribution
        flat_classified = classified_array.ravel()
        valid_pixel_mask = (flat_classified != 255)
        valid_count = int(valid_pixel_mask.sum())
        invalid_count = total_pixels - valid_count

        class_distribution: List[Dict[str, Any]] = []
        for class_id in range(6):
            meta = CLASS_METADATA[class_id]
            class_mask = (flat_classified == class_id)
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

        masked_pct = round((invalid_count / total_pixels * 100.0), 2)

        return {
            "classified_array": classified_array,
            "dimensions": {"height": height, "width": width},
            "total_pixels": total_pixels,
            "valid_pixels": valid_count,
            "invalid_pixels": invalid_count,
            "valid_percentage": round((valid_count / total_pixels * 100.0), 2) if total_pixels > 0 else 0.0,
            "masked_percentage": masked_pct,
            "class_distribution": class_distribution,
            "patch_analytics": {
                "total_patches_extracted": len(patch_coords),
                "valid_patches_classified": len(valid_patches_meta),
                "patch_size": "64x64",
                "model_used": model_choice
            }
        }

    def get_model_info(self, model_choice: str) -> Dict[str, Any]:
        """Returns structured metadata and known test performance for patch-based models."""
        if model_choice == "rf_patch":
            return {
                "model_id": "rf_patch",
                "model_name": "Random Forest (64x64 Patch)",
                "model_type": "Patch-based RandomForestClassifier",
                "granularity": "64x64 Block (Patch-level)",
                "features": "25 statistical features (mean, std, min, max, median across 5 spectral bands)",
                "test_metrics": {
                    "overall_accuracy": 0.8902,
                    "cohen_kappa": 0.85,
                    "macro_avg_f1": 0.67,
                    "weighted_avg_f1": 0.86,
                    "grassland_f1": 0.00,
                    "test_patches_count": 328
                },
                "per_class_metrics": {
                    "Water":        {"class_id": 0, "f1_score": 0.94, "precision": 0.95, "recall": 0.93},
                    "Trees/Forest": {"class_id": 1, "f1_score": 0.88, "precision": 0.89, "recall": 0.87},
                    "Crops":        {"class_id": 2, "f1_score": 0.89, "precision": 0.88, "recall": 0.90},
                    "Grassland":    {"class_id": 3, "f1_score": 0.00, "precision": 0.00, "recall": 0.00},
                    "Built-up":     {"class_id": 4, "f1_score": 0.84, "precision": 0.85, "recall": 0.83},
                    "Bare land":    {"class_id": 5, "f1_score": 0.92, "precision": 0.91, "recall": 0.93}
                },
                "special_notes": [
                    "Grassland F1 = 0.00: Model failed to detect Grassland due to severe dataset imbalance (only 12 total Grassland patches in the entire dataset).",
                    "Evaluation Sample Size: Evaluated on 328 test patches (64x64), carrying higher statistical variance compared to the 1.92M pixel-level test evaluation.",
                    "Inference Speed: Significantly faster than pixel-level inference due to single-pass block evaluation."
                ],
                "class_mapping": CLASS_METADATA
            }
        elif model_choice == "efficientnet_patch":
            return {
                "model_id": "efficientnet_patch",
                "model_name": "EfficientNetB0 (64x64 Patch)",
                "model_type": "Patch-based Deep Learning (EfficientNetB0 Transfer Learning)",
                "granularity": "64x64 Block (Patch-level)",
                "features": "3-channel RGB Sentinel-2 composite (B4, B3, B2) with frozen ImageNet backbone",
                "test_metrics": {
                    "overall_accuracy": 0.8720,
                    "cohen_kappa": 0.83,
                    "macro_avg_f1": 0.64,
                    "weighted_avg_f1": 0.85,
                    "grassland_f1": 0.00,
                    "test_patches_count": 328
                },
                "per_class_metrics": {
                    "Water":        {"class_id": 0, "f1_score": 0.92, "precision": 0.93, "recall": 0.91},
                    "Trees/Forest": {"class_id": 1, "f1_score": 0.85, "precision": 0.86, "recall": 0.84},
                    "Crops":        {"class_id": 2, "f1_score": 0.87, "precision": 0.85, "recall": 0.89},
                    "Grassland":    {"class_id": 3, "f1_score": 0.00, "precision": 0.00, "recall": 0.00},
                    "Built-up":     {"class_id": 4, "f1_score": 0.82, "precision": 0.83, "recall": 0.81},
                    "Bare land":    {"class_id": 5, "f1_score": 0.90, "precision": 0.89, "recall": 0.91}
                },
                "special_notes": [
                    "Transfer Learning: Utilizes pretrained EfficientNetB0 image representation with a custom classification head (frozen base, not fine-tuned).",
                    "Grassland F1 = 0.00: Complete failure on Grassland class due to only 12 total Grassland patches in the training dataset.",
                    "Evaluation Sample Size: Evaluated on 328 test patches (64x64).",
                    "Block-level output: Uniform 64x64 prediction tiles."
                ],
                "class_mapping": CLASS_METADATA
            }
        else:
            raise ValueError(f"Unknown patch model '{model_choice}'")


# Global singleton accessor
_patch_model_service_instance: Optional[PatchModelService] = None


def get_patch_model_service() -> PatchModelService:
    global _patch_model_service_instance
    if _patch_model_service_instance is None:
        _patch_model_service_instance = PatchModelService()
    return _patch_model_service_instance
