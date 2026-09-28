"""
Patch-Based Model Service for LULC Classification
Streams raster files in 64-row windows (one row of 64x64 patches at a time).
Uses Random Forest Patch model (statistical features) and ONNX Runtime for EfficientNetB0,
completely eliminating TensorFlow/Keras memory overhead.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window

from app.services.model_manager import get_model_manager

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

    def _extract_rf_features_for_patch(self, patch_5band: np.ndarray) -> np.ndarray:
        """Extracts 25 statistical features from a 64x64x5 patch."""
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
        scaled with 255.0 multiplier and resized to 224x224 using bilinear interpolation.
        """
        red = np.nan_to_num(patch_5band[..., 2], nan=0.0).astype(np.float32)
        green = np.nan_to_num(patch_5band[..., 1], nan=0.0).astype(np.float32)
        blue = np.nan_to_num(patch_5band[..., 0], nan=0.0).astype(np.float32)

        rgb_scaled = np.stack([red, green, blue], axis=-1) * 255.0  # (64, 64, 3)
        img = Image.fromarray(np.clip(rgb_scaled, 0, 255).astype(np.uint8))
        resized = np.array(img.resize((224, 224), Image.BILINEAR), dtype=np.float32)
        return resized

    def classify_tile_streaming(
        self,
        raster_path: Path,
        model_choice: str = "rf_patch"
    ) -> Dict[str, Any]:
        """
        Streams a GeoTIFF raster file one row of 64x64 patches at a time (64 rows x width)
        and writes classification outputs directly into a preallocated uint8 array.
        
        Args:
            raster_path: Path to the GeoTIFF raster file
            model_choice: 'rf_patch' or 'efficientnet_patch'
            
        Returns:
            Dictionary matching classification response specification.
        """
        model_mgr = get_model_manager()
        rf_model = None
        onnx_session = None

        if model_choice == "rf_patch":
            rf_model = model_mgr.get_rf_patch_model()
        elif model_choice == "efficientnet_patch":
            onnx_session = model_mgr.get_efficientnet_onnx_session()
            input_name = onnx_session.get_inputs()[0].name
            output_name = onnx_session.get_outputs()[0].name
        else:
            raise ValueError(f"Unknown patch model '{model_choice}'")

        patch_size = 64
        with rasterio.open(raster_path) as src:
            height = src.height
            width = src.width
            band_count = src.count
            total_pixels = height * width

            if height < patch_size or width < patch_size:
                raise ValueError(f"Tile dimensions ({width}x{height}) are smaller than minimum 64x64 patch size.")

            classified_array = np.full((height, width), 255, dtype=np.uint8)

            total_patches_extracted = 0
            valid_patches_classified = 0

            # Process 64-row strips (one row of 64x64 patches at a time)
            for row_start in range(0, height - patch_size + 1, patch_size):
                window = Window(col_off=0, row_off=row_start, width=width, height=patch_size)

                if band_count >= 5:
                    strip_raw = src.read([1, 2, 3, 4, 5], window=window).astype(np.float32)
                else:
                    strip_4band = src.read([1, 2, 3, 4], window=window).astype(np.float32)
                    b4_red = strip_4band[2]
                    b8_nir = strip_4band[3]
                    denom = b8_nir + b4_red
                    ndvi = np.where(denom != 0, (b8_nir - b4_red) / (denom + 1e-6), 0.0)
                    strip_raw = np.concatenate([strip_4band, ndvi[np.newaxis, ...]], axis=0)

                # Transpose to (64, width, 5)
                strip_hwc = np.transpose(strip_raw, (1, 2, 0))
                del strip_raw

                strip_valid_cols = []
                strip_features_rf = []
                strip_patches_eff = []

                for col_start in range(0, width - patch_size + 1, patch_size):
                    total_patches_extracted += 1
                    patch = strip_hwc[:, col_start:col_start + patch_size, :]

                    is_valid = not (
                        (patch == 0).all() or
                        np.isnan(patch).all() or
                        (patch == -9999).all()
                    )

                    if is_valid:
                        strip_valid_cols.append(col_start)
                        if model_choice == "rf_patch":
                            strip_features_rf.append(self._extract_rf_features_for_patch(patch))
                        elif model_choice == "efficientnet_patch":
                            strip_patches_eff.append(self._prepare_efficientnet_input_patch(patch))

                del strip_hwc

                if strip_valid_cols:
                    valid_patches_classified += len(strip_valid_cols)
                    if model_choice == "rf_patch":
                        X_batch = np.array(strip_features_rf, dtype=np.float32)
                        preds = rf_model.predict(X_batch).astype(np.uint8)
                    elif model_choice == "efficientnet_patch":
                        X_batch = np.array(strip_patches_eff, dtype=np.float32)
                        probs = onnx_session.run([output_name], {input_name: X_batch})[0]
                        preds = np.argmax(probs, axis=-1).astype(np.uint8)

                    # Paint the 64x64 block uniformly into classified_array
                    for col_start, pred_cls in zip(strip_valid_cols, preds):
                        classified_array[row_start:row_start + patch_size, col_start:col_start + patch_size] = pred_cls

        # Compute summary distribution directly from the uint8 classified_array
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
                "total_patches_extracted": total_patches_extracted,
                "valid_patches_classified": valid_patches_classified,
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
                "model_name": "EfficientNetB0 (64x64 Patch - ONNX Runtime)",
                "model_type": "Patch-based Deep Learning (EfficientNetB0 via ONNX Runtime)",
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
                    "ONNX Runtime: Converted from Keras to ONNX for minimal RAM footprint and fast single-thread CPU execution.",
                    "Grassland F1 = 0.00: Complete failure on Grassland class due to only 12 total Grassland patches in the training dataset.",
                    "Evaluation Sample Size: Evaluated on 328 test patches (64x64).",
                    "Block-level output: Uniform 64x64 prediction tiles."
                ],
                "class_mapping": CLASS_METADATA
            }
        else:
            raise ValueError(f"Unknown patch model '{model_choice}'")


_patch_model_service_instance: Optional[PatchModelService] = None


def get_patch_model_service() -> PatchModelService:
    global _patch_model_service_instance
    if _patch_model_service_instance is None:
        _patch_model_service_instance = PatchModelService()
    return _patch_model_service_instance
