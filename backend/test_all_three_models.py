"""
Integration Test for all 3 LULC Classification Models:
1. pixel_rf (pixel-level Random Forest)
2. rf_patch (64x64 patch-level Random Forest)
3. efficientnet_patch (64x64 patch-level EfficientNetB0)
"""

import time
from pathlib import Path
import numpy as np

from app.services.model_service import get_model_service
from app.services.patch_model_service import get_patch_model_service
from app.utils.raster_utils import read_geotiff_bytes, classified_array_to_base64


def test_three_models():
    project_root = Path(__file__).resolve().parents[1]
    tile_path = project_root / "dataset" / "LULCzip" / "LULC" / "input" / "Vijayawada_LULC_Input_2025-0000000000-0000000000.tif"

    if not tile_path.exists():
        raise FileNotFoundError(f"Sample tile not found at {tile_path}")

    print(f"Reading test tile: {tile_path.name}...")
    with open(tile_path, "rb") as f:
        file_bytes = f.read()

    raster_data, geo_meta = read_geotiff_bytes(file_bytes)
    print(f"Raster shape: {raster_data.shape}, CRS: {geo_meta['crs']}, Dim: {geo_meta['width']}x{geo_meta['height']}")

    pixel_service = get_model_service()
    patch_service = get_patch_model_service()

    models_to_test = ["pixel_rf", "rf_patch", "efficientnet_patch"]
    results = {}

    for model_id in models_to_test:
        print("\n" + "=" * 80)
        print(f"RUNNING INFERENCE WITH MODEL: {model_id}")
        print("=" * 80)
        t0 = time.time()
        
        if model_id == "pixel_rf":
            res = pixel_service.classify_tile(raster_data)
        else:
            res = patch_service.classify_tile_patches(raster_data, model_choice=model_id)

        duration = time.time() - t0
        b64 = classified_array_to_base64(res["classified_array"])
        
        results[model_id] = {
            "duration": round(duration, 3),
            "valid_pixels": res["valid_pixels"],
            "valid_percentage": res["valid_percentage"],
            "masked_percentage": res["masked_percentage"],
            "class_distribution": res["class_distribution"],
            "b64_len": len(b64)
        }

        print(f"  Duration          : {duration:.3f} seconds")
        print(f"  Valid Land Pixels : {res['valid_pixels']:,} ({res['valid_percentage']}%)")
        print(f"  Masked Pixels     : {res['invalid_pixels']:,} ({res['masked_percentage']}%)")
        print(f"  Generated PNG B64 : {len(b64):,} chars")
        print("  Class Distribution:")
        for c in res["class_distribution"]:
            print(f"    - {c['class_name']:<14s}: {c['percentage_valid']:6.2f}% ({c['pixel_count']:,} px)")

    print("\n" + "=" * 80)
    print("SPEED & PROCESSING TIME COMPARISON")
    print("=" * 80)
    print(f"{'Model Choice':<24} {'Time (s)':<12} {'Speedup vs Pixel RF':<20}")
    print("-" * 60)
    base_time = results["pixel_rf"]["duration"]
    for model_id, r in results.items():
        speedup = f"{base_time / r['duration']:.2f}x faster" if r['duration'] < base_time else "Baseline (1.00x)"
        print(f"{model_id:<24} {r['duration']:<12.3f} {speedup:<20}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    test_three_models()
