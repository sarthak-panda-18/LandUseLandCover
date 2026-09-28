"""
Verification script: confirms that chunked inference (250k pixels/batch)
produces identical predictions and class distribution on Tile 1 compared to unchunked inference.
"""

import sys
from pathlib import Path
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))
sys.path.insert(0, r"E:\ML Claude\lulc-project\backend")

from app.services.model_service import ModelService
from app.utils.raster_utils import read_geotiff_bytes

def verify():
    tile_path = Path(r"E:\ML Claude\lulc-project\dataset\LULCzip\LULC\input\Vijayawada_LULC_Input_2025-0000000000-0000000000.tif")
    print(f"Reading Tile 1: {tile_path} ({tile_path.stat().st_size / (1024*1024):.2f} MB)...")
    
    with open(tile_path, "rb") as f:
        file_bytes = f.read()

    raster_data, meta = read_geotiff_bytes(file_bytes)
    print(f"Raster shape: {raster_data.shape}, Metadata: {meta}")

    service = ModelService()
    service.ensure_loaded()

    # 1. Unchunked inference (batch size = total pixels)
    print("\nRunning UNCHUNKED inference...")
    res_unchunked = service.classify_tile(raster_data, batch_size=100_000_000)

    # 2. Chunked inference (batch size = 250,000)
    print("Running CHUNKED inference (batch_size=250,000)...")
    res_chunked = service.classify_tile(raster_data, batch_size=250_000)

    # Check array equality
    arr_unchunked = res_unchunked["classified_array"]
    arr_chunked = res_chunked["classified_array"]
    arrays_identical = np.array_equal(arr_unchunked, arr_chunked)
    diff_count = np.sum(arr_unchunked != arr_chunked)

    print("\n" + "=" * 70)
    print(f"TILE 1 COMPARISON: UNCHUNKED vs CHUNKED (250k)")
    print("=" * 70)
    print(f"Arrays Identical? : {arrays_identical}")
    print(f"Different Pixels  : {diff_count} (out of {arr_unchunked.size:,})")

    print("\nClass Distribution Comparison:")
    print(f"{'Class':<16} {'Unchunked Px':<15} {'Chunked Px':<15} {'Match?'}")
    print("-" * 60)
    for c1, c2 in zip(res_unchunked["class_distribution"], res_chunked["class_distribution"]):
        name = c1["class_name"]
        p1 = c1["pixel_count"]
        p2 = c2["pixel_count"]
        pct1 = c1["percentage_valid"]
        pct2 = c2["percentage_valid"]
        match = (p1 == p2 and pct1 == pct2)
        print(f"{name:<16} {p1:10,d} ({pct1:5.2f}%)   {p2:10,d} ({pct2:5.2f}%)   {match}")
    print("-" * 60)

    if arrays_identical:
        print("\nSUCCESS: Tile 1 predictions and distributions are 100% IDENTICAL.")
    else:
        print("\nFAILURE: Mismatch detected.")
        sys.exit(1)

if __name__ == "__main__":
    verify()
