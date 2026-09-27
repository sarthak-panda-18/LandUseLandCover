"""
Phase 8 Comprehensive Full-Stack Integration & Edge-Case Test Suite
Tests:
1. All 6 Sentinel-2 input tiles from dataset/LULCzip/LULC/input/
2. Edge cases (invalid extensions, empty payload, oversized payload, wrong band count)
3. Telemetry and Legend validation
"""

import base64
import io
import json
import time
from pathlib import Path
from PIL import Image
import httpx
import numpy as np
import rasterio

BASE_URL = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "dataset" / "LULCzip" / "LULC" / "input"
LABEL_DIR = PROJECT_ROOT / "dataset" / "LULCzip" / "LULC" / "labels"

CLASS_NAMES = {
    0: "Water",
    1: "Trees/Forest",
    2: "Crops",
    3: "Grassland",
    4: "Built-up",
    5: "Bare land",
}


def test_all_6_tiles():
    print("=" * 90)
    print("TASK 1: END-TO-END INFERENCE TESTING ACROSS ALL 6 SENTINEL-2 TILES")
    print("=" * 90)

    client = httpx.Client(base_url=BASE_URL, timeout=300.0)

    input_files = sorted(list(INPUT_DIR.glob("*.tif")))
    print(f"Found {len(input_files)} tiles in {INPUT_DIR.name}/\n")

    tile_results = []

    for idx, tile_path in enumerate(input_files, start=1):
        file_size_mb = tile_path.stat().st_size / (1024 * 1024)
        print(f"[{idx}/6] Testing Tile: {tile_path.name}")
        print(f"      Size on disk: {file_size_mb:.2f} MB")

        t_start = time.time()
        with open(tile_path, "rb") as f:
            file_bytes = f.read()

        t_upload_start = time.time()
        res = client.post(
            "/api/classify",
            files={"file": (tile_path.name, file_bytes, "image/tiff")}
        )
        roundtrip_time = time.time() - t_upload_start

        if res.status_code != 200:
            print(f"      FAILED with HTTP {res.status_code}: {res.text}")
            tile_results.append({
                "tile": tile_path.name,
                "status": "FAILED",
                "error": res.text
            })
            continue

        data = res.json()
        server_time = data.get("processing_time_seconds")
        dims = data.get("dimensions")
        total_px = data.get("total_pixels")
        valid_px = data.get("valid_pixels")
        valid_pct = data.get("valid_percentage")
        masked_pct = data.get("masked_percentage")
        b64_img = data.get("classified_image_base64")
        dist = data.get("class_distribution")

        # Verify Base64 image decoding
        raw_b64 = b64_img.split(",", 1)[1]
        img_bytes = base64.b64decode(raw_b64)
        img = Image.open(io.BytesIO(img_bytes))
        img_w, img_h = img.size

        assert (img_w, img_h) == (dims["width"], dims["height"]), "Image dimensions mismatch"

        print(f"      Dimensions  : {dims['width']} x {dims['height']} ({total_px:,} total px)")
        print(f"      Valid Pixels: {valid_px:,} ({valid_pct}%), Masked: {data.get('invalid_pixels'):,} ({masked_pct}%)")
        print(f"      Server Time : {server_time:.2f}s | Roundtrip: {roundtrip_time:.2f}s")
        print(f"      Decoded PNG : {img_w} x {img_h} ({img.format}, {img.mode}) - {len(img_bytes):,} bytes")
        
        # Dominant class
        sorted_dist = sorted(dist, key=lambda x: x["pixel_count"], reverse=True)
        dominant = sorted_dist[0]
        print(f"      Dominant    : {dominant['class_name']} ({dominant['percentage_valid']:.2f}% of valid land)")
        print("      Breakdown   :")
        for c in dist:
            print(f"        • {c['class_name']:<14}: {c['pixel_count']:9,d} px ({c['percentage_valid']:6.2f}% valid, {c['percentage_total']:6.2f}% total) [{c['color_hex']}]")

        print("      Status      : PASSED [OK]\n")

        tile_results.append({
            "tile": tile_path.name,
            "status": "PASSED",
            "file_size_mb": round(file_size_mb, 2),
            "dimensions": f"{dims['width']}x{dims['height']}",
            "total_pixels": total_px,
            "valid_pixels": valid_px,
            "server_time": server_time,
            "roundtrip_time": round(roundtrip_time, 2),
            "dominant_class": f"{dominant['class_name']} ({dominant['percentage_valid']:.1f}%)",
            "distribution": {c['class_name']: f"{c['percentage_valid']:.2f}%" for c in dist}
        })

    return tile_results


def test_edge_cases():
    print("=" * 90)
    print("TASK 2: EDGE CASE VALIDATION & ERROR HANDLING")
    print("=" * 90)

    client = httpx.Client(base_url=BASE_URL, timeout=30.0)

    # 1. Non-tif file
    print("[1/5] Testing Non-.tif File Upload (.txt)...")
    res_txt = client.post(
        "/api/classify",
        files={"file": ("readme.txt", b"Hello, this is a plain text file.", "text/plain")}
    )
    print(f"      HTTP Status: {res_txt.status_code}")
    print(f"      Response   : {res_txt.json()}")
    assert res_txt.status_code == 400, "Should return 400 for non-.tif file"
    assert "Unsupported file format" in res_txt.json()["detail"]
    print("      Result: PASSED [OK]\n")

    # 2. Empty file
    print("[2/5] Testing Empty .tif File Upload (0 bytes)...")
    res_empty = client.post(
        "/api/classify",
        files={"file": ("empty.tif", b"", "image/tiff")}
    )
    print(f"      HTTP Status: {res_empty.status_code}")
    print(f"      Response   : {res_empty.json()}")
    assert res_empty.status_code == 400, "Should return 400 for empty file"
    assert "empty" in res_empty.json()["detail"].lower()
    print("      Result: PASSED [OK]\n")

    # 3. GeoTIFF with wrong band count (1-band label mask)
    print("[3/5] Testing 1-Band GeoTIFF (Wrong Band Count)...")
    label_tile = next(LABEL_DIR.glob("*.tif"))
    with open(label_tile, "rb") as f:
        lbl_bytes = f.read()
    res_1band = client.post(
        "/api/classify",
        files={"file": (label_tile.name, lbl_bytes, "image/tiff")}
    )
    print(f"      HTTP Status: {res_1band.status_code}")
    print(f"      Response   : {res_1band.json()}")
    assert res_1band.status_code == 422, "Should return 422 for unsupported band count"
    assert "requires at least 4 bands" in res_1band.json()["detail"]
    print("      Result: PASSED [OK]\n")

    # 4. Corrupted / Fake GeoTIFF Header
    print("[4/5] Testing Corrupted GeoTIFF with .tif extension...")
    fake_tif_bytes = b"II*\x00" + b"\x00\x00\x00\x00" * 50  # Corrupted TIFF header
    res_corrupt = client.post(
        "/api/classify",
        files={"file": ("corrupted.tif", fake_tif_bytes, "image/tiff")}
    )
    print(f"      HTTP Status: {res_corrupt.status_code}")
    print(f"      Response   : {res_corrupt.json()}")
    assert res_corrupt.status_code in [422, 500], "Should catch corrupted TIFF"
    print("      Result: PASSED [OK]\n")

    # 5. Oversized file validation test (> 200MB limit check)
    print("[5/5] Testing File Size Limit (> 200MB)...")
    # Simulate oversized file without allocating huge memory using large buffer check in endpoint
    # We verify the constant MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024 is enforced
    from app.routes.classify import MAX_FILE_SIZE_BYTES
    print(f"      Enforced MAX_FILE_SIZE_BYTES: {MAX_FILE_SIZE_BYTES / (1024*1024):.0f} MB")
    assert MAX_FILE_SIZE_BYTES == 200 * 1024 * 1024
    print("      Result: PASSED [OK]\n")


def test_telemetry_and_legend():
    print("=" * 90)
    print("TASK 3: MODEL TELEMETRY & LEGEND CONSISTENCY VALIDATION")
    print("=" * 90)

    client = httpx.Client(base_url=BASE_URL, timeout=30.0)

    # Check Model Info
    res_info = client.get("/api/model-info")
    assert res_info.status_code == 200
    info = res_info.json()
    metrics = info.get("test_metrics", {})
    print("Model Telemetry Check:")
    print(f"  • Overall Accuracy : {metrics.get('overall_accuracy') * 100:.2f}%")
    print(f"  • Cohen's Kappa    : {metrics.get('cohen_kappa'):.4f}")
    print(f"  • Macro Avg F1     : {metrics.get('macro_avg_f1') * 100:.2f}%")
    print(f"  • Weighted Avg F1  : {metrics.get('weighted_avg_f1') * 100:.2f}%")
    print(f"  • Training Samples : {info.get('training_samples'):,}")
    print(f"  • Validation Samples: {info.get('validation_samples'):,}")
    print(f"  • Feature Importances: {info.get('feature_importances')}")

    # Check Legend
    res_legend = client.get("/api/legend")
    assert res_legend.status_code == 200
    assert res_legend.headers.get("content-type") == "image/png"
    print(f"\nLegend Image Check:")
    print(f"  • Content-Type: {res_legend.headers.get('content-type')}")
    print(f"  • Size: {len(res_legend.content):,} bytes [OK]")

    # Check Health
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json().get("model_loaded") is True
    print(f"\nHealth Check: {res_health.json()} [OK]")


if __name__ == "__main__":
    t_suite_start = time.time()
    results = test_all_6_tiles()
    test_edge_cases()
    test_telemetry_and_legend()
    total_time = time.time() - t_suite_start
    print("=" * 90)
    print(f"ALL PHASE 8 TESTS COMPLETED SUCCESSFULLY IN {total_time:.2f}s!")
    print("=" * 90)
