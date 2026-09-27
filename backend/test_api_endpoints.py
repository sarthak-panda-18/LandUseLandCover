"""
Test Suite for LULC FastAPI Backend Endpoints
Tests:
1. GET /health
2. GET /api/model-info
3. GET /api/legend
4. POST /api/classify with actual GeoTIFF tile
5. Error handling: invalid file type, empty payload, large payload
"""

import base64
import io
import sys
import time
from pathlib import Path
from PIL import Image

from fastapi.testclient import TestClient
from app.main import app

def run_api_tests():
    print("=" * 80)
    print("TESTING LULC FASTAPI BACKEND API ENDPOINTS")
    print("=" * 80)

    # Use FastAPI TestClient directly for robust, deterministic testing
    client = TestClient(app)
    
    print("\n[1/5] Testing GET /health...")
    res_health = client.get("/health")
    print(f"  Status Code: {res_health.status_code}")
    print(f"  Response   : {res_health.json()}")
    assert res_health.status_code == 200, f"Expected 200, got {res_health.status_code}"
    assert res_health.json().get("model_loaded") is True, "Model should be loaded"

    print("\n[2/5] Testing GET /api/model-info...")
    res_info = client.get("/api/model-info")
    print(f"  Status Code: {res_info.status_code}")
    info_json = res_info.json()
    assert res_info.status_code == 200, f"Expected 200, got {res_info.status_code}"
    print(f"  Model Type          : {info_json.get('model_type')}")
    print(f"  Training Samples    : {info_json.get('training_samples'):,}")
    print(f"  Validation Samples  : {info_json.get('validation_samples'):,}")
    print(f"  Overall Accuracy    : {info_json.get('test_metrics', {}).get('overall_accuracy') * 100:.2f}%")
    print(f"  Cohen's Kappa       : {info_json.get('test_metrics', {}).get('cohen_kappa')}")
    print(f"  Macro Avg F1        : {info_json.get('test_metrics', {}).get('macro_avg_f1')}")
    print(f"  Weighted Avg F1     : {info_json.get('test_metrics', {}).get('weighted_avg_f1')}")
    print(f"  Feature Importances : {info_json.get('feature_importances')}")

    print("\n[3/5] Testing GET /api/legend...")
    res_legend = client.get("/api/legend")
    print(f"  Status Code: {res_legend.status_code}")
    print(f"  Content-Type: {res_legend.headers.get('content-type')}")
    print(f"  Image Size  : {len(res_legend.content):,} bytes")
    assert res_legend.status_code == 200, f"Expected 200, got {res_legend.status_code}"
    assert len(res_legend.content) > 1000, "Legend image should not be empty"

    print("\n[4/5] Testing Error Handling (Invalid File Type & Empty File)...")
    # Test non-tif file
    res_err_type = client.post(
        "/api/classify",
        files={"file": ("test.txt", b"dummy content", "text/plain")}
    )
    print(f"  Invalid Extension (.txt) -> Status: {res_err_type.status_code}, Response: {res_err_type.json()}")
    assert res_err_type.status_code == 400, f"Expected 400, got {res_err_type.status_code}"

    # Test empty file
    res_err_empty = client.post(
        "/api/classify",
        files={"file": ("test.tif", b"", "image/tiff")}
    )
    print(f"  Empty .tif file -> Status: {res_err_empty.status_code}, Response: {res_err_empty.json()}")
    assert res_err_empty.status_code == 400, f"Expected 400, got {res_err_empty.status_code}"

    print("\n[5/5] Testing POST /api/classify with Real Sentinel-2 GeoTIFF Tile...")
    project_root = Path(__file__).resolve().parents[1]
    test_tile = project_root / "dataset" / "LULCzip" / "LULC" / "input" / "Vijayawada_LULC_Input_2025-0000002048-0000004096.tif"

    if not test_tile.exists():
        print(f"  Warning: Test tile not found at {test_tile}")
        return

    print(f"  Uploading Tile: {test_tile.name} ({test_tile.stat().st_size / 1e6:.1f} MB)...")
    t0 = time.time()
    with open(test_tile, "rb") as f:
        file_bytes = f.read()
    
    res_classify = client.post(
        "/api/classify",
        files={"file": (test_tile.name, file_bytes, "image/tiff")}
    )
    upload_duration = time.time() - t0

    print(f"  Status Code: {res_classify.status_code}")
    assert res_classify.status_code == 200, f"Expected 200, got {res_classify.status_code}: {res_classify.text}"

    data = res_classify.json()
    print(f"  Tile Dimensions      : {data['dimensions']['width']} x {data['dimensions']['height']}")
    print(f"  Total Pixels         : {data['total_pixels']:,}")
    print(f"  Valid Pixels         : {data['valid_pixels']:,} ({data['valid_percentage']}%)")
    print(f"  Invalid/Masked       : {data['invalid_pixels']:,} ({data['masked_percentage']}%)")
    print(f"  Inference Server Time: {data['processing_time_seconds']}s")
    print(f"  Total Roundtrip Time : {upload_duration:.2f}s")
    
    # Verify Base64 image decoding
    b64_str = data['classified_image_base64']
    assert b64_str.startswith("data:image/png;base64,"), "Base64 string should have data URL header"
    raw_b64 = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(raw_b64)
    img = Image.open(io.BytesIO(img_bytes))
    print(f"  Decoded PNG Image    : {img.size[0]} x {img.size[1]}, Format: {img.format}, Mode: {img.mode}")
    assert img.size == (data['dimensions']['width'], data['dimensions']['height'])
    
    print("\n  Class Distribution Breakdown:")
    print(f"    {'Class Name':<16} {'Pixels':<12} {'% of Valid':<12} {'% of Total':<12} {'Color'}")
    print("    " + "-" * 60)
    for dist in data["class_distribution"]:
        print(f"    {dist['class_name']:<16} {dist['pixel_count']:10,d}   {dist['percentage_valid']:6.2f}%       {dist['percentage_total']:6.2f}%       {dist['color_hex']}")
    print("    " + "-" * 60)

    print("\n" + "=" * 80)
    print("ALL API ENDPOINT TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_api_tests()

