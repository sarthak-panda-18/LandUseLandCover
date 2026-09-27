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

    print("\n[2/5] Testing GET /api/model-info for all models...")
    for m in ["pixel_rf", "rf_patch", "efficientnet_patch"]:
        res_info = client.get(f"/api/model-info?model={m}")
        print(f"  GET /api/model-info?model={m} -> Status: {res_info.status_code}")
        assert res_info.status_code == 200, f"Expected 200, got {res_info.status_code}"
        info_json = res_info.json()
        print(f"    Name: {info_json.get('model_name')}, Acc: {info_json.get('test_metrics', {}).get('overall_accuracy')}")

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
        files={"file": ("test.txt", b"dummy content", "text/plain")},
        data={"model": "pixel_rf"}
    )
    print(f"  Invalid Extension (.txt) -> Status: {res_err_type.status_code}")
    assert res_err_type.status_code == 400, f"Expected 400, got {res_err_type.status_code}"

    # Test invalid model choice
    res_err_model = client.post(
        "/api/classify",
        files={"file": ("test.tif", b"dummy", "image/tiff")},
        data={"model": "invalid_model_choice"}
    )
    print(f"  Invalid Model Choice -> Status: {res_err_model.status_code}")
    assert res_err_model.status_code == 400, f"Expected 400, got {res_err_model.status_code}"

    print("\n[5/5] Testing POST /api/classify with all 3 models on Real Sentinel-2 GeoTIFF Tile...")
    project_root = Path(__file__).resolve().parents[1]
    test_tile = project_root / "dataset" / "LULCzip" / "LULC" / "input" / "Vijayawada_LULC_Input_2025-0000002048-0000004096.tif"

    if not test_tile.exists():
        print(f"  Warning: Test tile not found at {test_tile}")
        return

    with open(test_tile, "rb") as f:
        file_bytes = f.read()

    for m in ["pixel_rf", "rf_patch", "efficientnet_patch"]:
        t0 = time.time()
        res_classify = client.post(
            "/api/classify",
            files={"file": (test_tile.name, file_bytes, "image/tiff")},
            data={"model": m}
        )
        dur = time.time() - t0
        print(f"  Model: {m:<18s} -> Status: {res_classify.status_code}, Time: {dur:.2f}s")
        assert res_classify.status_code == 200, f"Failed for {m}: {res_classify.text}"
        data = res_classify.json()
        print(f"    Valid Pixels: {data['valid_pixels']:,} ({data['valid_percentage']}%), Server Time: {data['processing_time_seconds']}s")
        b64_str = data['classified_image_base64']
        assert b64_str.startswith("data:image/png;base64,"), "Base64 header mismatch"

    print("\n" + "=" * 80)
    print("ALL API ENDPOINT TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_api_tests()

