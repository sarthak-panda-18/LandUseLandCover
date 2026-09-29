"""
Test Suite for LULC FastAPI Backend Endpoints (Async Job Flow)
Tests:
1. GET /health
2. GET /api/model-info
3. GET /api/legend
4. Error handling: invalid file type, empty payload, invalid model choice, non-existent job ID
5. POST /api/classify (202 Accepted) & GET /api/classify/status/{job_id} async polling with all 3 models
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
    print("TESTING LULC FASTAPI BACKEND API ENDPOINTS (ASYNC JOB PATTERN)")
    print("=" * 80)

    # Use FastAPI TestClient directly for robust, deterministic testing
    client = TestClient(app)
    
    print("\n[1/6] Testing GET /health...")
    res_health = client.get("/health")
    print(f"  Status Code: {res_health.status_code}")
    print(f"  Response   : {res_health.json()}")
    assert res_health.status_code == 200, f"Expected 200, got {res_health.status_code}"
    assert res_health.json().get("model_loaded") is True, "Model should be loaded"

    print("\n[2/6] Testing GET /api/model-info for all models...")
    for m in ["pixel_rf", "rf_patch", "efficientnet_patch"]:
        res_info = client.get(f"/api/model-info?model={m}")
        print(f"  GET /api/model-info?model={m} -> Status: {res_info.status_code}")
        assert res_info.status_code == 200, f"Expected 200, got {res_info.status_code}"
        info_json = res_info.json()
        print(f"    Name: {info_json.get('model_name')}, Acc: {info_json.get('test_metrics', {}).get('overall_accuracy')}")

    print("\n[3/6] Testing GET /api/legend...")
    res_legend = client.get("/api/legend")
    print(f"  Status Code: {res_legend.status_code}")
    print(f"  Content-Type: {res_legend.headers.get('content-type')}")
    print(f"  Image Size  : {len(res_legend.content):,} bytes")
    assert res_legend.status_code == 200, f"Expected 200, got {res_legend.status_code}"
    assert len(res_legend.content) > 1000, "Legend image should not be empty"

    print("\n[4/6] Testing Error Handling (Invalid File Type, Invalid Model & 404 Job Status)...")
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

    # Test 404 on non-existent job ID
    res_err_job = client.get("/api/classify/status/non-existent-uuid-12345")
    print(f"  GET /api/classify/status/non-existent-uuid -> Status: {res_err_job.status_code}")
    assert res_err_job.status_code == 404, f"Expected 404, got {res_err_job.status_code}"

    print("\n[5/6] Testing POST /api/classify (202 Accepted) & Polling GET /api/classify/status/{job_id}...")
    project_root = Path(__file__).resolve().parents[1]
    test_tile = project_root / "dataset" / "LULCzip" / "LULC" / "input" / "Vijayawada_LULC_Input_2025-0000002048-0000004096.tif"

    if not test_tile.exists():
        print(f"  Warning: Test tile not found at {test_tile}")
        return

    with open(test_tile, "rb") as f:
        file_bytes = f.read()

    for m in ["pixel_rf", "rf_patch", "efficientnet_patch"]:
        t0 = time.time()
        # 1. Post tile to start async job
        res_init = client.post(
            "/api/classify",
            files={"file": (test_tile.name, file_bytes, "image/tiff")},
            data={"model": m}
        )
        t_init = time.time() - t0
        print(f"\n  Model: {m:<18s} -> POST 202 in {t_init:.3f}s (Status: {res_init.status_code})")
        assert res_init.status_code == 202, f"Failed init for {m}: {res_init.text}"
        
        init_data = res_init.json()
        job_id = init_data["job_id"]
        assert init_data["status"] == "pending", f"Expected 'pending', got {init_data['status']}"
        assert job_id, "job_id must not be empty"

        # 2. Poll job status until done
        max_wait = 180.0
        poll_start = time.time()
        job_status = "pending"
        final_job = None

        while (time.time() - poll_start) < max_wait:
            time.sleep(1.0)
            res_status = client.get(f"/api/classify/status/{job_id}")
            assert res_status.status_code == 200, f"Failed status poll: {res_status.text}"
            status_data = res_status.json()
            job_status = status_data.get("status")
            print(f"    Polling status for job {job_id[:8]}... -> {job_status}")
            if job_status in ("done", "error"):
                final_job = status_data
                break

        assert job_status == "done", f"Job failed or timed out: {final_job}"
        assert final_job is not None
        data = final_job["result"]
        print(f"    Completed in {time.time() - poll_start:.2f}s | Valid Pixels: {data['valid_pixels']:,} ({data['valid_percentage']}%)")
        b64_str = data['classified_image_base64']
        assert b64_str.startswith("data:image/png;base64,"), "Base64 header mismatch"

    print("\n[6/6] Testing Sample Tiles API...")
    res_samples = client.get("/api/sample-tiles")
    assert res_samples.status_code == 200
    samples = res_samples.json()
    print(f"  Found {len(samples)} sample tiles available.")
    assert len(samples) == 6

    print("\n" + "=" * 80)
    print("ALL API ENDPOINT TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_api_tests()
