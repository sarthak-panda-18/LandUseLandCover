"""
Direct End-to-End Live HTTP Test against running server at http://127.0.0.1:8000
"""

import time
from pathlib import Path
import httpx


def test_live_server():
    client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=120.0)

    print("Checking /health...")
    h = client.get("/health")
    print(f"Health: {h.status_code}, {h.json()}")
    assert h.status_code == 200

    print("\nFetching Sample Tiles...")
    st = client.get("/api/sample-tiles")
    print(f"Sample Tiles: {st.status_code}, count: {len(st.json())}")
    assert st.status_code == 200
    tile_filename = st.json()[5]["filename"]  # Tile 6 (29.6 MB)
    print(f"Downloading sample tile bytes for {tile_filename}...")
    sample_bytes_res = client.get(f"/api/sample-tiles/{tile_filename}")
    assert sample_bytes_res.status_code == 200
    file_bytes = sample_bytes_res.content
    print(f"Downloaded {len(file_bytes):,} bytes.")

    print("\nSubmitting async classification job for rf_patch...")
    t0 = time.time()
    res_post = client.post(
        "/api/classify",
        files={"file": (tile_filename, file_bytes, "image/tiff")},
        data={"model": "rf_patch"}
    )
    post_dur = time.time() - t0
    print(f"POST returned {res_post.status_code} in {post_dur:.3f}s: {res_post.json()}")
    assert res_post.status_code == 202
    assert post_dur < 1.0, f"Expected < 1s, got {post_dur:.3f}s"
    job_id = res_post.json()["job_id"]

    print("\nPolling /api/classify/status/{job_id}...")
    poll_start = time.time()
    final_job = None
    while time.time() - poll_start < 60.0:
        time.sleep(1.0)
        res_status = client.get(f"/api/classify/status/{job_id}")
        assert res_status.status_code == 200
        st_data = res_status.json()
        print(f"  Status after {time.time() - poll_start:.1f}s: {st_data['status']}")
        if st_data["status"] in ("done", "error"):
            final_job = st_data
            break

    assert final_job is not None
    assert final_job["status"] == "done", f"Job failed: {final_job.get('error')}"
    result = final_job["result"]
    print(f"\nClassification Result received successfully in {time.time() - poll_start:.2f}s!")
    print(f"  Model: {result['model_used']}")
    print(f"  Dimensions: {result['dimensions']}")
    print(f"  Valid Pixels: {result['valid_pixels']:,} ({result['valid_percentage']}%)")
    print(f"  Processing Time: {result['processing_time_seconds']}s")
    print(f"  Class Distribution:")
    for c in result["class_distribution"]:
        print(f"    - {c['class_name']}: {c['percentage_valid']:.2f}% ({c['pixel_count']:,} px)")
    print("\nALL LIVE SERVER TESTS PASSED!")


if __name__ == "__main__":
    test_live_server()
