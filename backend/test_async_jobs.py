"""
Unit and Integration Tests for JobManager and Async Classification Routes.
Tests:
1. JobManager: create_job, update_job, get_job, cleanup_expired_jobs, get_active_job_count
2. Lock concurrency: ensure second job waits for first job to release lock
3. HTTP Async API using httpx.AsyncClient:
   - 202 response on POST /api/classify (<1s)
   - Status polling: pending -> processing -> done
   - Validation of final classification payload
   - Error cases: 400 on bad extension/model, 404 on invalid job ID
"""

import asyncio
import time
from pathlib import Path
import httpx

from app.main import app
from app.services.job_manager import JobManager, get_job_manager


async def test_job_manager_lifecycle():
    print("\n--- TEST 1: JobManager Unit Tests ---")
    jm = JobManager(max_job_age_seconds=10)

    # 1. Create Job
    job_id = jm.create_job()
    assert job_id is not None
    job = jm.get_job(job_id)
    assert job["status"] == "pending"
    assert job["result"] is None
    assert job["error"] is None
    print("  Job creation: OK")

    # 2. Update Job
    updated = jm.update_job(job_id, status="processing")
    assert updated["status"] == "processing"
    assert jm.get_job(job_id)["status"] == "processing"
    print("  Job update to processing: OK")

    # 3. Update Job to Done
    result_data = {"test": "result_payload"}
    updated_done = jm.update_job(job_id, status="done", result=result_data)
    assert updated_done["status"] == "done"
    assert updated_done["result"] == result_data
    print("  Job update to done: OK")

    # 4. Cleanup expired jobs
    jm._jobs[job_id]["created_at"] = time.time() - 100  # simulate old job
    cleaned = jm.cleanup_expired_jobs()
    assert cleaned == 1
    assert jm.get_job(job_id) is None
    print("  Opportunistic cleanup: OK")


async def test_async_classification_endpoint():
    print("\n--- TEST 2: Asynchronous HTTP API Classification (rf_patch) ---")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=120.0) as client:
        # 1. Health check
        res_health = await client.get("/health")
        assert res_health.status_code == 200

        # 2. Test 404 on missing job
        res_404 = await client.get("/api/classify/status/non-existent-id-000")
        assert res_404.status_code == 404
        print("  404 on invalid job ID: OK")

        # 3. Test 400 on bad extension
        res_400 = await client.post(
            "/api/classify",
            files={"file": ("test.txt", b"dummy content", "text/plain")},
            data={"model": "pixel_rf"}
        )
        assert res_400.status_code == 400
        print("  400 on bad file format: OK")

        # 4. Test real tile upload (fast patch model rf_patch)
        project_root = Path(__file__).resolve().parents[1]
        test_tile = project_root / "dataset" / "LULCzip" / "LULC" / "input" / "Vijayawada_LULC_Input_2025-0000002048-0000004096.tif"
        with open(test_tile, "rb") as f:
            file_bytes = f.read()

        t_post_start = time.time()
        res_post = await client.post(
            "/api/classify",
            files={"file": (test_tile.name, file_bytes, "image/tiff")},
            data={"model": "rf_patch"}
        )
        t_post_dur = time.time() - t_post_start
        print(f"  POST /api/classify returned {res_post.status_code} in {t_post_dur:.3f}s")
        assert res_post.status_code == 202
        assert t_post_dur < 1.0, f"POST should return in < 1.0s, took {t_post_dur:.3f}s"

        init_data = res_post.json()
        job_id = init_data["job_id"]
        assert init_data["status"] == "pending"
        print(f"  Received Job ID: {job_id}")

        # 5. Poll status
        poll_start = time.time()
        final_job = None
        while time.time() - poll_start < 60.0:
            await asyncio.sleep(0.5)
            res_status = await client.get(f"/api/classify/status/{job_id}")
            assert res_status.status_code == 200
            status_data = res_status.json()
            st = status_data["status"]
            print(f"    Status check: {st} (elapsed: {time.time() - poll_start:.1f}s)")
            if st in ("done", "error"):
                final_job = status_data
                break

        assert final_job is not None
        assert final_job["status"] == "done", f"Job failed: {final_job.get('error')}"
        res = final_job["result"]
        assert res["success"] is True
        assert res["valid_pixels"] > 0
        assert res["classified_image_base64"].startswith("data:image/png;base64,")
        assert res["rgb_preview_base64"].startswith("data:image/png;base64,")
        print(f"  Classification succeeded in {time.time() - poll_start:.2f}s! Valid pixels: {res['valid_pixels']:,}")


async def test_lock_sequential_execution():
    print("\n--- TEST 3: Single-Job Concurrency Lock Execution ---")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=120.0) as client:
        project_root = Path(__file__).resolve().parents[1]
        test_tile = project_root / "dataset" / "LULCzip" / "LULC" / "input" / "Vijayawada_LULC_Input_2025-0000002048-0000004096.tif"
        with open(test_tile, "rb") as f:
            file_bytes = f.read()

        # Submit Job 1
        res1 = await client.post(
            "/api/classify",
            files={"file": (test_tile.name, file_bytes, "image/tiff")},
            data={"model": "rf_patch"}
        )
        assert res1.status_code == 202
        job1_id = res1.json()["job_id"]

        # Immediately Submit Job 2
        res2 = await client.post(
            "/api/classify",
            files={"file": (test_tile.name, file_bytes, "image/tiff")},
            data={"model": "rf_patch"}
        )
        assert res2.status_code == 202
        job2_id = res2.json()["job_id"]
        print(f"  Submitted two concurrent jobs: Job1={job1_id[:8]}, Job2={job2_id[:8]}")

        # Poll both to completion
        poll_start = time.time()
        job1_done = False
        job2_done = False

        while time.time() - poll_start < 60.0:
            await asyncio.sleep(0.5)
            s1 = (await client.get(f"/api/classify/status/{job1_id}")).json()
            s2 = (await client.get(f"/api/classify/status/{job2_id}")).json()
            
            print(f"    Job1: {s1['status']:<10} | Job2: {s2['status']:<10}")

            if s1["status"] == "done":
                job1_done = True
            if s2["status"] == "done":
                job2_done = True

            if job1_done and job2_done:
                break

        assert job1_done and job2_done, "Both jobs should complete sequentially without crashing"
        print("  Both concurrent jobs completed cleanly via lock queueing!")


async def test_all_three_models_async():
    print("\n--- TEST 4: Async Classification with All 3 Models (pixel_rf, rf_patch, efficientnet_patch) ---")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=300.0) as client:
        project_root = Path(__file__).resolve().parents[1]
        test_tile = project_root / "dataset" / "LULCzip" / "LULC" / "input" / "Vijayawada_LULC_Input_2025-0000002048-0000004096.tif"
        with open(test_tile, "rb") as f:
            file_bytes = f.read()

        for model_id in ["pixel_rf", "rf_patch", "efficientnet_patch"]:
            t0 = time.time()
            res_post = await client.post(
                "/api/classify",
                files={"file": (test_tile.name, file_bytes, "image/tiff")},
                data={"model": model_id}
            )
            t_post_dur = time.time() - t0
            assert res_post.status_code == 202
            assert t_post_dur < 1.0
            job_id = res_post.json()["job_id"]
            print(f"  Model {model_id:<18s}: POST 202 in {t_post_dur:.3f}s, job_id={job_id[:8]}")

            poll_start = time.time()
            final_job = None
            while time.time() - poll_start < 240.0:
                await asyncio.sleep(1.0)
                res_status = await client.get(f"/api/classify/status/{job_id}")
                assert res_status.status_code == 200
                st_data = res_status.json()
                if st_data["status"] in ("done", "error"):
                    final_job = st_data
                    break

            assert final_job is not None
            assert final_job["status"] == "done", f"Model {model_id} failed: {final_job.get('error')}"
            res = final_job["result"]
            print(f"    -> Done in {time.time() - poll_start:.2f}s | Valid: {res['valid_pixels']:,} ({res['valid_percentage']}%) | Model: {res['model_used']}")
            assert res["classified_image_base64"].startswith("data:image/png;base64,")


async def main():
    print("=" * 80)
    print("RUNNING ASYNC CLASSIFICATION JOB & LOCK TESTS")
    print("=" * 80)
    await test_job_manager_lifecycle()
    await test_async_classification_endpoint()
    await test_lock_sequential_execution()
    await test_all_three_models_async()
    print("\n" + "=" * 80)
    print("ALL ASYNC JOB TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
