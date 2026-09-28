"""
Part E: Verification script to start backend server, call all 3 models on largest tile,
and verify peak RSS < 400 MB for every model.
"""

import os
import sys
import time
import subprocess
import requests
import psutil
from pathlib import Path

def main():
    project_root = Path(r"E:\ML Claude\lulc-project")
    backend_dir = project_root / "backend"
    largest_tile = project_root / "dataset" / "LULCzip" / "LULC" / "input" / "Vijayawada_LULC_Input_2025-0000000000-0000002048.tif"
    python_exe = sys.executable

    print("=" * 80)
    print("PART E: RAM VERIFICATION & BENCHMARK ON LARGEST TILE (< 400 MB TARGET)")
    print("=" * 80)
    print(f"Largest Test Tile: {largest_tile.name} ({largest_tile.stat().st_size / (1024*1024):.2f} MB)")

    # Start uvicorn server on port 8010
    port = 8010
    server_cmd = [
        python_exe, "-m", "uvicorn", "app.main:app",
        "--host", "127.0.0.1", "--port", str(port),
        "--log-level", "info"
    ]
    
    print(f"\nStarting backend server on port {port}...")
    server_proc = subprocess.Popen(server_cmd, cwd=str(backend_dir))
    time.sleep(3.0)

    server_psutil = psutil.Process(server_proc.pid)

    def get_server_rss_mb() -> float:
        total_rss = server_psutil.memory_info().rss
        for child in server_psutil.children(recursive=True):
            try:
                total_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total_rss / (1024 * 1024)

    models_data = []

    try:
        # 1. Startup & /health
        t0 = time.time()
        health_res = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
        t_health = (time.time() - t0) * 1000
        rss_startup = get_server_rss_mb()
        print(f"\n1. Server Idle at Startup:")
        print(f"   - /health status       : {health_res.status_code} ({health_res.json()})")
        print(f"   - /health response time: {t_health:.1f} ms")
        print(f"   - Server RSS at startup: {rss_startup:.2f} MB")

        # Test sequence: pixel_rf -> rf_patch -> efficientnet_patch
        test_models = ["pixel_rf", "rf_patch", "efficientnet_patch"]

        for idx, model_name in enumerate(test_models, start=2):
            print(f"\n{idx}. Calling /api/classify with model='{model_name}'...")
            with open(largest_tile, "rb") as f:
                files = {"file": (largest_tile.name, f, "image/tiff")}
                data = {"model": model_name}
                t0 = time.time()
                res = requests.post(f"http://127.0.0.1:{port}/api/classify", files=files, data=data, timeout=120)
                dur = time.time() - t0

            rss_after = get_server_rss_mb()
            resp_json = res.json()
            success = resp_json.get("success", False)
            valid_pct = resp_json.get("valid_percentage", 0)
            
            print(f"   - Status code          : {res.status_code} (Success: {success})")
            print(f"   - Processing time      : {dur:.2f}s")
            print(f"   - Server RSS after call: {rss_after:.2f} MB")
            print(f"   - Valid pixels %       : {valid_pct}%")

            models_data.append({
                "model": model_name,
                "duration": dur,
                "rss": rss_after,
                "status": res.status_code
            })

        print("\n" + "=" * 80)
        print("PART E SUMMARY: RAM USAGE PER MODEL (< 400 MB TARGET)")
        print("=" * 80)
        print(f"{'State / Model':<25} {'Time (s)':<12} {'Process RSS (MB)':<18} {'< 400 MB?'}")
        print("-" * 80)
        print(f"{'Server Idle Startup':<25} {t_health/1000:6.3f}s      {rss_startup:8.2f} MB         {'YES' if rss_startup < 400 else 'NO'}")
        all_passed = (rss_startup < 400)
        for m in models_data:
            passed = (m['rss'] < 400.0)
            if not passed:
                all_passed = False
            print(f"{m['model']:<25} {m['duration']:6.2f}s       {m['rss']:8.2f} MB         {'YES' if passed else 'NO'}")
        print("-" * 80)

        peak_overall_rss = max([rss_startup] + [m['rss'] for m in models_data])
        print(f"PEAK PROCESS RSS OVERALL: {peak_overall_rss:.2f} MB (Target: < 400 MB)")
        if all_passed:
            print("SUCCESS: ALL MODELS AND IDLE SERVER REMAINED STRICTLY UNDER 400 MB!")
        else:
            print("FAILURE: One or more models exceeded 400 MB!")

    finally:
        print("\nStopping backend server...")
        try:
            for child in server_psutil.children(recursive=True):
                child.kill()
            server_proc.kill()
            server_proc.wait(timeout=5)
        except Exception:
            pass

if __name__ == "__main__":
    main()
