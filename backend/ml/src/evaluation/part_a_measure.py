"""
Part A: Baseline RAM benchmark on current codebase before any changes.
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
    print("PART A: BASELINE RAM MEASUREMENT (BEFORE CHANGES)")
    print("=" * 80)
    print(f"Largest Test Tile: {largest_tile.name} ({largest_tile.stat().st_size / (1024*1024):.2f} MB)")

    # Start uvicorn server on port 8009
    port = 8009
    server_cmd = [
        python_exe, "-m", "uvicorn", "app.main:app",
        "--host", "127.0.0.1", "--port", str(port),
        "--log-level", "warning"
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

    try:
        # 1. Startup
        rss_startup = get_server_rss_mb()
        health_res = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
        print(f"\n1. Server Idle at Startup:")
        print(f"   - /health status       : {health_res.status_code}")
        print(f"   - Server RSS at startup: {rss_startup:.2f} MB")

        # 2. Call pixel_rf
        print(f"\n2. Calling /api/classify with model='pixel_rf'...")
        with open(largest_tile, "rb") as f:
            files = {"file": (largest_tile.name, f, "image/tiff")}
            data = {"model": "pixel_rf"}
            t0 = time.time()
            res_pixel = requests.post(f"http://127.0.0.1:{port}/api/classify", files=files, data=data, timeout=60)
            t_pixel = time.time() - t0

        rss_after_pixel = get_server_rss_mb()
        print(f"   - Status code          : {res_pixel.status_code}")
        print(f"   - Classification time  : {t_pixel:.2f}s")
        print(f"   - Server RSS after call: {rss_after_pixel:.2f} MB (Delta from startup: +{rss_after_pixel - rss_startup:.2f} MB)")

        # 3. Call rf_patch
        print(f"\n3. Calling /api/classify with model='rf_patch'...")
        with open(largest_tile, "rb") as f:
            files = {"file": (largest_tile.name, f, "image/tiff")}
            data = {"model": "rf_patch"}
            t0 = time.time()
            res_rf_patch = requests.post(f"http://127.0.0.1:{port}/api/classify", files=files, data=data, timeout=60)
            t_rf_patch = time.time() - t0

        rss_after_rf_patch = get_server_rss_mb()
        print(f"   - Status code          : {res_rf_patch.status_code}")
        print(f"   - Classification time  : {t_rf_patch:.2f}s")
        print(f"   - Server RSS after call: {rss_after_rf_patch:.2f} MB (Delta from pixel_rf: +{rss_after_rf_patch - rss_after_pixel:.2f} MB)")

        # 4. Call efficientnet_patch
        print(f"\n4. Calling /api/classify with model='efficientnet_patch'...")
        with open(largest_tile, "rb") as f:
            files = {"file": (largest_tile.name, f, "image/tiff")}
            data = {"model": "efficientnet_patch"}
            t0 = time.time()
            res_eff = requests.post(f"http://127.0.0.1:{port}/api/classify", files=files, data=data, timeout=180)
            t_eff = time.time() - t0

        rss_after_eff = get_server_rss_mb()
        print(f"   - Status code          : {res_eff.status_code}")
        print(f"   - Classification time  : {t_eff:.2f}s")
        print(f"   - Server RSS after call: {rss_after_eff:.2f} MB (Delta from rf_patch: +{rss_after_eff - rss_after_rf_patch:.2f} MB)")

        # Summary
        print("\n" + "=" * 80)
        print("PART A SUMMARY: CURRENT BASELINE RSS")
        print("=" * 80)
        print(f"  Idle at Startup        : {rss_startup:7.2f} MB")
        print(f"  After pixel_rf         : {rss_after_pixel:7.2f} MB (+{rss_after_pixel - rss_startup:.2f} MB)")
        print(f"  After rf_patch         : {rss_after_rf_patch:7.2f} MB (+{rss_after_rf_patch - rss_after_pixel:.2f} MB)")
        print(f"  After efficientnet_patch: {rss_after_eff:7.2f} MB (+{rss_after_eff - rss_after_rf_patch:.2f} MB)")
        print("=" * 80)

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
