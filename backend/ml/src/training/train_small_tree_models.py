"""
Train and benchmark smaller tree models (30 to 50 trees) for extreme memory reduction (<120MB load delta).
"""

import os
import sys
import gc
import time
import subprocess
from pathlib import Path
import joblib
import numpy as np
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

def measure_load_ram_mb(model_path: Path) -> float:
    code = f"""
import os, psutil, gc, joblib
proc = psutil.Process(os.getpid())
gc.collect()
rss_before = proc.memory_info().rss / (1024 * 1024)
model = joblib.load(r'{model_path}')
gc.collect()
rss_after = proc.memory_info().rss / (1024 * 1024)
print(f"{{rss_after - rss_before:.2f}}")
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return float(res.stdout.strip())

def main():
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"
    tmp_dir = models_dir / "compact_experiments"

    train_data = np.load(data_dir / "train.npz")
    test_data = np.load(data_dir / "test.npz")
    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    configs = [
        {"id": "cfg_40t_1500leaf", "n_estimators": 40, "max_leaf_nodes": 1500, "min_samples_leaf": 5, "max_samples": 0.20},
        {"id": "cfg_40t_2000leaf", "n_estimators": 40, "max_leaf_nodes": 2000, "min_samples_leaf": 4, "max_samples": 0.25},
        {"id": "cfg_50t_2000leaf", "n_estimators": 50, "max_leaf_nodes": 2000, "min_samples_leaf": 4, "max_samples": 0.25},
        {"id": "cfg_50t_2500leaf", "n_estimators": 50, "max_leaf_nodes": 2500, "min_samples_leaf": 4, "max_samples": 0.25},
    ]

    for cfg in configs:
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_leaf_nodes=cfg["max_leaf_nodes"],
            min_samples_leaf=cfg["min_samples_leaf"],
            max_samples=cfg["max_samples"],
            class_weight="balanced",
            random_state=42,
            n_jobs=4
        )
        rf.fit(X_train, y_train)
        p = tmp_dir / f"{cfg['id']}.joblib"
        joblib.dump(rf, p, compress=3)
        file_size = p.stat().st_size / (1024 * 1024)
        ram_delta = measure_load_ram_mb(p)
        preds = rf.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro")
        acc = accuracy_score(y_test, preds)
        kappa = cohen_kappa_score(y_test, preds)
        print(f"[{cfg['id']}] File: {file_size:.2f} MB | RAM: {ram_delta:.2f} MB | Acc: {acc*100:.2f}% | Kappa: {kappa:.4f} | Macro F1: {f1*100:.2f}% (Fit: {time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
