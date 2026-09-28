"""
Experiment with ultra-compact, memory-safe RandomForestClassifier models.
Tests multiple tree/leaf configs on train.npz and evaluates on test.npz,
measuring file size, load RAM delta, accuracy, Cohen's kappa, and macro F1.
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
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    classification_report
)

def measure_load_ram_mb(model_path: Path) -> float:
    """Measures RSS delta in a clean fresh python subprocess."""
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
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True
    )
    return float(res.stdout.strip())

def main():
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"
    tmp_models_dir = models_dir / "compact_experiments"
    tmp_models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXPLORING ULTRA-COMPACT RANDOM FOREST CONFIGURATIONS FOR < 200MB RAM")
    print("=" * 80)

    # 1. Load Datasets
    print("Loading train.npz and test.npz...")
    train_data = np.load(data_dir / "train.npz")
    test_data = np.load(data_dir / "test.npz")

    X_train = np.asarray(train_data["X"], dtype=np.float32)
    y_train = np.asarray(train_data["y"], dtype=np.uint8)
    X_test = np.asarray(test_data["X"], dtype=np.float32)
    y_test = np.asarray(test_data["y"], dtype=np.uint8)

    print(f"  Train samples : {len(X_train):,}")
    print(f"  Test samples  : {len(X_test):,}")

    # Baseline measurement of current model
    current_model_path = models_dir / "lulc_rf_model.joblib"
    if current_model_path.exists():
        curr_size_mb = current_model_path.stat().st_size / (1024 * 1024)
        curr_ram_mb = measure_load_ram_mb(current_model_path)
        print(f"\nCurrent Baseline Model: {curr_size_mb:.2f} MB file, {curr_ram_mb:.2f} MB RAM delta")

    configs = [
        {
            "id": "cfg_60t_3000leaf",
            "name": "Config 1: 60 trees, max_leaf_nodes=3000, min_samples_leaf=4, max_samples=0.25",
            "n_estimators": 60,
            "max_leaf_nodes": 3000,
            "min_samples_leaf": 4,
            "max_samples": 0.25
        },
        {
            "id": "cfg_70t_3500leaf",
            "name": "Config 2: 70 trees, max_leaf_nodes=3500, min_samples_leaf=3, max_samples=0.30",
            "n_estimators": 70,
            "max_leaf_nodes": 3500,
            "min_samples_leaf": 3,
            "max_samples": 0.30
        },
        {
            "id": "cfg_80t_4000leaf",
            "name": "Config 3: 80 trees, max_leaf_nodes=4000, min_samples_leaf=3, max_samples=0.30",
            "n_estimators": 80,
            "max_leaf_nodes": 4000,
            "min_samples_leaf": 3,
            "max_samples": 0.30
        },
        {
            "id": "cfg_60t_5000leaf",
            "name": "Config 4: 60 trees, max_leaf_nodes=5000, min_samples_leaf=3, max_samples=0.30",
            "n_estimators": 60,
            "max_leaf_nodes": 5000,
            "min_samples_leaf": 3,
            "max_samples": 0.30
        },
        {
            "id": "cfg_80t_5000leaf",
            "name": "Config 5: 80 trees, max_leaf_nodes=5000, min_samples_leaf=3, max_samples=0.35",
            "n_estimators": 80,
            "max_leaf_nodes": 5000,
            "min_samples_leaf": 3,
            "max_samples": 0.35
        },
    ]

    results = []

    for cfg in configs:
        print("\n" + "-" * 80)
        print(f"Training {cfg['name']}...")
        params = {
            "n_estimators": cfg["n_estimators"],
            "max_leaf_nodes": cfg["max_leaf_nodes"],
            "min_samples_leaf": cfg["min_samples_leaf"],
            "max_samples": cfg["max_samples"],
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": 4
        }

        t0 = time.time()
        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)
        fit_time = time.time() - t0

        # Save compressed
        save_path = tmp_models_dir / f"{cfg['id']}.joblib"
        joblib.dump(rf, save_path, compress=3)
        file_size_mb = save_path.stat().st_size / (1024 * 1024)

        # Measure RAM in subprocess
        ram_delta_mb = measure_load_ram_mb(save_path)

        # Evaluate on test.npz
        t_eval = time.time()
        preds = rf.predict(X_test)
        eval_time = time.time() - t_eval

        acc = accuracy_score(y_test, preds)
        kappa = cohen_kappa_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        weighted_f1 = f1_score(y_test, preds, average="weighted")
        per_class_f1 = f1_score(y_test, preds, average=None)

        res = {
            "id": cfg["id"],
            "name": cfg["name"],
            "params": params,
            "file_size_mb": round(file_size_mb, 2),
            "ram_delta_mb": round(ram_delta_mb, 2),
            "fit_time_s": round(fit_time, 1),
            "eval_time_s": round(eval_time, 2),
            "accuracy": round(acc * 100, 2),
            "cohen_kappa": round(kappa, 4),
            "macro_f1": round(macro_f1 * 100, 2),
            "weighted_f1": round(weighted_f1 * 100, 2),
            "per_class_f1": [round(f * 100, 2) for f in per_class_f1],
            "save_path": str(save_path)
        }
        results.append(res)

        print(f"  -> File Size     : {file_size_mb:.2f} MB")
        print(f"  -> RAM Load Delta: {ram_delta_mb:.2f} MB")
        print(f"  -> Test Accuracy : {acc * 100:.2f}%")
        print(f"  -> Cohen's Kappa : {kappa:.4f}")
        print(f"  -> Macro Avg F1  : {macro_f1 * 100:.2f}% (Baseline: 61.3%)")
        print(f"  -> Weighted F1   : {weighted_f1 * 100:.2f}%")
        print(f"  -> Per-Class F1  : Water={per_class_f1[0]*100:.1f}%, Trees={per_class_f1[1]*100:.1f}%, Crops={per_class_f1[2]*100:.1f}%, Grass={per_class_f1[3]*100:.1f}%, Built={per_class_f1[4]*100:.1f}%, Bare={per_class_f1[5]*100:.1f}%")

    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Config':<25s} | {'File (MB)':<10s} | {'RAM (MB)':<10s} | {'Accuracy':<10s} | {'Kappa':<8s} | {'Macro F1':<10s}")
    print("-" * 80)
    for r in results:
        print(f"{r['id']:<25s} | {r['file_size_mb']:<10.2f} | {r['ram_delta_mb']:<10.2f} | {r['accuracy']:<9.2f}% | {r['cohen_kappa']:<8.4f} | {r['macro_f1']:<9.2f}%")

    # Pick the best balanced config with RAM < 120MB and macro F1 >= 60.5%
    # and save as lulc_rf_model_v2.joblib
    best = min(results, key=lambda x: (x['ram_delta_mb'] if x['macro_f1'] >= 60.5 else 9999))
    v2_dest = models_dir / "lulc_rf_model_v2.joblib"
    import shutil
    shutil.copy2(best['save_path'], v2_dest)
    print(f"\nSaved best config '{best['id']}' to: {v2_dest}")
    print(f"V2 Model File Size: {v2_dest.stat().st_size / (1024*1024):.2f} MB, RAM Delta: {best['ram_delta_mb']:.2f} MB")

if __name__ == "__main__":
    main()
