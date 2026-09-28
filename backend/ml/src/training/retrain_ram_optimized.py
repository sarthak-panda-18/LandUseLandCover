"""
Script to train and benchmark 3 RAM-optimized RandomForest configurations
and evaluate the selected model.
"""

import os
import sys
import gc
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
    confusion_matrix
)

CLASS_NAMES = {
    0: "Water",
    1: "Trees/Forest",
    2: "Crops",
    3: "Grassland",
    4: "Built-up",
    5: "Bare land"
}

FEATURE_NAMES = ["B2 (Blue)", "B3 (Green)", "B4 (Red)", "B8 (NIR)", "NDVI"]


def measure_ram_delta_subprocess(model_path: str) -> float:
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
    script_dir = Path(__file__).resolve().parent
    base_ml_dir = script_dir.parents[1]
    data_dir = base_ml_dir / "data" / "processed"
    models_dir = base_ml_dir / "models"
    metrics_dir = base_ml_dir / "outputs" / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PART B: RETRAINING RAM-OPTIMIZED RANDOM FOREST CONFIGURATIONS")
    print("=" * 80)

    print("Loading train.npz and val.npz (leaving test.npz untouched until evaluation)...")
    train_data = np.load(data_dir / "train.npz")
    val_data = np.load(data_dir / "val.npz")

    X_train = np.asarray(train_data["X"], dtype=np.float32)
    y_train = np.asarray(train_data["y"], dtype=np.uint8)
    X_val = np.asarray(val_data["X"], dtype=np.float32)
    y_val = np.asarray(val_data["y"], dtype=np.uint8)

    print(f"  Training samples   : {len(X_train):,}")
    print(f"  Validation samples : {len(X_val):,}")

    configs = [
        {"id": "a", "name": "Config A (100 trees, 5000 max_leaf_nodes)", "n_estimators": 100, "max_leaf_nodes": 5000},
        {"id": "b", "name": "Config B (100 trees, 10000 max_leaf_nodes)", "n_estimators": 100, "max_leaf_nodes": 10000},
        {"id": "c", "name": "Config C (150 trees, 8000 max_leaf_nodes)", "n_estimators": 150, "max_leaf_nodes": 8000},
    ]

    results = []

    for cfg in configs:
        print("\n" + "-" * 70)
        print(f"Training {cfg['name']}...")
        params = {
            "n_estimators": cfg["n_estimators"],
            "max_leaf_nodes": cfg["max_leaf_nodes"],
            "min_samples_leaf": 3,
            "max_samples": 0.35,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        }
        
        t0 = time.time()
        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)
        fit_time = time.time() - t0
        print(f"  Fit completed in {fit_time:.2f}s")

        # Validation evaluation
        val_preds = rf.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        print(f"  Validation Accuracy : {val_acc * 100:.2f}% ({val_acc:.4f})")

        # Save temporarily with compress=3
        tmp_model_path = models_dir / f"temp_rf_model_{cfg['id']}.joblib"
        joblib.dump(rf, tmp_model_path, compress=3)
        file_size_mb = os.path.getsize(tmp_model_path) / (1024 * 1024)
        print(f"  Model File Size     : {file_size_mb:.2f} MB")

        # Measure RAM delta after joblib.load
        ram_delta_mb = measure_ram_delta_subprocess(str(tmp_model_path))
        print(f"  RAM Delta (RSS)     : {ram_delta_mb:.2f} MB")

        results.append({
            "cfg": cfg,
            "rf": rf,
            "val_acc": val_acc,
            "file_size_mb": file_size_mb,
            "ram_delta_mb": ram_delta_mb,
            "path": tmp_model_path
        })

    print("\n" + "=" * 80)
    print("SUMMARY OF CANDIDATE CONFIGURATIONS")
    print("=" * 80)
    print(f"{'Config':<10} {'n_est':<8} {'max_leaf':<10} {'Val Acc':<12} {'File Size':<14} {'RAM Delta':<14} {'<= 300MB?'}")
    print("-" * 80)
    for r in results:
        c = r["cfg"]
        qualifies = "YES" if r["ram_delta_mb"] <= 300 else "NO"
        print(f"{c['id']:<10} {c['n_estimators']:<8} {c['max_leaf_nodes']:<10} {r['val_acc']*100:6.2f}%      {r['file_size_mb']:6.2f} MB       {r['ram_delta_mb']:6.2f} MB       {qualifies}")
    print("-" * 80)

    # Filter <= 300 MB
    qualifying = [r for r in results if r["ram_delta_mb"] <= 300.0]
    if not qualifying:
        print("\n[ERROR] None of the configs have RAM delta <= 300 MB! Stopping.")
        sys.exit(1)

    # Pick best validation accuracy
    best = max(qualifying, key=lambda x: x["val_acc"])
    print(f"\nSelected Winner: Config {best['cfg']['id']} ({best['cfg']['name']})")
    print(f"  Val Acc: {best['val_acc']*100:.2f}%, File Size: {best['file_size_mb']:.2f} MB, RAM Delta: {best['ram_delta_mb']:.2f} MB")

    # Overwrite backend/ml/models/lulc_rf_model.joblib
    final_model_path = models_dir / "lulc_rf_model.joblib"
    print(f"\nSaving chosen model to {final_model_path} (compress=3)...")
    joblib.dump(best["rf"], final_model_path, compress=3)
    final_file_size_mb = os.path.getsize(final_model_path) / (1024 * 1024)
    print(f"Final Model Saved. Size: {final_file_size_mb:.2f} MB")

    # Clean up temp files
    for r in results:
        if r["path"].exists():
            try:
                r["path"].unlink()
            except Exception:
                pass

    # Evaluate on test.npz
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST.NPZ")
    print("=" * 80)
    test_data = np.load(data_dir / "test.npz")
    X_test = np.asarray(test_data["X"], dtype=np.float32)
    y_test = np.asarray(test_data["y"], dtype=np.uint8)
    print(f"Test samples: {len(X_test):,}")

    batch_size = 250_000
    test_preds_list = []
    for i in range(0, len(X_test), batch_size):
        test_preds_list.append(best["rf"].predict(X_test[i:i+batch_size]))
    test_preds = np.concatenate(test_preds_list).astype(np.uint8)

    test_acc = accuracy_score(y_test, test_preds)
    test_kappa = cohen_kappa_score(y_test, test_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, test_preds, labels=list(range(6)), zero_division=0
    )
    macro_f1 = np.mean(f1)
    weighted_f1 = np.sum(f1 * support) / np.sum(support)
    cm = confusion_matrix(y_test, test_preds, labels=list(range(6)))

    print(f"Test Accuracy    : {test_acc * 100:.2f}% (Previous: 66.26%)")
    print(f"Cohen's Kappa (κ): {test_kappa:.4f} (Previous: 0.5323)")
    print(f"Macro Average F1 : {macro_f1 * 100:.2f}% (Previous: 62.34%)")
    print(f"Weighted Avg F1  : {weighted_f1 * 100:.2f}% (Previous: 69.81%)")

    print("\nPer-Class Breakdown:")
    per_class_test_dict = {}
    for i in range(6):
        c_name = CLASS_NAMES[i]
        p_val, r_val, f_val, s_val = float(precision[i]), float(recall[i]), float(f1[i]), int(support[i])
        per_class_test_dict[c_name] = {
            "class_id": i,
            "precision": round(p_val, 4),
            "recall": round(r_val, 4),
            "f1_score": round(f_val, 4),
            "support_pixels": s_val
        }
        print(f"  {c_name:<16}: Precision={p_val*100:5.2f}%, Recall={r_val*100:5.2f}%, F1={f_val*100:5.2f}% (Support: {s_val:,})")

    # Feature Importances
    importances = best["rf"].feature_importances_
    feat_imp_dict = {f: round(float(imp), 4) for f, imp in zip(FEATURE_NAMES, importances)}

    # Update model_metadata.json
    cfg_chosen = best["cfg"]
    metadata = {
        "model_type": "RandomForestClassifier",
        "version_note": f"RAM-Optimized Production Model: n_estimators={cfg_chosen['n_estimators']}, max_leaf_nodes={cfg_chosen['max_leaf_nodes']}, min_samples_leaf=3, max_samples=0.35, compress=3 (File Size: {final_file_size_mb:.2f} MB, RAM Delta: {best['ram_delta_mb']:.2f} MB).",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples_count": int(len(X_train)),
        "validation_samples_count": int(len(X_val)),
        "hyperparameters": {
            "n_estimators": cfg_chosen["n_estimators"],
            "max_leaf_nodes": cfg_chosen["max_leaf_nodes"],
            "min_samples_leaf": 3,
            "max_samples": 0.35,
            "class_weight": "balanced",
            "random_state": 42
        },
        "metrics": {
            "validation_accuracy": round(float(best["val_acc"]), 4),
            "test_accuracy": round(float(test_acc), 4),
            "cohen_kappa": round(float(test_kappa), 4),
            "macro_avg_f1": round(float(macro_f1), 4),
            "weighted_avg_f1": round(float(weighted_f1), 4),
            "per_class_test_f1": {k: v["f1_score"] for k, v in per_class_test_dict.items()}
        },
        "model_file_size_mb": round(final_file_size_mb, 2),
        "model_ram_delta_mb": round(best["ram_delta_mb"], 2),
        "feature_importances": feat_imp_dict,
        "class_mapping": CLASS_NAMES
    }

    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved updated model metadata -> {metadata_path}")

    # Update test_metrics.json
    test_metrics_data = {
        "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_dataset_path": str(data_dir / "test.npz"),
        "total_test_samples": int(len(y_test)),
        "summary_metrics": {
            "overall_accuracy": round(float(test_acc), 4),
            "cohen_kappa": round(float(test_kappa), 4),
            "macro_avg_f1": round(float(macro_f1), 4),
            "weighted_avg_f1": round(float(weighted_f1), 4)
        },
        "per_class_metrics": per_class_test_dict,
        "confusion_matrix_counts": cm.tolist(),
        "class_label_mapping": CLASS_NAMES,
        "model_file_size_mb": round(final_file_size_mb, 2),
        "model_ram_delta_mb": round(best["ram_delta_mb"], 2)
    }

    test_metrics_path = metrics_dir / "test_metrics.json"
    with open(test_metrics_path, "w") as f:
        json.dump(test_metrics_data, f, indent=2)
    print(f"Saved updated test metrics -> {test_metrics_path}")


if __name__ == "__main__":
    main()
