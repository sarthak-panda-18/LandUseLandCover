"""
Direct Retraining and Evaluation Script for Compact LULC Model
Trains RandomForest(n_estimators=300, max_depth=30, min_samples_leaf=3, max_samples=0.35, class_weight='balanced')
Saves with joblib.dump(compress=3) and evaluates immediately on test.npz.
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
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


def retrain_and_evaluate():
    script_dir = Path(__file__).resolve().parent
    base_ml_dir = script_dir.parents[1]
    data_dir = base_ml_dir / "data" / "processed"
    models_dir = base_ml_dir / "models"
    metrics_dir = base_ml_dir / "outputs" / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RETRAINING COMPACT RANDOM FOREST MODEL (max_depth=30, compress=3)")
    print("=" * 80)

    # 1. Load Data
    print("Loading datasets...")
    train_data = np.load(data_dir / "train.npz")
    val_data = np.load(data_dir / "val.npz")
    test_data = np.load(data_dir / "test.npz")

    X_train_f32 = np.asarray(train_data["X"], dtype=np.float32)
    y_train_u8 = np.asarray(train_data["y"], dtype=np.uint8)
    X_val_f32 = np.asarray(val_data["X"], dtype=np.float32)
    y_val_u8 = np.asarray(val_data["y"], dtype=np.uint8)
    X_test_f32 = np.asarray(test_data["X"], dtype=np.float32)
    y_test_u8 = np.asarray(test_data["y"], dtype=np.uint8)

    print(f"  Training samples   : {len(X_train_f32):,}")
    print(f"  Validation samples : {len(X_val_f32):,}")
    print(f"  Test samples       : {len(X_test_f32):,}")

    # 2. Train Model
    params = {
        "n_estimators": 300,
        "max_depth": 30,
        "min_samples_leaf": 3,
        "max_samples": 0.35,
        "class_weight": "balanced",
        "n_jobs": 4,
        "random_state": 42
    }

    print("\nTraining configuration:")
    for k, v in params.items():
        print(f"  - {k:<18s}: {v}")

    t0 = time.time()
    rf = RandomForestClassifier(**params)
    rf.fit(X_train_f32, y_train_u8)
    fit_duration = time.time() - t0
    print(f"\nModel fit completed in {fit_duration:.2f}s ({fit_duration/60:.2f} min)")

    # 3. Save Model with Compression
    model_path = models_dir / "lulc_rf_model.joblib"
    print(f"\nSaving model with joblib.dump(compress=3) -> {model_path}...")
    t_save = time.time()
    joblib.dump(rf, model_path, compress=3)
    save_duration = time.time() - t_save

    file_size_bytes = os.path.getsize(model_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"SAVED MODEL FILE SIZE: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
    print(f"Save compression time : {save_duration:.2f}s")

    # 4. Check if size is under 300MB, else test max_leaf_nodes=20000
    if file_size_mb > 300:
        print(f"\n[Notice] Model size ({file_size_mb:.2f} MB) is still > 300MB.")
        print("Testing max_leaf_nodes=20000 constraint as requested...")
        params["max_leaf_nodes"] = 20000
        t0_leaf = time.time()
        rf = RandomForestClassifier(**params)
        rf.fit(X_train_f32, y_train_u8)
        print(f"Fit with max_leaf_nodes=20000 completed in {time.time()-t0_leaf:.2f}s")

        joblib.dump(rf, model_path, compress=3)
        file_size_bytes = os.path.getsize(model_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"UPDATED MODEL FILE SIZE (max_leaf_nodes=20000): {file_size_mb:.2f} MB")

    # 5. Evaluate on Validation Set
    val_preds = rf.predict(X_val_f32)
    val_acc = accuracy_score(y_val_u8, val_preds)
    print(f"\nValidation Accuracy : {val_acc * 100:.2f}%")

    # Per-Class Validation Accuracies
    per_class_val = {}
    for c in range(6):
        mask = (y_val_u8 == c)
        acc = float((val_preds[mask] == c).sum() / mask.sum()) if mask.sum() > 0 else 0.0
        per_class_val[CLASS_NAMES[c]] = round(acc, 4)

    # 6. Evaluate on Test Set (Phase 5)
    print("\nRunning inference on Test Set (1.92M pixels)...")
    t_test = time.time()
    batch_size = 250_000
    test_preds_list = []
    for i in range(0, len(X_test_f32), batch_size):
        test_preds_list.append(rf.predict(X_test_f32[i:i+batch_size]))
    test_preds = np.concatenate(test_preds_list).astype(np.uint8)
    test_infer_time = time.time() - t_test

    test_acc = accuracy_score(y_test_u8, test_preds)
    test_kappa = cohen_kappa_score(y_test_u8, test_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test_u8, test_preds, labels=list(range(6)), zero_division=0
    )
    macro_f1 = np.mean(f1)
    weighted_f1 = np.sum(f1 * support) / np.sum(support)

    cm = confusion_matrix(y_test_u8, test_preds, labels=list(range(6)))

    print("\n" + "=" * 80)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 80)
    print(f"Overall Test Accuracy : {test_acc * 100:.2f}%")
    print(f"Cohen's Kappa (κ)     : {test_kappa:.4f}")
    print(f"Macro Average F1      : {macro_f1 * 100:.2f}%")
    print(f"Weighted Average F1   : {weighted_f1 * 100:.2f}%")
    print("\nPer-Class Breakdown on Test Split:")
    print(f"{'Class ID':<10} {'Class Name':<16} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<14}")
    print("-" * 76)
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
        print(f"{i:<10} {c_name:<16} {p_val*100:6.2f}%      {r_val*100:6.2f}%      {f_val*100:6.2f}%      {s_val:12,d}")
    print("-" * 76)

    # Feature Importances
    importances = rf.feature_importances_
    feat_imp_dict = {f: round(float(imp), 4) for f, imp in zip(FEATURE_NAMES, importances)}

    # 7. Update model_metadata.json
    metadata = {
        "model_type": "RandomForestClassifier",
        "version_note": f"Compact Production Model: max_depth={params.get('max_depth')}, max_leaf_nodes={params.get('max_leaf_nodes')}, class_weight='balanced', min_samples_leaf=3, max_samples=0.35, compress=3 (File Size: {file_size_mb:.2f} MB).",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples_count": int(len(X_train_f32)),
        "validation_samples_count": int(len(X_val_f32)),
        "hyperparameters": {
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "max_leaf_nodes": params.get("max_leaf_nodes"),
            "min_samples_leaf": params["min_samples_leaf"],
            "max_samples": params["max_samples"],
            "class_weight": params["class_weight"],
            "random_state": 42
        },
        "metrics": {
            "training_accuracy_subset": round(float(rf.score(X_train_f32[:200_000], y_train_u8[:200_000])), 4),
            "validation_accuracy": round(float(val_acc), 4),
            "per_class_validation_accuracy": per_class_val
        },
        "feature_importances": feat_imp_dict,
        "class_mapping": CLASS_NAMES,
        "model_file_size_mb": round(file_size_mb, 2)
    }

    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved updated metadata -> {metadata_path}")

    # 8. Update test_metrics.json
    test_metrics_data = {
        "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_dataset_path": str(data_dir / "test.npz"),
        "total_test_samples": int(len(y_test_u8)),
        "summary_metrics": {
            "overall_accuracy": round(float(test_acc), 4),
            "cohen_kappa": round(float(test_kappa), 4),
            "macro_avg_f1": round(float(macro_f1), 4),
            "weighted_avg_f1": round(float(weighted_f1), 4)
        },
        "per_class_metrics": per_class_test_dict,
        "confusion_matrix_counts": cm.tolist(),
        "class_label_mapping": CLASS_NAMES,
        "model_file_size_mb": round(file_size_mb, 2)
    }

    test_metrics_path = metrics_dir / "test_metrics.json"
    with open(test_metrics_path, "w") as f:
        json.dump(test_metrics_data, f, indent=2)
    print(f"Saved updated test metrics -> {test_metrics_path}")

    return file_size_mb, test_acc, test_kappa, macro_f1, weighted_f1, per_class_test_dict["Grassland"]


if __name__ == "__main__":
    retrain_and_evaluate()
