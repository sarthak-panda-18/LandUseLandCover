"""
Evaluate the newly trained lulc_rf_model.joblib on test.npz and update metadata JSONs.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
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


def main():
    script_dir = Path(__file__).resolve().parent
    base_ml_dir = script_dir.parents[1]
    data_dir = base_ml_dir / "data" / "processed"
    models_dir = base_ml_dir / "models"
    metrics_dir = base_ml_dir / "outputs" / "metrics"

    model_path = models_dir / "lulc_rf_model.joblib"
    print(f"Loading model: {model_path}")
    rf = joblib.load(model_path)
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    test_data = np.load(data_dir / "test.npz")
    X_test = np.asarray(test_data["X"], dtype=np.float32)
    y_test = np.asarray(test_data["y"], dtype=np.uint8)
    print(f"Test samples: {len(X_test):,}")

    batch_size = 250_000
    test_preds_list = []
    for i in range(0, len(X_test), batch_size):
        test_preds_list.append(rf.predict(X_test[i:i+batch_size]))
    test_preds = np.concatenate(test_preds_list).astype(np.uint8)

    test_acc = accuracy_score(y_test, test_preds)
    test_kappa = cohen_kappa_score(y_test, test_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, test_preds, labels=list(range(6)), zero_division=0
    )
    macro_f1 = np.mean(f1)
    weighted_f1 = np.sum(f1 * support) / np.sum(support)
    cm = confusion_matrix(y_test, test_preds, labels=list(range(6)))

    print("=" * 80)
    print("TEST METRICS EVALUATION")
    print("=" * 80)
    print(f"Overall Test Accuracy : {test_acc * 100:.2f}% (0.6250)")
    print(f"Cohen Kappa           : {test_kappa:.4f}")
    print(f"Macro Average F1      : {macro_f1 * 100:.2f}%")
    print(f"Weighted Average F1   : {weighted_f1 * 100:.2f}%")

    per_class_test_dict = {}
    print("\nPer-Class Breakdown:")
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

    # Feature importances
    importances = rf.feature_importances_
    feat_imp_dict = {f: round(float(imp), 4) for f, imp in zip(FEATURE_NAMES, importances)}

    # Update model_metadata.json
    metadata = {
        "model_type": "RandomForestClassifier",
        "version_note": f"RAM-Optimized Production Model: n_estimators=100, max_leaf_nodes=5000, min_samples_leaf=3, max_samples=0.35, compress=3 (File Size: {file_size_mb:.2f} MB, RAM Delta: 260.22 MB).",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples_count": 8976744,
        "validation_samples_count": 1923588,
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": None,
            "max_leaf_nodes": 5000,
            "min_samples_leaf": 3,
            "max_samples": 0.35,
            "class_weight": "balanced",
            "random_state": 42
        },
        "metrics": {
            "validation_accuracy": 0.6249,
            "test_accuracy": round(float(test_acc), 4),
            "cohen_kappa": round(float(test_kappa), 4),
            "macro_avg_f1": round(float(macro_f1), 4),
            "weighted_avg_f1": round(float(weighted_f1), 4),
            "per_class_test_f1": {k: v["f1_score"] for k, v in per_class_test_dict.items()}
        },
        "model_file_size_mb": round(file_size_mb, 2),
        "model_ram_delta_mb": 260.22,
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
        "model_file_size_mb": round(file_size_mb, 2),
        "model_ram_delta_mb": 260.22
    }

    test_metrics_path = metrics_dir / "test_metrics.json"
    with open(test_metrics_path, "w") as f:
        json.dump(test_metrics_data, f, indent=2)
    print(f"Saved updated test metrics -> {test_metrics_path}")


if __name__ == "__main__":
    main()
