"""
Phase 5 - LULC Model Evaluation Module
Evaluates trained Random Forest model on the untouched test split (test.npz).
Computes overall accuracy, per-class Precision/Recall/F1, Cohen's Kappa,
generates raw and normalized confusion matrix PNGs, and saves test_metrics.json.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless PNG generation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support
)

CLASS_NAMES = {
    0: "Water",
    1: "Trees/Forest",
    2: "Crops",
    3: "Grassland",
    4: "Built-up",
    5: "Bare land"
}

CLASS_LIST = [CLASS_NAMES[i] for i in range(6)]


def plot_confusion_matrix(cm, classes, output_path, title="Confusion Matrix", normalize=False, cmap=plt.cm.Blues):
    """
    Renders and saves a beautifully styled, publication-ready Confusion Matrix PNG.
    """
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)

    if normalize:
        cm_display = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
        im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap, vmin=0, vmax=1.0)
    else:
        cm_display = cm
        im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    if normalize:
        cbar.set_label("Fraction of True Class", rotation=270, labelpad=18, fontsize=11, fontweight="bold")
    else:
        cbar.set_label("Pixel Count", rotation=270, labelpad=18, fontsize=11, fontweight="bold")

    # Tick marks & labels
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=11, fontweight="semibold")
    ax.set_yticklabels(classes, fontsize=11, fontweight="semibold")

    # Annotate cell numbers
    thresh = cm_display.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_display[i, j]
            if normalize:
                text_str = f"{val * 100:.1f}%\n({cm[i, j]:,d})"
            else:
                text_str = f"{cm[i, j]:,d}"

            ax.text(
                j, i, text_str,
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=9.5,
                fontweight="semibold"
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("True Ground-Truth Class", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold", labelpad=10)
    ax.grid(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix chart -> {output_path}")


def evaluate_lulc_test_set():
    start_time = time.time()
    script_dir = Path(__file__).resolve().parent
    base_ml_dir = script_dir.parents[1]
    
    data_dir = base_ml_dir / "data" / "processed"
    models_dir = base_ml_dir / "models"
    outputs_dir = base_ml_dir / "outputs"
    metrics_dir = outputs_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 5 - MODEL EVALUATION ON UNTOUCHED TEST SET")
    print("=" * 80)

    # 1. Load Artifacts & Test Data
    test_path = data_dir / "test.npz"
    model_path = models_dir / "lulc_rf_model.joblib"
    scaler_path = models_dir / "scaler.joblib"
    metadata_path = models_dir / "model_metadata.json"

    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    print(f"Loading test dataset    : {test_path}")
    test_data = np.load(test_path)
    X_test, y_test = test_data["X"], test_data["y"]
    print(f"  Test Sample Count     : {len(X_test):,} pixels, Features = {X_test.shape[1]}")

    print(f"Loading trained model   : {model_path}")
    rf_model = joblib.load(model_path)
    print(f"  Model Type            : {type(rf_model).__name__}")
    print(f"  Estimator Count       : {rf_model.n_estimators}")
    print(f"  Max Depth             : {rf_model.max_depth}")

    # 2. Run Predictions on Test Set
    print("\nRunning inference on test split...")
    t0 = time.time()
    batch_size = 250_000
    y_pred_list = []
    for i in range(0, len(X_test), batch_size):
        batch_X = np.asarray(X_test[i:i+batch_size], dtype=np.float32)
        batch_pred = rf_model.predict(batch_X)
        y_pred_list.append(batch_pred)
    
    y_pred = np.concatenate(y_pred_list).astype(np.uint8)
    y_test_u8 = np.asarray(y_test, dtype=np.uint8)
    infer_time = time.time() - t0
    print(f"Inference completed in {infer_time:.2f} seconds ({(len(X_test)/infer_time):,.0f} px/sec)")

    # 3. Compute Metrics
    print("\nCalculating formal classification evaluation metrics...")
    overall_acc = accuracy_score(y_test_u8, y_pred)
    kappa_score = cohen_kappa_score(y_test_u8, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test_u8, y_pred, labels=list(range(6)), zero_division=0
    )

    # 4. Generate Confusion Matrices
    cm_counts = confusion_matrix(y_test_u8, y_pred, labels=list(range(6)))
    
    # Save Confusion Matrix Charts
    cm_counts_png = metrics_dir / "confusion_matrix_counts.png"
    cm_norm_png = metrics_dir / "confusion_matrix_normalized.png"

    plot_confusion_matrix(
        cm_counts,
        classes=CLASS_LIST,
        output_path=cm_counts_png,
        title="LULC Random Forest - Test Confusion Matrix (Raw Counts)",
        normalize=False,
        cmap=plt.cm.Blues
    )

    plot_confusion_matrix(
        cm_counts,
        classes=CLASS_LIST,
        output_path=cm_norm_png,
        title="LULC Random Forest - Test Confusion Matrix (Normalized %)",
        normalize=True,
        cmap=plt.cm.Greens
    )

    # 5. Format & Save test_metrics.json
    per_class_metrics = {}
    print("\n" + "-" * 80)
    print(f"{'Class ID':<10} {'Class Name':<16} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support (px)':<14}")
    print("-" * 80)
    for i in range(6):
        c_name = CLASS_NAMES[i]
        p_val = float(precision[i])
        r_val = float(recall[i])
        f_val = float(f1[i])
        s_val = int(support[i])
        per_class_metrics[c_name] = {
            "class_id": i,
            "precision": round(p_val, 4),
            "recall": round(r_val, 4),
            "f1_score": round(f_val, 4),
            "support_pixels": s_val
        }
        print(f"{i:<10} {c_name:<16} {p_val*100:6.2f}%      {r_val*100:6.2f}%      {f_val*100:6.2f}%      {s_val:12,d}")
    print("-" * 80)

    test_metrics_data = {
        "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_dataset_path": str(test_path),
        "total_test_samples": int(len(y_test_u8)),
        "summary_metrics": {
            "overall_accuracy": round(float(overall_acc), 4),
            "cohen_kappa": round(float(kappa_score), 4),
            "macro_avg_f1": round(float(np.mean(f1)), 4),
            "weighted_avg_f1": round(float(np.sum(f1 * support) / np.sum(support)), 4)
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix_counts": cm_counts.tolist(),
        "class_label_mapping": CLASS_NAMES
    }

    test_metrics_json = metrics_dir / "test_metrics.json"
    with open(test_metrics_json, "w") as f:
        json.dump(test_metrics_data, f, indent=2)
    print(f"\nSaved test metrics JSON -> {test_metrics_json}")

    # 6. Print Final Summary Block
    total_duration = time.time() - start_time
    print("\n" + "=" * 80)
    print("PHASE 5 - TEST EVALUATION SUMMARY BLOCK")
    print("=" * 80)
    print(f"Total Test Set Pixels   : {len(y_test_u8):,} pixels")
    print(f"Overall Test Accuracy   : {overall_acc * 100:.2f}%")
    print(f"Cohen's Kappa Score     : {kappa_score:.4f}")
    print(f"Macro Average F1-Score  : {np.mean(f1) * 100:.2f}%")
    print(f"Weighted Average F1     : {(np.sum(f1 * support) / np.sum(support)) * 100:.2f}%")
    print("\nPer-Class F1 Scores:")
    for i in range(6):
        c_name = CLASS_NAMES[i]
        print(f"  • {c_name:<14s}: {f1[i] * 100:5.2f}% (Support: {support[i]:,d} px)")
    print(f"\nSaved Metrics Artifacts :")
    print(f"  1. {cm_counts_png}")
    print(f"  2. {cm_norm_png}")
    print(f"  3. {test_metrics_json}")
    print(f"Total Evaluation Time   : {total_duration:.2f}s")
    print("=" * 80)

    return test_metrics_data


if __name__ == "__main__":
    evaluate_lulc_test_set()
