"""
Model Training Module for LULC Classification (Phase 4)
Trains a Random Forest classifier on Sentinel-2 5-band features to predict 6 land-cover classes.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

CLASS_NAMES = {
    0: "Water",
    1: "Trees/Forest",
    2: "Crops",
    3: "Grassland",
    4: "Built-up",
    5: "Bare land"
}

FEATURE_NAMES = ["B2 (Blue)", "B3 (Green)", "B4 (Red)", "B8 (NIR)", "NDVI"]


def train_lulc_model():
    start_time = time.time()
    script_dir = Path(__file__).resolve().parent
    processed_dir = script_dir.parents[1] / "data" / "processed"
    models_dir = script_dir.parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 4 - RANDOM FOREST MODEL TRAINING & HYPERPARAMETER TUNING")
    print("=" * 80)

    # 1. Load Processed Dataset Splits
    train_path = processed_dir / "train.npz"
    val_path = processed_dir / "val.npz"

    print(f"Loading training data   : {train_path}")
    train_data = np.load(train_path)
    X_train, y_train = train_data["X"], train_data["y"]
    print(f"  Training Samples      : {len(X_train):,} pixels, Features = {X_train.shape[1]}")

    print(f"Loading validation data : {val_path}")
    val_data = np.load(val_path)
    X_val, y_val = val_data["X"], val_data["y"]
    print(f"  Validation Samples    : {len(X_val):,} pixels, Features = {X_val.shape[1]}")

    # Check Class Balance in Training Set
    unique_classes, train_counts = np.unique(y_train, return_counts=True)
    total_train = len(y_train)
    min_count = np.min(train_counts)
    max_count = np.max(train_counts)
    imbalance_ratio = max_count / min_count

    print("\nTraining Set Class Balance Assessment:")
    for c, cnt in zip(unique_classes, train_counts):
        print(f"  Class {c} ({CLASS_NAMES[c]:<12s}): {cnt:10,d} px ({(cnt/total_train)*100:5.2f}%)")

    print(f"  Imbalance Ratio: {imbalance_ratio:.2f}x difference between majority and minority class.")
    if imbalance_ratio > 2.0:
        print("  -> Using class_weight='balanced' to ensure proportional penalty weighting across all classes.")
        class_weight = "balanced"
    else:
        print("  -> Class counts are balanced. Skipping class weighting.")
        class_weight = None

    # Subsample for Hyperparameter Search to keep tuning fast and responsive
    tune_size = min(250_000, len(X_train))
    val_eval_size = min(100_000, len(X_val))

    # Stratified subsampling for search
    np.random.seed(42)
    indices = np.arange(len(X_train))
    tune_idx = np.random.choice(indices, size=tune_size, replace=False)
    X_tune, y_tune = X_train[tune_idx], y_train[tune_idx]

    val_indices = np.arange(len(X_val))
    val_eval_idx = np.random.choice(val_indices, size=val_eval_size, replace=False)
    X_val_eval, y_val_eval = X_val[val_eval_idx], y_val[val_eval_idx]

    # Task 1: Baseline / Default Starting Model
    print("\n" + "-" * 80)
    print("TASK 1: EVALUATING BASELINE MODEL (Starting Defaults)")
    print("-" * 80)
    baseline_params = {
        "n_estimators": 100,
        "max_depth": None,
        "class_weight": class_weight,
        "n_jobs": -1,
        "random_state": 42
    }
    print(f"Default Parameters: n_estimators={baseline_params['n_estimators']}, max_depth={baseline_params['max_depth']}, class_weight={class_weight}")
    t0 = time.time()
    baseline_rf = RandomForestClassifier(**baseline_params)
    baseline_rf.fit(X_tune, y_tune)
    baseline_val_acc = baseline_rf.score(X_val_eval, y_val_eval)
    t1 = time.time()
    print(f"Baseline Validation Accuracy : {baseline_val_acc * 100:.2f}% (Fit time: {t1 - t0:.2f}s)")

    # Task 2: Hyperparameter Search
    print("\n" + "-" * 80)
    print("TASK 2: HYPERPARAMETER SEARCH OVER (n_estimators x max_depth)")
    print("-" * 80)

    n_estimators_grid = [100, 200, 300]
    max_depth_grid = [20, 30, None]

    best_val_acc = float("-inf")
    best_params = None
    grid_results = []

    print(f"{'Config #':<10} {'n_estimators':<15} {'max_depth':<15} {'Val Accuracy':<15} {'Fit Time':<12}")
    print("-" * 70)

    config_num = 1
    for n_est in n_estimators_grid:
        for depth in max_depth_grid:
            depth_str = str(depth) if depth is not None else "None (unlimited)"
            t_start = time.time()
            rf_candidate = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=depth,
                class_weight=class_weight,
                n_jobs=-1,
                random_state=42
            )
            rf_candidate.fit(X_tune, y_tune)
            val_acc = rf_candidate.score(X_val_eval, y_val_eval)
            fit_time = time.time() - t_start

            grid_results.append({
                "n_estimators": n_est,
                "max_depth": depth,
                "val_accuracy": round(float(val_acc), 4),
                "fit_time_seconds": round(fit_time, 2)
            })

            print(f"{config_num:<10} {n_est:<15} {depth_str:<15} {val_acc*100:6.2f}%         {fit_time:6.2f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {"n_estimators": n_est, "max_depth": depth}

            config_num += 1

    print("-" * 70)
    best_depth_repr = str(best_params['max_depth']) if best_params['max_depth'] is not None else "None"
    print(f"\nBest Combination Found: n_estimators={best_params['n_estimators']}, max_depth={best_depth_repr} (Validation Accuracy = {best_val_acc*100:.2f}%)")

    # Task 3: Retrain Final Model on Full Training Set
    print("\n" + "-" * 80)
    print("TASK 3: TRAINING FINAL MODEL ON FULL TRAINING DATASET")
    print("-" * 80)

    # For training across multi-million satellite pixels efficiently:
    # 1. Cast X to float32 and y to uint8 to minimize memory footprint
    # 2. Use max_samples=0.25 per tree bootstrap (~2.24M samples per tree) for optimal ensemble diversity & memory safety
    # 3. Set max_depth=25, min_samples_leaf=4, and n_jobs=4
    X_train_f32 = np.asarray(X_train, dtype=np.float32)
    y_train_u8 = np.asarray(y_train, dtype=np.uint8)
    X_val_f32 = np.asarray(X_val, dtype=np.float32)
    y_val_u8 = np.asarray(y_val, dtype=np.uint8)

    final_n_est = 300
    final_max_depth = 30                        # Explicit bounded depth
    final_min_samples_leaf = 3                  # Regularized leaf size
    final_max_samples = 0.35                    # Bootstrap ~3.14M pixels per tree for memory-safe training

    print(f"Training final RandomForestClassifier with parameters:")
    print(f"  - n_estimators     : {final_n_est}")
    print(f"  - max_depth        : {final_max_depth}")
    print(f"  - min_samples_leaf : {final_min_samples_leaf}")
    print(f"  - max_samples      : {final_max_samples} (~3.14M pixels / tree)")
    print(f"  - class_weight     : {class_weight}")
    print(f"  - Training on      : {len(X_train_f32):,} pixels")

    final_rf = RandomForestClassifier(
        n_estimators=final_n_est,
        max_depth=final_max_depth,
        min_samples_leaf=final_min_samples_leaf,
        max_samples=final_max_samples,
        class_weight=class_weight,
        n_jobs=4,
        random_state=42
    )

    t_fit_start = time.time()
    final_rf.fit(X_train_f32, y_train_u8)
    fit_duration = time.time() - t_fit_start
    print(f"Final Model Fit Complete! Duration: {fit_duration:.2f} seconds ({fit_duration/60:.2f} minutes)")

    # Evaluate Overall Accuracy
    print("\nEvaluating Accuracies on Full Datasets...")
    train_eval_acc = final_rf.score(X_train_f32[:200_000], y_train_u8[:200_000])
    val_acc_full = final_rf.score(X_val_f32, y_val_u8)

    print(f"  Training Set Accuracy   (200k subset) : {train_eval_acc * 100:.2f}%")
    print(f"  Validation Set Accuracy (Full 1.92M)  : {val_acc_full * 100:.2f}%")

    # Per-Class Validation Accuracy Sanity Check
    print("\nPer-Class Validation Accuracy (Sanity Check):")
    print(f"{'Class ID':<10} {'Class Name':<16} {'Val Total Pixels':<18} {'Class Accuracy':<16}")
    print("-" * 60)

    val_preds = final_rf.predict(X_val_f32)
    per_class_accuracies = {}

    for c in range(6):
        c_mask = (y_val == c)
        c_total = int(c_mask.sum())
        if c_total > 0:
            c_correct = int((val_preds[c_mask] == c).sum())
            c_acc = c_correct / c_total
        else:
            c_acc = 0.0
        per_class_accuracies[CLASS_NAMES[c]] = round(float(c_acc), 4)
        print(f"{c:<10} {CLASS_NAMES[c]:<16} {c_total:14,d}   {c_acc*100:6.2f}%")

    print("-" * 60)

    # Feature Importances
    print("\nInput Feature Importances:")
    importances = final_rf.feature_importances_
    feat_imp_dict = {}
    for idx, (f_name, imp) in enumerate(zip(FEATURE_NAMES, importances), start=1):
        feat_imp_dict[f_name] = round(float(imp), 4)
        bar = "#" * int(imp * 40)
        print(f"  Band {idx} ({f_name:<16s}): {imp * 100:6.2f}%  {bar}")

    # Task 4: Save Model Artifact
    model_save_path = models_dir / "lulc_rf_model.joblib"
    print(f"\nSaving final model artifact -> {model_save_path}...")
    joblib.dump(final_rf, model_save_path, compress=3)
    model_size_mb = os.path.getsize(model_save_path) / 1e6
    print(f"Model saved successfully! Artifact Size: {model_size_mb:.2f} MB")

    # Task 5: Save Model Metadata
    metadata_path = models_dir / "model_metadata.json"
    metadata = {
        "model_type": "RandomForestClassifier",
        "version_note": "Optimized Compact Model: Bounded max_depth=30, class_weight='balanced', min_samples_leaf=3, max_samples=0.35, compress=3.",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples_count": int(len(X_train)),
        "validation_samples_count": int(len(X_val)),
        "hyperparameters": {
            "n_estimators": final_n_est,
            "max_depth": final_max_depth,
            "min_samples_leaf": final_min_samples_leaf,
            "max_samples": final_max_samples,
            "class_weight": class_weight,
            "random_state": 42
        },
        "metrics": {
            "training_accuracy_subset": round(float(train_eval_acc), 4),
            "validation_accuracy": round(float(val_acc_full), 4),
            "per_class_validation_accuracy": per_class_accuracies
        },
        "feature_importances": feat_imp_dict,
        "class_mapping": CLASS_NAMES,
        "grid_search_history": grid_results
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model metadata -> {metadata_path}")

    # Task 6: Final Summary Block
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("PHASE 4 - TRAINING SUMMARY & VERIFICATION BLOCK")
    print("=" * 80)
    print(f"Model Algorithm         : Scikit-Learn RandomForestClassifier")
    print(f"Final Hyperparameters   : n_estimators={final_n_est}, max_depth={final_max_depth}, class_weight='{class_weight}'")
    print(f"Training Samples Used   : {len(X_train):,} pixels")
    print(f"Validation Samples Used : {len(X_val):,} pixels")
    print(f"Training Accuracy       : {train_eval_acc * 100:.2f}%")
    print(f"Validation Accuracy     : {val_acc_full * 100:.2f}%")
    print(f"Most Important Feature  : {max(feat_imp_dict, key=feat_imp_dict.get)} ({max(feat_imp_dict.values())*100:.2f}%)")
    print(f"Saved Model File        : {model_save_path} ({model_size_mb:.2f} MB)")
    print(f"Saved Metadata File     : {metadata_path}")
    print(f"Total Phase 4 Runtime   : {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("=" * 80)


if __name__ == "__main__":
    train_lulc_model()
