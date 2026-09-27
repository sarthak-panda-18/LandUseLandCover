"""
Build Dataset Module for LULC Classification (Phase 3)
Extracts, cleans, standardizes, and stratifies Sentinel-2 multispectral imagery
and matching ground truth label masks into train, validation, and test splits.
"""

import os
import re
import time
from pathlib import Path
import joblib
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Class mappings as defined in project specifications
CLASS_NAMES = {
    0: "Water",
    1: "Trees/Forest",
    2: "Crops",
    3: "Grassland",
    4: "Built-up",
    5: "Bare land",
    255: "Invalid/Masked"
}

FEATURE_NAMES = ["B2 (Blue)", "B3 (Green)", "B4 (Red)", "B8 (NIR)", "NDVI"]


def extract_tile_offset(filename: str) -> str:
    """Extract spatial tile offset coordinate string from filename (e.g. 0000000000-0000002048)."""
    match = re.search(r"(\d{10}-\d{10})", filename)
    return match.group(1) if match else filename


def find_dataset_directories():
    """Locate raw input and label dataset directories across project structure."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[3]  # lulc-project

    candidate_paths = [
        (
            project_root / "dataset" / "LULCzip" / "LULC" / "input",
            project_root / "dataset" / "LULCzip" / "LULC" / "labels",
        ),
        (
            script_dir.parents[1] / "data" / "input",
            script_dir.parents[1] / "data" / "labels",
        ),
    ]

    for in_dir, lbl_dir in candidate_paths:
        if in_dir.exists() and lbl_dir.exists():
            in_tifs = [f for f in os.listdir(in_dir) if f.lower().endswith(".tif")]
            lbl_tifs = [f for f in os.listdir(lbl_dir) if f.lower().endswith(".tif")]
            if len(in_tifs) > 0 and len(lbl_tifs) > 0:
                return in_dir, lbl_dir

    raise FileNotFoundError("Could not locate input and label dataset directories with GeoTIFF files.")


def load_and_flatten_tiles(input_dir: Path, label_dir: Path):
    """
    Load matching input and label GeoTIFF tile pairs, flatten pixels,
    and filter out invalid/masked pixels (255) and NaNs.
    """
    input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".tif")])
    label_files = sorted([f for f in os.listdir(label_dir) if f.lower().endswith(".tif")])

    print("=" * 80)
    print("PHASE 3 - DATASET EXTRACTION, PREPROCESSING & STRATIFIED SPLIT")
    print("=" * 80)
    print(f"Input Directory : {input_dir}")
    print(f"Label Directory : {label_dir}")
    print(f"Discovered      : {len(input_files)} input tiles and {len(label_files)} label tiles\n")

    input_map = {extract_tile_offset(f): f for f in input_files}
    label_map = {extract_tile_offset(f): f for f in label_files}

    matched_keys = sorted(list(set(input_map.keys()) & set(label_map.keys())))
    if len(matched_keys) != len(input_files):
        print(f"WARNING: Matched {len(matched_keys)} tile pairs out of {len(input_files)} input files!")

    x_list = []
    y_list = []

    total_raw_pixels = 0
    total_masked_255 = 0
    total_nan_pixels = 0
    total_valid_pixels = 0

    print("-" * 80)
    print("STEP 1: LOADING & PER-TILE PIXEL EXTRACTION")
    print("-" * 80)

    for idx, key in enumerate(matched_keys, start=1):
        in_file = input_map[key]
        lbl_file = label_map[key]

        in_path = input_dir / in_file
        lbl_path = label_dir / lbl_file

        with rasterio.open(in_path) as src_in, rasterio.open(lbl_path) as src_lbl:
            in_data = src_in.read()  # Shape: (5, Height, Width)
            lbl_data = src_lbl.read(1)  # Shape: (Height, Width)

        height, width = lbl_data.shape
        raw_count = height * width
        total_raw_pixels += raw_count

        # Transpose to (Height * Width, 5)
        # in_data shape (5, H, W) -> (H, W, 5) -> (N, 5)
        x_flat = in_data.transpose(1, 2, 0).reshape(-1, 5).astype(np.float32)
        y_flat = lbl_data.flatten().astype(np.uint8)

        # 1. Identify label 255 (invalid/masked)
        mask_255 = (y_flat == 255)
        count_255 = int(mask_255.sum())
        total_masked_255 += count_255

        # 2. Identify input NaNs
        mask_nan = np.isnan(x_flat).any(axis=1)
        count_nan = int(mask_nan.sum())
        total_nan_pixels += count_nan

        # 3. Valid pixels must be: not 255 AND not NaN
        valid_mask = (~mask_255) & (~mask_nan)
        count_valid = int(valid_mask.sum())
        total_valid_pixels += count_valid

        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]

        x_list.append(x_valid)
        y_list.append(y_valid)

        print(f"Tile {idx}/{len(matched_keys)} [{key}]:")
        print(f"  Dimensions   : {width} x {height} ({raw_count:,} pixels)")
        print(f"  Masked (255) : {count_255:10,d} px ({(count_255/raw_count)*100:6.2f}%)")
        print(f"  Input NaNs   : {count_nan:10,d} px ({(count_nan/raw_count)*100:6.2f}%)")
        print(f"  Valid Kept   : {count_valid:10,d} px ({(count_valid/raw_count)*100:6.2f}%)")
        print("-" * 60)

    print("\nExtraction Summary Across All Tiles:")
    print(f"  Total Raw Pixels       : {total_raw_pixels:,}")
    print(f"  Dropped Masked (255)   : {total_masked_255:,} ({(total_masked_255/total_raw_pixels)*100:.2f}%)")
    print(f"  Dropped Input NaNs     : {total_nan_pixels:,} ({(total_nan_pixels/total_raw_pixels)*100:.4f}%)")
    print(f"  Total Valid Retained   : {total_valid_pixels:,} ({(total_valid_pixels/total_raw_pixels)*100:.2f}%)")

    # Combine all tiles
    x_all = np.vstack(x_list)
    y_all = np.concatenate(y_list)

    return x_all, y_all, total_raw_pixels, total_masked_255, total_nan_pixels


def build_and_save_dataset():
    """Main pipeline execution for Phase 3 dataset preprocessing and splitting."""
    start_time = time.time()
    input_dir, label_dir = find_dataset_directories()

    script_dir = Path(__file__).resolve().parent
    processed_dir = script_dir.parents[1] / "data" / "processed"
    models_dir = script_dir.parents[1] / "models"

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and flatten
    x_all, y_all, total_raw, total_255, total_nan = load_and_flatten_tiles(input_dir, label_dir)

    print("\n" + "-" * 80)
    print("STEP 2: STRATIFIED TRAIN / VALIDATION / TEST SPLIT (70% / 15% / 15%)")
    print("-" * 80)

    # Split 1: 70% Train, 30% Temp (Val + Test)
    x_train_raw, x_temp_raw, y_train, y_temp = train_test_split(
        x_all,
        y_all,
        test_size=0.30,
        random_state=42,
        stratify=y_all
    )

    # Split 2: 50% / 50% of Temp -> 15% Val, 15% Test
    x_val_raw, x_test_raw, y_val, y_test = train_test_split(
        x_temp_raw,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    del x_all, y_all, x_temp_raw, y_temp  # Free memory

    print(f"Train Set Shape : X={x_train_raw.shape}, y={y_train.shape} (70.0%)")
    print(f"Val Set Shape   : X={x_val_raw.shape}, y={y_val.shape} (15.0%)")
    print(f"Test Set Shape  : X={x_test_raw.shape}, y={y_test.shape} (15.0%)")

    # 2. Scaling
    print("\n" + "-" * 80)
    print("STEP 3: FEATURE STANDARDIZATION & SCALER PERSISTENCE")
    print("-" * 80)

    scaler = StandardScaler()
    print("Fitting StandardScaler on training set (X_train)...")
    x_train = scaler.fit_transform(x_train_raw).astype(np.float32)
    x_val = scaler.transform(x_val_raw).astype(np.float32)
    x_test = scaler.transform(x_test_raw).astype(np.float32)

    del x_train_raw, x_val_raw, x_test_raw  # Free unscaled arrays

    scaler_path = models_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved fitted StandardScaler -> {scaler_path}")
    print("\nFitted Scaler Parameters per Feature:")
    for idx, name in enumerate(FEATURE_NAMES):
        print(f"  Feature {idx+1} ({name:<16s}): Mean = {scaler.mean_[idx]:8.4f}, Scale (Std) = {scaler.scale_[idx]:8.4f}")

    # 3. Save Processed Dataset Splits
    print("\n" + "-" * 80)
    print("STEP 4: SAVING COMPRESSED DATASET ARRAYS (.npz)")
    print("-" * 80)

    train_path = processed_dir / "train.npz"
    val_path = processed_dir / "val.npz"
    test_path = processed_dir / "test.npz"

    print(f"Saving {train_path.name}...")
    np.savez_compressed(train_path, X=x_train, y=y_train)

    print(f"Saving {val_path.name}...")
    np.savez_compressed(val_path, X=x_val, y=y_val)

    print(f"Saving {test_path.name}...")
    np.savez_compressed(test_path, X=x_test, y=y_test)

    print("All dataset splits saved successfully.")

    # 4. Class Distribution & Imbalance Report
    print("\n" + "-" * 80)
    print("STEP 5: DETAILED CLASS DISTRIBUTION ACROSS SPLITS")
    print("-" * 80)

    splits_data = [
        ("Train Set (70%)", y_train),
        ("Val Set   (15%)", y_val),
        ("Test Set  (15%)", y_test),
    ]

    header = f"{'Class ID':<10} {'Class Name':<16} {'Train Count (%)':<24} {'Val Count (%)':<24} {'Test Count (%)':<24}"
    print(header)
    print("-" * len(header))

    for c in range(6):
        c_name = CLASS_NAMES[c]
        counts = []
        for _, y_split in splits_data:
            c_cnt = int((y_split == c).sum())
            c_pct = (c_cnt / len(y_split)) * 100
            counts.append(f"{c_cnt:10,d} ({c_pct:5.2f}%)")

        print(f"{c:<10} {c_name:<16} {counts[0]:<24} {counts[1]:<24} {counts[2]:<24}")

    print("-" * len(header))

    # 5. Final Audit & Shapes Confirmation
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("PHASE 3 DATASET PREPARATION SUMMARY & INTEGRITY AUDIT")
    print("=" * 80)
    print(f"Total Raw Pixels Across 6 Tiles     : {total_raw:,}")
    print(f"Total Dropped Invalid (Label=255)   : {total_255:,} ({(total_255/total_raw)*100:.2f}%)")
    print(f"Total Dropped Input NaNs            : {total_nan:,} ({(total_nan/total_raw)*100:.4f}%)")
    print(f"Total Valid Clean Pixels Processed  : {len(x_train) + len(x_val) + len(x_test):,}")
    print("\nSaved Artifacts:")
    print(f"  - Scaler      : {scaler_path} (Size: {os.path.getsize(scaler_path):,} bytes)")
    print(f"  - train.npz   : X={x_train.shape} [{x_train.dtype}], y={y_train.shape} [{y_train.dtype}] ({os.path.getsize(train_path)/1e6:.2f} MB)")
    print(f"  - val.npz     : X={x_val.shape} [{x_val.dtype}], y={y_val.shape} [{y_val.dtype}] ({os.path.getsize(val_path)/1e6:.2f} MB)")
    print(f"  - test.npz    : X={x_test.shape} [{x_test.dtype}], y={y_test.shape} [{y_test.dtype}] ({os.path.getsize(test_path)/1e6:.2f} MB)")
    print(f"\nProcessing Time: {elapsed:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    build_and_save_dataset()
