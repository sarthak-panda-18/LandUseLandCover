import os
import re
from pathlib import Path
import numpy as np
import rasterio

# Resolve data directories relative to script
SCRIPT_DIR = Path(__file__).resolve().parent
LABEL_DIR = SCRIPT_DIR.parents[1] / "data" / "labels"
INPUT_DIR = SCRIPT_DIR.parents[1] / "data" / "input"

CLASS_NAMES = {
    0: "Water",
    1: "Trees/Forest",
    2: "Crops",
    3: "Grassland",
    4: "Built-up",
    5: "Bare land",
    255: "Invalid/Masked"
}


def extract_tile_offset(filename: str):
    """Extract spatial tile offset coordinate string from filename."""
    match = re.search(r"(\d{10}-\d{10})", filename)
    return match.group(1) if match else filename


def check_label_classes():
    print("=" * 80)
    print("PHASE 2 - LULC LABEL CLASS DISTRIBUTION & PIXEL VALIDATION")
    print("=" * 80)
    print(f"Label Directory : {LABEL_DIR}")

    label_files = sorted([f for f in os.listdir(LABEL_DIR) if f.lower().endswith(".tif")])
    print(f"\nFound {len(label_files)} label files.\n")

    total_counts = {value: 0 for value in CLASS_NAMES}
    per_file_stats = {}
    unexpected_values = set()

    for file in label_files:
        path = LABEL_DIR / file
        tile_key = extract_tile_offset(file)

        with rasterio.open(path) as src:
            label_data = src.read(1)

        unique, counts = np.unique(label_data, return_counts=True)
        file_counts = {val: 0 for val in CLASS_NAMES}
        file_total = int(label_data.size)

        print(f"[LABEL FILE] {file}")
        print(f"  Tile ID        : {tile_key}")
        print(f"  Total Pixels   : {file_total:,} ({label_data.shape[1]}x{label_data.shape[0]})")
        print("  Class Breakdown:")

        for val, cnt in zip(unique, counts):
            val = int(val)
            cnt = int(cnt)
            if val in CLASS_NAMES:
                file_counts[val] = cnt
                total_counts[val] += cnt
            else:
                unexpected_values.add(val)
                print(f"    WARNING: Unexpected label value {val} with {cnt:,} pixels!")

        for class_id in range(6):
            cnt = file_counts[class_id]
            pct = (cnt / file_total) * 100
            print(f"    Class {class_id} ({CLASS_NAMES[class_id]:12s}): {cnt:10,} px ({pct:6.2f}%)")

        inv_cnt = file_counts[255]
        inv_pct = (inv_cnt / file_total) * 100
        print(f"    Value 255 (Invalid/Masked): {inv_cnt:10,} px ({inv_pct:6.2f}%)")
        print("-" * 60)

        per_file_stats[tile_key] = file_counts

    # Combined Summary across all 6 tiles
    print("\n" + "=" * 80)
    print("COMBINED LULC CLASS DISTRIBUTION (ALL 6 TILES)")
    print("=" * 80)

    valid_pixels = sum(total_counts[i] for i in range(6))
    invalid_pixels = total_counts[255]
    total_pixels = valid_pixels + invalid_pixels

    print(f"\nTotal Pixels Across All 6 Tiles : {total_pixels:,}")
    print(f"Total Valid (Trainable) Pixels  : {valid_pixels:,} ({valid_pixels/total_pixels*100:.2f}% of total)")
    print(f"Total Invalid / Masked (255)    : {invalid_pixels:,} ({invalid_pixels/total_pixels*100:.2f}% of total)")
    print(f"Unexpected Label Values Found   : {list(unexpected_values) if unexpected_values else 'None (Clean)'}")

    print("\n" + "-" * 70)
    print(f"{'Class ID':<10} {'Class Name':<16} {'Pixel Count':<16} {'% of Valid':<14} {'% of Total':<14}")
    print("-" * 70)

    for class_id in range(6):
        cnt = total_counts[class_id]
        pct_valid = (cnt / valid_pixels * 100) if valid_pixels > 0 else 0
        pct_total = (cnt / total_pixels * 100) if total_pixels > 0 else 0
        print(f"{class_id:<10} {CLASS_NAMES[class_id]:<16} {cnt:<16,} {pct_valid:<14.2f}% {pct_total:<14.2f}%")

    print("-" * 70)

    # Class Imbalance Analysis
    max_class_id = max(range(6), key=lambda c: total_counts[c])
    min_class_id = min(range(6), key=lambda c: total_counts[c])
    imbalance_ratio = total_counts[max_class_id] / total_counts[min_class_id] if total_counts[min_class_id] > 0 else float("inf")

    print("\nClass Imbalance Assessment:")
    print(f"  Majority Class : Class {max_class_id} ({CLASS_NAMES[max_class_id]}) with {total_counts[max_class_id]:,} px ({total_counts[max_class_id]/valid_pixels*100:.2f}% of valid)")
    print(f"  Minority Class : Class {min_class_id} ({CLASS_NAMES[min_class_id]}) with {total_counts[min_class_id]:,} px ({total_counts[min_class_id]/valid_pixels*100:.2f}% of valid)")
    print(f"  Imbalance Ratio: {imbalance_ratio:.2f}x difference between majority and minority classes")

    print("=" * 80)
    print("LABEL VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    check_label_classes()
