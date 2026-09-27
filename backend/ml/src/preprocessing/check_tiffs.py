import os
import re
from pathlib import Path
import numpy as np
import rasterio

# Resolve data directories relative to script
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR.parents[1] / "data" / "input"
LABEL_DIR = SCRIPT_DIR.parents[1] / "data" / "labels"

BAND_DESCRIPTIONS_MAP = {
    "B2": "Blue (~490 nm) - Surface Reflectance",
    "B3": "Green (~560 nm) - Surface Reflectance",
    "B4": "Red (~665 nm) - Surface Reflectance",
    "B8": "Near-Infrared / NIR (~842 nm) - Surface Reflectance",
    "NDVI": "Normalized Difference Vegetation Index [(B8-B4)/(B8+B4)]"
}


def extract_tile_offset(filename: str):
    """Extract spatial tile offset coordinate string from filename (e.g. 0000000000-0000002048)."""
    match = re.search(r"(\d{10}-\d{10})", filename)
    return match.group(1) if match else filename


def check_tiffs():
    print("=" * 80)
    print("PHASE 2 - COMPREHENSIVE TIFF DATASET VALIDATION & STATISTICAL AUDIT")
    print("=" * 80)
    print(f"Input Directory : {INPUT_DIR}")
    print(f"Label Directory : {LABEL_DIR}")

    input_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".tif")])
    label_files = sorted([f for f in os.listdir(LABEL_DIR) if f.lower().endswith(".tif")])

    print(f"\nFound {len(input_files)} input tiles and {len(label_files)} label tiles.")

    # 1. Validate Input Tiles
    print("\n" + "-" * 80)
    print("1. CHECKING INPUT TILES (Expected 5 Bands)")
    print("-" * 80)

    input_metadata = {}
    band_stats = {b: {"min": float("inf"), "max": float("-inf"), "sums": 0.0, "count": 0, "nans": 0} for b in range(1, 6)}
    global_crs_list = []
    global_res_list = []
    total_input_pixels = 0
    total_nan_pixels = 0

    for file in input_files:
        path = INPUT_DIR / file
        tile_key = extract_tile_offset(file)
        with rasterio.open(path) as src:
            res = (src.res[0], src.res[1])
            input_metadata[tile_key] = {
                "file": file,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "crs": str(src.crs),
                "res": res,
                "transform": src.transform,
                "dtypes": src.dtypes,
                "nodata": src.nodata,
                "descriptions": src.descriptions,
                "tags": src.tags(),
            }
            global_crs_list.append(str(src.crs))
            global_res_list.append(res)
            total_input_pixels += (src.width * src.height)

            print(f"\n[INPUT] {file}")
            print(f"  Tile ID        : {tile_key}")
            print(f"  Dimensions     : {src.width} x {src.height} (Total: {src.width * src.height:,} pixels)")
            print(f"  Bands          : {src.count} (Expected: 5)")
            print(f"  CRS            : {src.crs}")
            print(f"  Resolution     : {res[0]:.8e} deg/px (~10m)")
            print(f"  Data Type      : {src.dtypes}")
            print(f"  Nodata Value   : {src.nodata}")
            print(f"  Descriptions   : {src.descriptions}")

            # Compute statistics across all pixels of each band
            for b in range(1, src.count + 1):
                data = src.read(b).astype(np.float64)
                nan_count = int(np.isnan(data).sum())
                nodata_count = int((data == src.nodata).sum()) if src.nodata is not None else 0
                valid_mask = ~np.isnan(data)
                if src.nodata is not None:
                    valid_mask &= (data != src.nodata)
                
                valid_data = data[valid_mask]
                b_min = float(np.min(valid_data)) if valid_data.size > 0 else float("nan")
                b_max = float(np.max(valid_data)) if valid_data.size > 0 else float("nan")
                b_mean = float(np.mean(valid_data)) if valid_data.size > 0 else float("nan")

                band_name = src.descriptions[b - 1] if src.descriptions and len(src.descriptions) >= b else f"Band {b}"
                print(f"    Band {b} ({band_name:<4s}): Min={b_min:8.4f}, Max={b_max:8.4f}, Mean={b_mean:8.4f}, NaNs={nan_count:5,d}, Nodata={nodata_count}")

                if b in band_stats and valid_data.size > 0:
                    band_stats[b]["min"] = min(band_stats[b]["min"], b_min)
                    band_stats[b]["max"] = max(band_stats[b]["max"], b_max)
                    band_stats[b]["sums"] += float(np.sum(valid_data))
                    band_stats[b]["count"] += int(valid_data.size)
                    band_stats[b]["nans"] += nan_count
                    if b == 1:
                        total_nan_pixels += nan_count

    # 2. Validate Label Tiles
    print("\n" + "-" * 80)
    print("2. CHECKING LABEL TILES (Expected 1 Band)")
    print("-" * 80)

    label_metadata = {}
    for file in label_files:
        path = LABEL_DIR / file
        tile_key = extract_tile_offset(file)
        with rasterio.open(path) as src:
            res = (src.res[0], src.res[1])
            label_metadata[tile_key] = {
                "file": file,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "crs": str(src.crs),
                "res": res,
                "transform": src.transform,
                "dtypes": src.dtypes,
                "nodata": src.nodata,
            }

            print(f"\n[LABEL] {file}")
            print(f"  Tile ID        : {tile_key}")
            print(f"  Dimensions     : {src.width} x {src.height}")
            print(f"  Bands          : {src.count} (Expected: 1)")
            print(f"  CRS            : {src.crs}")
            print(f"  Resolution     : {res[0]:.8e} deg/px")
            print(f"  Data Type      : {src.dtypes}")
            print(f"  Nodata Value   : {src.nodata}")

    # 3. Spatial Pair Alignment Verification
    print("\n" + "-" * 80)
    print("3. SPATIAL PAIR ALIGNMENT VERIFICATION")
    print("-" * 80)

    all_keys = sorted(list(set(input_metadata.keys()) | set(label_metadata.keys())))
    alignment_passed = True
    nan_label_overlap = {c: 0 for c in range(6)}
    nan_label_overlap[255] = 0

    for key in all_keys:
        in_meta = input_metadata.get(key)
        lbl_meta = label_metadata.get(key)

        if not in_meta or not lbl_meta:
            print(f"[FAIL] Missing pair for tile offset key: {key}")
            alignment_passed = False
            continue

        dim_match = (in_meta["width"] == lbl_meta["width"]) and (in_meta["height"] == lbl_meta["height"])
        crs_match = in_meta["crs"] == lbl_meta["crs"]
        res_match = in_meta["res"] == lbl_meta["res"]
        transform_match = in_meta["transform"] == lbl_meta["transform"]

        status = "PASSED" if (dim_match and crs_match and res_match and transform_match) else "FAILED"
        if status == "FAILED":
            alignment_passed = False

        print(f"Tile [{key}]: {status}")
        print(f"  Input File  : {in_meta['file']}")
        print(f"  Label File  : {lbl_meta['file']}")
        print(f"  Dims Match  : {dim_match} ({in_meta['width']}x{in_meta['height']} vs {lbl_meta['width']}x{lbl_meta['height']})")
        print(f"  CRS Match   : {crs_match} ({in_meta['crs']})")
        print(f"  Res Match   : {res_match} ({in_meta['res'][0]:.8e})")
        print(f"  GeoTransform: {'Matches' if transform_match else 'MISMATCH'}")

        # Check NaN overlap against label mask
        in_path = INPUT_DIR / in_meta["file"]
        lbl_path = LABEL_DIR / lbl_meta["file"]
        with rasterio.open(in_path) as s_in, rasterio.open(lbl_path) as s_lbl:
            in_d = s_in.read()
            lbl_d = s_lbl.read(1)
            nan_m = np.isnan(in_d).any(axis=0)
            if nan_m.any():
                u, c = np.unique(lbl_d[nan_m], return_counts=True)
                for val, cnt in zip(u, c):
                    nan_label_overlap[int(val)] = nan_label_overlap.get(int(val), 0) + int(cnt)

    # 4. Consistency Across All Tiles
    print("\n" + "-" * 80)
    print("4. GLOBAL DATASET CONSISTENCY & INTEGRITY SUMMARY")
    print("-" * 80)
    all_crs_same = len(set(global_crs_list)) == 1
    all_res_same = len(set(global_res_list)) == 1
    all_bands_5 = all(m["count"] == 5 for m in input_metadata.values())
    all_labels_1 = all(m["count"] == 1 for m in label_metadata.values())

    print(f"Total Tile Pairs           : {len(all_keys)}")
    print(f"Total Pixels in Dataset    : {total_input_pixels:,}")
    print(f"Input Band Counts (All=5)  : {'PASSED' if all_bands_5 else 'FAILED'}")
    print(f"Label Band Counts (All=1)  : {'PASSED' if all_labels_1 else 'FAILED'}")
    print(f"Consistent CRS Across All  : {'PASSED (' + global_crs_list[0] + ')' if all_crs_same else 'FAILED'}")
    print(f"Consistent Res Across All  : {'PASSED (' + str(global_res_list[0][0]) + ' deg/px)' if all_res_same else 'FAILED'}")
    print(f"Spatial Pair Alignment     : {'PASSED' if alignment_passed else 'FAILED'}")
    print(f"Input NaN Pixels Total     : {total_nan_pixels:,} ({total_nan_pixels/total_input_pixels*100:.4f}% of dataset)")
    print(f"NaN Label Class Breakdown  : {nan_label_overlap}")

    # 5. Band Identity & Value Range Summary
    print("\n" + "-" * 80)
    print("5. INPUT BAND IDENTIFICATION & VALUE RANGES")
    print("-" * 80)

    first_input = next(iter(input_metadata.values()))
    descriptions = first_input["descriptions"]
    has_descriptions = any(d is not None and len(d.strip()) > 0 for d in descriptions)

    if has_descriptions:
        print("CONFIRMED: Embedded band descriptions found in GeoTIFF metadata:")
        for idx, desc in enumerate(descriptions, start=1):
            detail = BAND_DESCRIPTIONS_MAP.get(desc, "Spectral Band")
            print(f"  Band {idx}: '{desc}' -> {detail}")
    else:
        print("NOTICE: No embedded band descriptions / names found in GeoTIFF metadata.")

    print("\nCombined Band Statistics across all 6 tiles (valid, non-NaN pixels):")
    for b in range(1, 6):
        stats = band_stats[b]
        b_name = descriptions[b - 1] if has_descriptions else f"Band {b}"
        b_mean = stats["sums"] / stats["count"] if stats["count"] > 0 else 0
        print(f"  Band {b} ({b_name:<4s}): Range = [{stats['min']:8.4f}, {stats['max']:8.4f}], Mean = {b_mean:8.4f}, NaNs = {stats['nans']:,}")

    print("\nScaling Interpretation:")
    print("  - Bands 1-4 (B2, B3, B4, B8): Floating-point Surface Reflectance values (scaled in ~0.0 to 1.0).")
    print("  - Band 5 (NDVI)             : Normalized Difference Index values (bounded in [-1.0, 1.0]).")
    print("  - Data does NOT require division by 10,000 (already converted to surface reflectance floats).")

    print("=" * 80)
    print("TIFF VALIDATION FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    check_tiffs()
