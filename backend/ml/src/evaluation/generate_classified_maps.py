"""
Phase 5 - LULC Full-Tile Classified Map Generation Module
Reconstructs 2D Land-Use/Land-Cover classification maps for all Sentinel-2 input tiles,
produces color-mapped GeoTIFF/PNG outputs, generates a shared legend image,
and exports a side-by-side comparison (True Color RGB vs Ground Truth vs Model Prediction).
"""

import os
import time
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import rasterio

# Class Mapping & Color Palette (Hex & RGBA)
CLASS_METADATA = {
    0: {"name": "Water",        "color_hex": "#1E88E5", "rgba": (30, 136, 229, 255)},
    1: {"name": "Trees/Forest", "color_hex": "#2E7D32", "rgba": (46, 125, 50, 255)},
    2: {"name": "Crops",        "color_hex": "#81C784", "rgba": (129, 199, 132, 255)},
    3: {"name": "Grassland",    "color_hex": "#DCE775", "rgba": (220, 231, 117, 255)},
    4: {"name": "Built-up",     "color_hex": "#E53935", "rgba": (229, 57, 53, 255)},
    5: {"name": "Bare land",    "color_hex": "#8D6E63", "rgba": (141, 110, 99, 255)},
    255: {"name": "Masked",     "color_hex": "#000000", "rgba": (0, 0, 0, 0)}
}


def create_legend_image(output_path):
    """
    Renders a standalone, high-resolution visual legend image.
    """
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=300)
    ax.axis("off")

    patches = []
    labels = []
    for c_id in range(6):
        meta = CLASS_METADATA[c_id]
        patches.append(mpatches.Patch(color=meta["color_hex"], label=f"Class {c_id}: {meta['name']}"))
        labels.append(f"Class {c_id}: {meta['name']}")

    legend = ax.legend(
        handles=patches,
        loc="center",
        frameon=True,
        facecolor="#1e293b",
        edgecolor="#334155",
        fontsize=11,
        title="Land Use / Land Cover Classes",
        title_fontsize=13,
        labelspacing=0.8,
        handlelength=1.6,
        handleheight=1.2
    )

    # Styling legend text for dark mode theme
    plt.setp(legend.get_title(), color="white", fontweight="bold")
    for text in legend.get_texts():
        plt.setp(text, color="white", fontweight="medium")

    fig.patch.set_facecolor("#0f172a")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Saved legend graphic -> {output_path}")


def colorize_label_array(label_arr):
    """
    Transforms a 2D integer class array into a 4-channel (RGBA) uint8 image.
    """
    h, w = label_arr.shape
    rgba_img = np.zeros((h, w, 4), dtype=np.uint8)

    for c_id, meta in CLASS_METADATA.items():
        mask = (label_arr == c_id)
        rgba_img[mask] = meta["rgba"]

    return rgba_img


def generate_rgb_composite(input_raster):
    """
    Creates an enhanced True-Color (RGB: B4, B3, B2) image from 5-band Sentinel-2 tile.
    Applies 2-98% percentile contrast stretch for satellite visualization.
    """
    # Bands: 0: B2 (Blue), 1: B3 (Green), 2: B4 (Red), 3: B8 (NIR), 4: NDVI
    red = input_raster[2].astype(np.float32)
    green = input_raster[1].astype(np.float32)
    blue = input_raster[0].astype(np.float32)

    rgb = np.stack([red, green, blue], axis=-1)

    # Percentile contrast stretch per channel
    rgb_stretched = np.zeros_like(rgb, dtype=np.uint8)
    for i in range(3):
        channel = rgb[..., i]
        valid_px = channel[channel > 0]
        if len(valid_px) > 0:
            p2, p98 = np.percentile(valid_px, (2, 98))
            if p98 > p2:
                clipped = np.clip((channel - p2) / (p98 - p2), 0, 1)
                rgb_stretched[..., i] = (clipped * 255).astype(np.uint8)
            else:
                rgb_stretched[..., i] = np.clip(channel, 0, 255).astype(np.uint8)

    return rgb_stretched


def generate_all_classified_maps():
    start_time = time.time()
    script_dir = Path(__file__).resolve().parent
    base_ml_dir = script_dir.parents[1]
    
    # Locate dataset input & labels
    project_root = script_dir.parents[3]
    raw_input_dir = project_root / "dataset" / "LULCzip" / "LULC" / "input"
    raw_labels_dir = project_root / "dataset" / "LULCzip" / "LULC" / "labels"

    if not raw_input_dir.exists():
        raw_input_dir = base_ml_dir / "data" / "input"
        raw_labels_dir = base_ml_dir / "data" / "labels"
    
    models_dir = base_ml_dir / "models"
    outputs_dir = base_ml_dir / "outputs"
    maps_dir = outputs_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 5 - FULL-TILE CLASSIFIED MAP RECONSTRUCTION")
    print("=" * 80)

    # 1. Load Model & Scaler
    model_path = models_dir / "lulc_rf_model.joblib"
    scaler_path = models_dir / "scaler.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    print(f"Loading Model  : {model_path}")
    rf_model = joblib.load(model_path)
    print(f"Loading Scaler : {scaler_path}")
    scaler = joblib.load(scaler_path)

    # 2. Generate Shared Legend
    legend_png = maps_dir / "legend.png"
    create_legend_image(legend_png)

    # 3. Discover Input & Label Tiles
    input_files = sorted(list(raw_input_dir.glob("*.tif")))
    label_files = sorted(list(raw_labels_dir.glob("*.tif")))

    print(f"\nFound {len(input_files)} Sentinel-2 input tiles to classify.")

    generated_map_paths = []
    comparison_saved = False

    for idx, (inp_path, lbl_path) in enumerate(zip(input_files, label_files), start=1):
        tile_name = inp_path.stem
        print(f"\n[{idx}/{len(input_files)}] Processing Tile: {tile_name}")
        t_tile_start = time.time()

        with rasterio.open(inp_path) as src_in:
            input_raster = src_in.read()  # Shape: (5, H, W)
            profile = src_in.profile

        with rasterio.open(lbl_path) as src_lbl:
            label_raster = src_lbl.read(1)  # Shape: (H, W)

        _, height, width = input_raster.shape
        print(f"  Tile Dimensions : {width} x {height} ({width * height:,} pixels)")

        # Reshape to (N, 5)
        # Bands: B2, B3, B4, B8, NDVI
        pixel_features = input_raster.transpose(1, 2, 0).reshape(-1, 5)
        flat_labels = label_raster.reshape(-1)

        # Identify valid pixels (exclude masked/255 and nodata)
        valid_mask = (flat_labels != 255) & (~np.isnan(pixel_features).any(axis=1))
        valid_count = int(valid_mask.sum())
        print(f"  Valid Pixels    : {valid_count:,} ({(valid_count / (width * height))*100:.1f}%)")

        classified_grid = np.full((height * width), 255, dtype=np.uint8)

        if valid_count > 0:
            X_valid = pixel_features[valid_mask]
            X_scaled = scaler.transform(X_valid).astype(np.float32)

            # Predict in chunks
            chunk_size = 300_000
            preds_list = []
            for c_start in range(0, len(X_scaled), chunk_size):
                chunk = X_scaled[c_start:c_start + chunk_size]
                chunk_pred = rf_model.predict(chunk)
                preds_list.append(chunk_pred)

            classified_grid[valid_mask] = np.concatenate(preds_list).astype(np.uint8)

        classified_2d = classified_grid.reshape(height, width)

        # Colorize and Save Classified Map PNG
        classified_rgba = colorize_label_array(classified_2d)
        map_png_path = maps_dir / f"{tile_name}_classified.png"
        img = Image.fromarray(classified_rgba)
        img.save(map_png_path, "PNG", optimize=True)
        generated_map_paths.append(map_png_path)
        print(f"  Saved Classified Map -> {map_png_path} ({time.time() - t_tile_start:.2f}s)")

        # Generate Side-by-Side Comparison for Tile 1
        if not comparison_saved:
            print("  Generating side-by-side visual sanity check composite...")
            rgb_comp = generate_rgb_composite(input_raster)
            gt_rgba = colorize_label_array(label_raster)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=300)
            
            axes[0].imshow(rgb_comp)
            axes[0].set_title("Sentinel-2 True Color (RGB: B4, B3, B2)", fontsize=13, fontweight="bold")
            axes[0].axis("off")

            axes[1].imshow(gt_rgba)
            axes[1].set_title("Ground Truth LULC Labels (ESRI 10m)", fontsize=13, fontweight="bold")
            axes[1].axis("off")

            axes[2].imshow(classified_rgba)
            axes[2].set_title("Random Forest Predicted LULC Map", fontsize=13, fontweight="bold")
            axes[2].axis("off")

            plt.suptitle(f"LULC Classification Visual Comparison - {tile_name}", fontsize=15, fontweight="bold", y=0.98)
            plt.tight_layout()

            comp_png_path = maps_dir / "model_comparison_tile1.png"
            plt.savefig(comp_png_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  Saved Comparison Composite -> {comp_png_path}")
            comparison_saved = True

    # 4. Final Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("PHASE 5 - MAP GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total Tiles Classified : {len(generated_map_paths)} full tiles")
    print(f"Legend Graphic Saved   : {legend_png}")
    print(f"Comparison Saved       : {maps_dir / 'model_comparison_tile1.png'}")
    print(f"All Map PNGs Location  : {maps_dir}")
    print(f"Total Execution Time   : {total_duration_str(total_time)}")
    print("=" * 80)

    return generated_map_paths


def total_duration_str(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    return f"{seconds / 60:.2f} minutes"


if __name__ == "__main__":
    generate_all_classified_maps()
