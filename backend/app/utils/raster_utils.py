"""
Raster Utilities Module for LULC Backend
Handles streaming GeoTIFF reading with rasterio, memory-efficient True-Color preview generation,
RGBA color-mapping of classified arrays, and Base64 PNG encoding.
"""

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling

logger = logging.getLogger("lulc.utils.raster")

# Color mapping matching LULC specifications
CLASS_PALETTE: Dict[int, Tuple[int, int, int, int]] = {
    0: (30, 136, 229, 255),    # Water: Vibrant Blue
    1: (46, 125, 50, 255),     # Trees/Forest: Dark Green
    2: (129, 199, 132, 255),   # Crops: Light Green
    3: (220, 231, 117, 255),   # Grassland: Yellow-Green
    4: (229, 57, 53, 255),     # Built-up: Red
    5: (141, 110, 99, 255),    # Bare land: Brown
    255: (0, 0, 0, 0)          # Masked/Invalid: Transparent
}


def get_geotiff_metadata(raster_path: Path) -> Dict[str, Any]:
    """Reads basic spatial metadata and dimensions from GeoTIFF header without reading raster pixels."""
    with rasterio.open(raster_path) as src:
        return {
            "width": src.width,
            "height": src.height,
            "band_count": src.count,
            "crs": str(src.crs) if src.crs else "Unknown",
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top
            }
        }


def generate_rgb_preview_from_file(raster_path: Path) -> Optional[str]:
    """
    Generates True-Color RGB preview (Base64 PNG) from bands (B4, B3, B2) using a decimated read
    and percentile contrast stretching, requiring negligible memory (< 10 MB).
    """
    try:
        with rasterio.open(raster_path) as src:
            h, w = src.height, src.width
            # Read bands 3 (Red), 2 (Green), 1 (Blue) at manageable preview resolution (max 1024x1024)
            preview_max_dim = 1024
            scale_factor = min(1.0, preview_max_dim / max(h, w))
            out_h = max(1, int(h * scale_factor))
            out_w = max(1, int(w * scale_factor))

            # Decimated read directly into 3 channels float32
            rgb = src.read(
                [3, 2, 1],
                out_shape=(3, out_h, out_w),
                resampling=Resampling.bilinear
            ).astype(np.float32)

            # Transpose to (out_h, out_w, 3)
            rgb = np.transpose(rgb, (1, 2, 0))

            # Percentile contrast stretch per channel (2-98%)
            rgb_stretched = np.zeros((out_h, out_w, 3), dtype=np.uint8)
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

            img = Image.fromarray(rgb_stretched, mode="RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{b64_str}"

    except Exception as e:
        logger.warning(f"Could not generate RGB preview: {e}")
        return None


def classified_array_to_png_bytes(
    classified_array: np.ndarray,
    apply_smoothing: bool = False,
    blur_radius: float = 4.0
) -> bytes:
    """
    Converts a 2D uint8 class array (H, W) into an RGBA PNG byte buffer.
    """
    height, width = classified_array.shape
    rgba_image = np.zeros((height, width, 4), dtype=np.uint8)

    for class_id, rgba in CLASS_PALETTE.items():
        mask = (classified_array == class_id)
        rgba_image[mask] = rgba

    img = Image.fromarray(rgba_image, mode="RGBA")

    if apply_smoothing:
        from PIL import ImageFilter
        r, g, b, a = img.split()
        rgb_img = Image.merge("RGB", (r, g, b))
        blurred_rgb = rgb_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        br, bg, bb = blurred_rgb.split()
        img = Image.merge("RGBA", (br, bg, bb, a))

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def classified_array_to_base64(
    classified_array: np.ndarray,
    apply_smoothing: bool = False,
    blur_radius: float = 4.0
) -> str:
    """
    Converts a 2D integer class array into a data URL base64 string: 'data:image/png;base64,...'
    """
    png_bytes = classified_array_to_png_bytes(
        classified_array,
        apply_smoothing=apply_smoothing,
        blur_radius=blur_radius
    )
    b64_str = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"
