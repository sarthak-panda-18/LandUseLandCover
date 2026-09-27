"""
Raster Utilities Module for LULC Backend
Handles GeoTIFF parsing with rasterio, NDVI computation if needed,
RGBA color-mapping of classified arrays, and Base64 PNG image encoding.
"""

import base64
import io
import logging
from typing import Dict, Optional, Tuple
import numpy as np
from PIL import Image
import rasterio
from rasterio.io import MemoryFile

logger = logging.getLogger("lulc.utils.raster")

# Color mapping matching Phase 5 specifications
CLASS_PALETTE: Dict[int, Tuple[int, int, int, int]] = {
    0: (30, 136, 229, 255),    # Water: Vibrant Blue
    1: (46, 125, 50, 255),     # Trees/Forest: Dark Green
    2: (129, 199, 132, 255),   # Crops: Light Green
    3: (220, 231, 117, 255),   # Grassland: Yellow-Green
    4: (229, 57, 53, 255),     # Built-up: Red
    5: (141, 110, 99, 255),    # Bare land: Brown
    255: (0, 0, 0, 0)          # Masked/Invalid: Transparent
}


def read_geotiff_bytes(file_bytes: bytes) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Reads GeoTIFF bytes into a 5-band numpy array (5, H, W) [B2, B3, B4, B8, NDVI].
    
    Args:
        file_bytes: Raw bytes from uploaded .tif file
        
    Returns:
        Tuple of (array of shape (5, H, W), metadata dictionary)
        
    Raises:
        ValueError: If file is corrupted, invalid format, or unsupported band count.
    """
    try:
        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:
                band_count = src.count
                height = src.height
                width = src.width
                crs = str(src.crs) if src.crs else "Unknown"
                bounds = src.bounds

                if band_count < 4:
                    raise ValueError(
                        f"Uploaded GeoTIFF has only {band_count} band(s). "
                        "Sentinel-2 LULC classification requires at least 4 bands (B2, B3, B4, B8) or 5 bands (B2, B3, B4, B8, NDVI)."
                    )

                raw_raster = src.read().astype(np.float32)  # Shape: (band_count, H, W)

                # If 5 or more bands, take the first 5 [B2, B3, B4, B8, NDVI]
                if band_count >= 5:
                    processed_raster = raw_raster[:5]
                elif band_count == 4:
                    # Automatically compute NDVI = (B8 - B4) / (B8 + B4 + 1e-6)
                    # Band 0: B2, Band 1: B3, Band 2: B4, Band 3: B8
                    b4_red = raw_raster[2]
                    b8_nir = raw_raster[3]
                    
                    denom = b8_nir + b4_red
                    ndvi = np.where(denom != 0, (b8_nir - b4_red) / (denom + 1e-6), 0.0)
                    processed_raster = np.concatenate([raw_raster, ndvi[np.newaxis, ...]], axis=0)

                metadata = {
                    "width": width,
                    "height": height,
                    "band_count": band_count,
                    "crs": crs,
                    "bounds": {
                        "left": bounds.left,
                        "bottom": bounds.bottom,
                        "right": bounds.right,
                        "top": bounds.top
                    }
                }

                return processed_raster, metadata

    except rasterio.errors.RasterioError as e:
        raise ValueError(f"Failed to decode GeoTIFF image: {str(e)}")
    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        raise ValueError(f"Error parsing uploaded raster: {str(e)}")


def classified_array_to_png_bytes(
    classified_array: np.ndarray,
    apply_smoothing: bool = False,
    blur_radius: float = 4.0
) -> bytes:
    """
    Converts a 2D integer class array (H, W) into an RGBA PNG byte buffer.
    
    IMPORTANT ARCHITECTURAL SEPARATION:
    - The classification metrics, per-class pixel counts, percentages, and analytics
      are strictly computed on the raw discrete integer class array BEFORE this function.
    - The Gaussian blur smoothing applied here is STRICTLY visual/cosmetic for display
      purposes on block-level patch predictions (rf_patch, efficientnet_patch).
    - Transparent / masked pixels (value 255) are preserved sharply by keeping the original
      binary alpha channel intact so transparency never bleeds into classified regions.
    """
    height, width = classified_array.shape
    rgba_image = np.zeros((height, width, 4), dtype=np.uint8)

    for class_id, rgba in CLASS_PALETTE.items():
        mask = (classified_array == class_id)
        rgba_image[mask] = rgba

    img = Image.fromarray(rgba_image, mode="RGBA")

    if apply_smoothing:
        from PIL import ImageFilter
        # 1. Split RGBA channels
        r, g, b, a = img.split()
        # 2. Merge RGB channels and apply Gaussian blur to the color image only
        rgb_img = Image.merge("RGB", (r, g, b))
        blurred_rgb = rgb_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        # 3. Re-attach the original sharp unblurred alpha channel (preserving transparent masked areas)
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
    Optionally applies visual-only smoothing to soften 64x64 block boundaries for patch models.
    """
    png_bytes = classified_array_to_png_bytes(
        classified_array,
        apply_smoothing=apply_smoothing,
        blur_radius=blur_radius
    )
    b64_str = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def generate_rgb_preview_base64(input_raster: np.ndarray) -> Optional[str]:
    """
    Creates a True-Color RGB satellite thumbnail (Base64) from bands (B4, B3, B2).
    """
    try:
        # Expected input shape: (5, H, W) -> Red: index 2, Green: index 1, Blue: index 0
        red = input_raster[2].astype(np.float32)
        green = input_raster[1].astype(np.float32)
        blue = input_raster[0].astype(np.float32)

        rgb = np.stack([red, green, blue], axis=-1)

        # Percentile contrast stretch per channel (2-98%)
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

        img = Image.fromarray(rgb_stretched, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"
    except Exception as e:
        logger.warning(f"Could not generate RGB preview: {e}")
        return None
