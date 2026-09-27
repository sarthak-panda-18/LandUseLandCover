"""
API Routes for LULC Classification & Model Telemetry
Exposes:
- POST /api/classify     : Upload GeoTIFF, classify pixels, return base64 map and class stats
- GET  /api/model-info   : Return trained model metadata and test performance metrics
- GET  /api/legend       : Return shared class color legend PNG image
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

from app.services.model_service import ModelService, get_model_service
from app.services.patch_model_service import PatchModelService, get_patch_model_service
from app.utils.raster_utils import (
    classified_array_to_base64,
    generate_rgb_preview_base64,
    read_geotiff_bytes,
)

logger = logging.getLogger("lulc.routes.classify")

router = APIRouter(prefix="/api", tags=["LULC Classification"])

# Maximum allowed file upload size: 200 MB
MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024
ALLOWED_MODELS = ["pixel_rf", "rf_patch", "efficientnet_patch"]


@router.post("/classify", summary="Classify Sentinel-2 GeoTIFF Tile")
async def classify_tile_endpoint(
    file: UploadFile = File(..., description="5-band or 4-band Sentinel-2 GeoTIFF (.tif, .tiff)"),
    model: str = Form("pixel_rf", description="Classification model: 'pixel_rf', 'rf_patch', 'efficientnet_patch'"),
    model_service: ModelService = Depends(get_model_service),
    patch_model_service: PatchModelService = Depends(get_patch_model_service)
):
    """
    Accepts a Sentinel-2 GeoTIFF tile upload, applies the selected model (pixel-level RF or 64x64 patch models),
    and returns the colorized LULC map (Base64) alongside comprehensive per-class analytics.
    """
    start_time = time.time()

    # 1. Validate Model Selection
    if model not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model '{model}'. Allowed options are: {ALLOWED_MODELS}"
        )

    # 2. Validate File Extension
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename."
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in [".tif", ".tiff"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Only GeoTIFF tiles (.tif, .tiff) are accepted."
        )

    # 3. Read File Contents with Size Limit Check
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file payload: {str(e)}"
        )

    file_size_bytes = len(contents)
    if file_size_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        file_mb = file_size_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({file_mb:.1f} MB) exceeds maximum allowed limit of {max_mb:.0f} MB."
        )

    # 4. Parse GeoTIFF with Rasterio
    try:
        raster_data, geo_metadata = read_geotiff_bytes(contents)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while parsing GeoTIFF: {str(e)}"
        )

    # 5. Perform Model Inference (Routed by model parameter)
    try:
        if model == "pixel_rf":
            classification_result = model_service.classify_tile(raster_data)
        else:
            classification_result = patch_model_service.classify_tile_patches(raster_data, model_choice=model)
    except Exception as e:
        logger.error(f"Inference error on tile {file.filename} with model {model}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {str(e)}"
        )

    # 6. Generate Visual Map Outputs (Base64)
    # Visual smoothing is applied ONLY to rendered preview PNGs for patch-based models (rf_patch, efficientnet_patch).
    # All underlying classification arrays, per-class counts, percentages, and metrics remain strictly untouched.
    apply_smoothing = (model in ["rf_patch", "efficientnet_patch"])
    classified_array = classification_result["classified_array"]
    classified_base64 = classified_array_to_base64(
        classified_array,
        apply_smoothing=apply_smoothing,
        blur_radius=4.0
    )
    rgb_preview_base64 = generate_rgb_preview_base64(raster_data)

    total_processing_time = round(time.time() - start_time, 3)

    actual_resolution = (
        "64x64 blocks (visual smoothing applied for display only)"
        if apply_smoothing
        else "10m Native Pixel Resolution"
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "success": True,
            "filename": file.filename,
            "model_used": model,
            "visual_smoothing_applied": apply_smoothing,
            "actual_resolution": actual_resolution,
            "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
            "processing_time_seconds": total_processing_time,
            "dimensions": classification_result["dimensions"],
            "total_pixels": classification_result["total_pixels"],
            "valid_pixels": classification_result["valid_pixels"],
            "invalid_pixels": classification_result["invalid_pixels"],
            "valid_percentage": classification_result["valid_percentage"],
            "masked_percentage": classification_result["masked_percentage"],
            "class_distribution": classification_result["class_distribution"],
            "patch_analytics": classification_result.get("patch_analytics"),
            "geo_metadata": geo_metadata,
            "classified_image_base64": classified_base64,
            "rgb_preview_base64": rgb_preview_base64
        }
    )


@router.get("/model-info", summary="Get Trained Model Telemetry & Test Metrics")
def get_model_info_endpoint(
    model: str = Query("pixel_rf", description="Model choice: 'pixel_rf', 'rf_patch', 'efficientnet_patch'"),
    model_service: ModelService = Depends(get_model_service),
    patch_model_service: PatchModelService = Depends(get_patch_model_service)
):
    """
    Returns training metadata, hyperparameters, and test performance metrics
    for the selected model ('pixel_rf', 'rf_patch', 'efficientnet_patch').
    """
    if model not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model '{model}'. Allowed options are: {ALLOWED_MODELS}"
        )

    try:
        if model == "pixel_rf":
            info = model_service.get_model_info()
            info["model_id"] = "pixel_rf"
            info["model_name"] = "Random Forest (Pixel-Level)"
            info["granularity"] = "Pixel-Level (10m Resolution)"
        else:
            info = patch_model_service.get_model_info(model)

        return JSONResponse(status_code=status.HTTP_200_OK, content=info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model telemetry for '{model}': {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model telemetry: {str(e)}"
        )


@router.get("/legend", summary="Get LULC Class Color Legend Image")
def get_legend_endpoint():
    """
    Returns the visual color palette legend image (PNG) for the 6 LULC classes.
    """
    base_dir = Path(__file__).resolve().parents[2]
    legend_path = base_dir / "ml" / "outputs" / "maps" / "legend.png"

    if not legend_path.exists():
        # Fallback to creating the legend if missing
        from ml.src.evaluation.generate_classified_maps import create_legend_image
        create_legend_image(legend_path)

    return FileResponse(
        path=str(legend_path),
        media_type="image/png",
        filename="lulc_legend.png"
    )


@router.get("/sample-tiles", summary="List Available Sample Sentinel-2 Tiles")
def list_sample_tiles():
    """Returns list of bundled sample Sentinel-2 GeoTIFF tiles for quick testing."""
    project_root = Path(__file__).resolve().parents[3]
    input_dir = project_root / "dataset" / "LULCzip" / "LULC" / "input"
    if not input_dir.exists():
        return []
    files = sorted(list(input_dir.glob("*.tif")))
    return [
        {
            "filename": f.name,
            "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
            "label": f"Tile {i+1} ({round(f.stat().st_size / (1024 * 1024), 1)} MB)"
        }
        for i, f in enumerate(files)
    ]


@router.get("/sample-tiles/{filename}", summary="Fetch Sample Sentinel-2 Tile Bytes")
def get_sample_tile(filename: str):
    """Streams a sample Sentinel-2 GeoTIFF tile."""
    project_root = Path(__file__).resolve().parents[3]
    input_dir = project_root / "dataset" / "LULCzip" / "LULC" / "input"
    target_path = input_dir / filename
    if not target_path.exists() or not target_path.name.endswith(".tif"):
        raise HTTPException(status_code=404, detail="Sample tile not found")
    return FileResponse(path=str(target_path), media_type="image/tiff", filename=filename)

