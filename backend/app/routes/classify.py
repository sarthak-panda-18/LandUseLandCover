"""
API Routes for LULC Classification & Model Telemetry
Streams file uploads in chunks to disk and invokes windowed strip raster processing
to guarantee minimal RAM footprint (< 400 MB peak RSS).
"""

import gc
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional
import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.services.model_service import ModelService, get_model_service
from app.services.patch_model_service import PatchModelService, get_patch_model_service
from app.utils.raster_utils import (
    classified_array_to_base64,
    generate_rgb_preview_from_file,
    get_geotiff_metadata,
)

logger = logging.getLogger("lulc.routes.classify")

router = APIRouter(prefix="/api", tags=["LULC Classification"])

# Maximum allowed file upload size: 200 MB
MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024
ALLOWED_MODELS = ["pixel_rf", "rf_patch", "efficientnet_patch"]

SAMPLE_TILES = [
    {"filename": "Vijayawada_LULC_Input_2025-0000000000-0000000000.tif", "size_mb": 104.14},
    {"filename": "Vijayawada_LULC_Input_2025-0000000000-0000002048.tif", "size_mb": 105.22},
    {"filename": "Vijayawada_LULC_Input_2025-0000000000-0000004096.tif", "size_mb": 46.98},
    {"filename": "Vijayawada_LULC_Input_2025-0000002048-0000000000.tif", "size_mb": 65.85},
    {"filename": "Vijayawada_LULC_Input_2025-0000002048-0000002048.tif", "size_mb": 66.01},
    {"filename": "Vijayawada_LULC_Input_2025-0000002048-0000004096.tif", "size_mb": 29.62},
]
VALID_SAMPLE_FILENAMES = {t["filename"] for t in SAMPLE_TILES}
GITHUB_RELEASE_SAMPLE_BASE_URL = (
    "https://github.com/sarthak-panda-18/LandUseLandCover/releases/download/v1.0-sample-tiles"
)


@router.post("/classify", summary="Classify Sentinel-2 GeoTIFF Tile")
async def classify_tile_endpoint(
    file: UploadFile = File(..., description="5-band or 4-band Sentinel-2 GeoTIFF (.tif, .tiff)"),
    model: str = Form("pixel_rf", description="Classification model: 'pixel_rf', 'rf_patch', 'efficientnet_patch'"),
    model_service: ModelService = Depends(get_model_service),
    patch_model_service: PatchModelService = Depends(get_patch_model_service)
):
    """
    Accepts a Sentinel-2 GeoTIFF tile upload, streams bytes directly to a temporary file,
    applies streaming windowed classification, and returns colorized map & class analytics.
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

    # 3. Stream Uploaded File to Disk in 1MB Chunks (Never load whole payload into memory)
    tmp_path = None
    file_size_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_path = Path(tmp_file.name)
            while chunk := await file.read(1024 * 1024):  # 1 MB chunk
                file_size_bytes += len(chunk)
                if file_size_bytes > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File size exceeds maximum allowed limit of {MAX_FILE_SIZE_BYTES / (1024*1024):.0f} MB."
                    )
                tmp_file.write(chunk)

        if file_size_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes)."
            )

        # 4. Read Metadata from File Header
        try:
            geo_metadata = get_geotiff_metadata(tmp_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid or corrupted GeoTIFF file: {str(e)}"
            )

        # 5. Perform Streaming Model Inference
        try:
            if model == "pixel_rf":
                classification_result = model_service.classify_tile_streaming(tmp_path)
            else:
                classification_result = patch_model_service.classify_tile_streaming(tmp_path, model_choice=model)
        except Exception as e:
            logger.error(f"Inference error on tile {file.filename} with model {model}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model inference failed: {str(e)}"
            )

        # 6. Generate True-Color Preview & Colorized Output Map
        apply_smoothing = (model in ["rf_patch", "efficientnet_patch"])
        classified_array = classification_result["classified_array"]
        classified_base64 = classified_array_to_base64(
            classified_array,
            apply_smoothing=apply_smoothing,
            blur_radius=4.0
        )
        rgb_preview_base64 = generate_rgb_preview_from_file(tmp_path)

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

    finally:
        # Guarantee cleanup of temporary file
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception as e:
                logger.warning(f"Could not delete temporary file {tmp_path}: {e}")


@router.get("/model-info", summary="Get Trained Model Telemetry & Test Metrics")
def get_model_info_endpoint(
    model: str = Query("pixel_rf", description="Model choice: 'pixel_rf', 'rf_patch', 'efficientnet_patch'"),
    model_service: ModelService = Depends(get_model_service),
    patch_model_service: PatchModelService = Depends(get_patch_model_service)
):
    """Returns training metadata, hyperparameters, and test performance metrics."""
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


@router.get("/legend", summary="Get LULC Class Color Legend Image")
def get_legend_endpoint():
    """Returns the visual color palette legend image (PNG)."""
    base_dir = Path(__file__).resolve().parents[2]
    legend_path = base_dir / "ml" / "outputs" / "maps" / "legend.png"

    if not legend_path.exists():
        from ml.src.evaluation.generate_classified_maps import create_legend_image
        create_legend_image(legend_path)

    return FileResponse(
        path=str(legend_path),
        media_type="image/png",
        filename="lulc_legend.png"
    )


@router.get("/sample-tiles", summary="List Available Sample Sentinel-2 Tiles")
def list_sample_tiles():
    """Returns hardcoded list of available sample Sentinel-2 GeoTIFF tiles."""
    return [
        {
            "filename": tile["filename"],
            "size_mb": tile["size_mb"],
            "label": f"Tile {i+1} ({round(tile['size_mb'], 1)} MB)"
        }
        for i, tile in enumerate(SAMPLE_TILES)
    ]


@router.get("/sample-tiles/{filename}", summary="Fetch Sample Sentinel-2 Tile Bytes")
async def get_sample_tile(filename: str):
    """Streams a sample Sentinel-2 GeoTIFF tile directly from GitHub releases."""
    if filename not in VALID_SAMPLE_FILENAMES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sample tile not found"
        )

    url = f"{GITHUB_RELEASE_SAMPLE_BASE_URL}/{filename}"

    client = httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(60.0, connect=15.0))
    try:
        req = client.build_request("GET", url)
        res = await client.send(req, stream=True)
        if res.status_code != 200:
            await res.aclose()
            await client.aclose()
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to fetch sample tile from GitHub release (HTTP {res.status_code})"
            )
    except httpx.RequestError as exc:
        await client.aclose()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Network error while connecting to GitHub release: {str(exc)}"
        )

    async def stream_chunks():
        try:
            async for chunk in res.aiter_bytes(chunk_size=1024 * 1024):
                yield chunk
        finally:
            await res.aclose()
            await client.aclose()

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
    }
    if "content-length" in res.headers:
        headers["Content-Length"] = res.headers["content-length"]

    return StreamingResponse(
        stream_chunks(),
        media_type="image/tiff",
        headers=headers
    )
