"""
API Routes for LULC Classification & Model Telemetry
Streams file uploads in chunks to disk and processes classification jobs asynchronously
with a single-job concurrency lock to guarantee minimal RAM footprint (< 400 MB peak RSS).
"""

import asyncio
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

from app.services.job_manager import JobManager, get_job_manager
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
MAX_QUEUED_JOBS = 10

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


def _execute_classification_sync(
    tmp_path: Path,
    model: str,
    original_filename: str,
    file_size_bytes: int,
    start_time: float,
) -> Dict[str, Any]:
    """
    Synchronous worker executed in a threadpool to perform metadata extraction,
    streaming ML inference, colorization, and true-color preview rendering.
    """
    # 1. Read Metadata from File Header
    try:
        geo_metadata = get_geotiff_metadata(tmp_path)
    except Exception as e:
        raise ValueError(f"Invalid or corrupted GeoTIFF file: {str(e)}")

    # 2. Perform Streaming Model Inference
    model_service = get_model_service()
    patch_model_service = get_patch_model_service()

    try:
        if model == "pixel_rf":
            classification_result = model_service.classify_tile_streaming(tmp_path)
        else:
            classification_result = patch_model_service.classify_tile_streaming(tmp_path, model_choice=model)
    except Exception as e:
        logger.error(f"Inference error on tile {original_filename} with model {model}: {e}", exc_info=True)
        raise RuntimeError(f"Model inference failed: {str(e)}")

    # 3. Generate True-Color Preview & Colorized Output Map
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

    return {
        "success": True,
        "filename": original_filename,
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
        "rgb_preview_base64": rgb_preview_base64,
    }


async def _run_classification_job_async(
    job_id: str,
    tmp_path: Path,
    model: str,
    original_filename: str,
    file_size_bytes: int,
    start_time: float,
) -> None:
    """
    Background worker that acquires the single-instance lock, updates job status to processing,
    runs the classification in a worker thread, and writes the final result or error.
    Guarantees temporary file cleanup in the finally block.
    """
    job_mgr = get_job_manager()
    lock = job_mgr.get_lock()

    try:
        async with lock:
            job_mgr.update_job(job_id, status="processing")
            logger.info(
                f"[Job {job_id}] Lock acquired, status -> processing. File: {original_filename} "
                f"({file_size_bytes / (1024 * 1024):.2f} MB), Model: {model}"
            )

            # Run synchronous inference in threadpool so FastAPI event loop stays completely responsive
            result = await asyncio.to_thread(
                _execute_classification_sync,
                tmp_path=tmp_path,
                model=model,
                original_filename=original_filename,
                file_size_bytes=file_size_bytes,
                start_time=start_time,
            )

            job_mgr.update_job(job_id, status="done", result=result)
            logger.info(f"[Job {job_id}] Finished successfully in {time.time() - start_time:.2f}s")

    except Exception as e:
        err_msg = str(e)
        logger.error(f"[Job {job_id}] Failed with error: {err_msg}", exc_info=True)
        job_mgr.update_job(job_id, status="error", error=err_msg)

    finally:
        # Guarantee cleanup of temporary uploaded file
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
                logger.info(f"[Job {job_id}] Cleaned up temporary file {tmp_path}")
            except Exception as e:
                logger.warning(f"[Job {job_id}] Could not delete temporary file {tmp_path}: {e}")


@router.post("/classify", status_code=status.HTTP_202_ACCEPTED, summary="Classify Sentinel-2 GeoTIFF Tile (Async Job)")
async def classify_tile_endpoint(
    file: UploadFile = File(..., description="5-band or 4-band Sentinel-2 GeoTIFF (.tif, .tiff)"),
    model: str = Form("pixel_rf", description="Classification model: 'pixel_rf', 'rf_patch', 'efficientnet_patch'"),
    job_mgr: JobManager = Depends(get_job_manager),
):
    """
    Accepts a Sentinel-2 GeoTIFF tile upload, streams bytes directly to disk, creates an async job,
    and immediately returns 202 Accepted with a unique job_id (< 1 second response time).
    """
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

    # 3. Check Queue Capacity (prevent unbounded memory exhaustion)
    if job_mgr.get_active_job_count() >= MAX_QUEUED_JOBS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Another classification is in progress, please wait"
        )

    # 4. Stream Uploaded File to Disk in 1MB Chunks
    tmp_path = None
    file_size_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_path = Path(tmp_file.name)
            while chunk := await file.read(1024 * 1024):  # 1 MB chunk
                file_size_bytes += len(chunk)
                if file_size_bytes > MAX_FILE_SIZE_BYTES:
                    if tmp_path and tmp_path.exists():
                        tmp_path.unlink()
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File size exceeds maximum allowed limit of {MAX_FILE_SIZE_BYTES / (1024*1024):.0f} MB."
                    )
                tmp_file.write(chunk)

        if file_size_bytes == 0:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes)."
            )

        # 5. Create Job Record
        job_id = job_mgr.create_job()
        start_time = time.time()

        # 6. Launch Background Worker Task
        task = asyncio.create_task(
            _run_classification_job_async(
                job_id=job_id,
                tmp_path=tmp_path,
                model=model,
                original_filename=file.filename,
                file_size_bytes=file_size_bytes,
                start_time=start_time,
            )
        )
        job_mgr.track_task(task)

        # 7. Immediately return 202 Accepted
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "job_id": job_id,
                "status": "pending"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        logger.error(f"Error handling file upload for {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process uploaded file: {str(e)}"
        )


@router.get("/classify/status/{job_id}", summary="Get Classification Job Status")
def get_classification_status_endpoint(
    job_id: str,
    job_mgr: JobManager = Depends(get_job_manager)
):
    """
    Returns the current execution status and result (or error) for an asynchronous classification job.
    """
    job = job_mgr.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Classification job '{job_id}' not found or expired."
        )
    return JSONResponse(status_code=status.HTTP_200_OK, content=job)


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
