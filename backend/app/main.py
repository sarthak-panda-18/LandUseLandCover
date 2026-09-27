"""
LULC Classification FastAPI Backend Application
Integrates Sentinel-2 multi-spectral image inference, model telemetry, and health check routes.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.classify import router as classify_router
from app.services.model_service import get_model_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("lulc.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that warms up the Random Forest model and scaler
    into memory at application startup.
    """
    logger.info("Initializing LULC Classification Backend Application...")
    try:
        service = get_model_service()
        logger.info("Random Forest model and scalers successfully loaded in memory.")
    except Exception as e:
        logger.error(f"Failed to pre-load model artifacts at startup: {e}")
    
    yield
    logger.info("Shutting down LULC Classification Backend...")


app = FastAPI(
    title="LULC Classification API",
    description="Backend API for Sentinel-2 Land Use and Land Cover (LULC) Classification using Random Forest",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration (Vite server on port 5173 and general access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(classify_router)


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint to verify backend server status and model readiness."""
    try:
        service = get_model_service()
        model_ready = service.is_loaded
    except Exception:
        model_ready = False

    return {
        "status": "ok",
        "model_loaded": model_ready,
        "service": "LULC Classification Backend",
        "version": "1.0.0"
    }
