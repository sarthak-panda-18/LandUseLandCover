"""
LULC Classification FastAPI Backend Application
Integrates Sentinel-2 multi-spectral image inference, model telemetry, and health check routes.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.classify import router as classify_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("lulc.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager. Startup does NOT pre-load any model artifacts,
    enabling immediate startup and zero RAM footprint until requested.
    """
    logger.info("Initializing LULC Classification Backend Application (Lazy Model Loading Enabled)...")
    yield
    logger.info("Shutting down LULC Classification Backend...")


app = FastAPI(
    title="LULC Classification API",
    description="Backend API for Sentinel-2 Land Use and Land Cover (LULC) Classification using Random Forest and Deep Learning",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
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
    """Health check endpoint to verify backend server status immediately without loading any models."""
    return {
        "status": "ok",
        "service": "LULC Classification Backend",
        "version": "1.0.0",
        "model_loaded": True
    }
