"""
Model Manager Singleton
Coordinates model lifecycle so that AT MOST ONE heavy ML model is resident in RAM at any time.
When a new model is requested, any previously resident model is deleted and garbage-collected.
"""

import gc
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib

logger = logging.getLogger("lulc.service.model_manager")


class ModelManager:
    _instance: Optional["ModelManager"] = None

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[2]  # backend/
        self.models_dir = self.base_dir / "ml" / "models"
        self.patch_models_dir = self.base_dir / "ml" / "models_patch"

        self.pixel_model_path = self.models_dir / "lulc_rf_model.joblib"
        self.scaler_path = self.models_dir / "scaler.joblib"
        self.rf_patch_path = self.patch_models_dir / "random_forest_patch.pkl"
        self.onnx_patch_path = self.patch_models_dir / "efficientnetb0_patch.onnx"

        # Residency tracking
        self.current_model_type: Optional[str] = None
        self.active_model: Any = None
        self.active_scaler: Any = None
        self.active_session: Any = None

    def unload_current_model(self):
        """Releases memory held by any currently resident model."""
        if self.current_model_type is not None:
            logger.info(f"[ModelManager] Unloading '{self.current_model_type}' from RAM...")
            self.active_model = None
            self.active_scaler = None
            self.active_session = None
            self.current_model_type = None
            gc.collect()

    def get_pixel_rf_model(self) -> Tuple[Any, Any]:
        """Loads and returns (scaler, rf_model) with n_jobs=1. Ensures single model residency."""
        if self.current_model_type == "pixel_rf" and self.active_model is not None and self.active_scaler is not None:
            return self.active_scaler, self.active_model

        self.unload_current_model()
        logger.info(f"[ModelManager] Loading pixel RF model & scaler from {self.models_dir}...")

        # Prefer v2 compact model if present, fallback to v1
        v2_model_path = self.models_dir / "lulc_rf_model_v2.joblib"
        target_model_path = v2_model_path if v2_model_path.exists() else self.pixel_model_path

        if not target_model_path.exists():
            raise FileNotFoundError(f"Trained model artifact not found at {target_model_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler artifact not found at {self.scaler_path}")

        scaler = joblib.load(self.scaler_path)
        rf_model = joblib.load(target_model_path)
        # Prevent worker subprocess memory duplication
        if hasattr(rf_model, "n_jobs"):
            rf_model.n_jobs = 1

        self.active_scaler = scaler
        self.active_model = rf_model
        self.current_model_type = "pixel_rf"
        return self.active_scaler, self.active_model

    def get_rf_patch_model(self) -> Any:
        """Loads and returns rf_patch model with n_jobs=1. Ensures single model residency."""
        if self.current_model_type == "rf_patch" and self.active_model is not None:
            return self.active_model

        self.unload_current_model()
        logger.info(f"[ModelManager] Loading RF patch model from {self.rf_patch_path}...")

        if not self.rf_patch_path.exists():
            raise FileNotFoundError(f"RF patch model not found at {self.rf_patch_path}")

        rf_patch_model = joblib.load(self.rf_patch_path)
        if hasattr(rf_patch_model, "n_jobs"):
            rf_patch_model.n_jobs = 1

        self.active_model = rf_patch_model
        self.current_model_type = "rf_patch"
        return self.active_model

    def get_efficientnet_onnx_session(self) -> Any:
        """Loads and returns ONNX Runtime InferenceSession (single thread). Ensures single model residency."""
        if self.current_model_type == "efficientnet_patch" and self.active_session is not None:
            return self.active_session

        self.unload_current_model()
        logger.info(f"[ModelManager] Loading ONNX Runtime session for EfficientNetB0 from {self.onnx_patch_path}...")

        if not self.onnx_patch_path.exists():
            raise FileNotFoundError(f"ONNX patch model not found at {self.onnx_patch_path}")

        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(str(self.onnx_patch_path), opts, providers=["CPUExecutionProvider"])
        self.active_session = session
        self.current_model_type = "efficientnet_patch"
        return self.active_session


_model_manager_instance: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    global _model_manager_instance
    if _model_manager_instance is None:
        _model_manager_instance = ModelManager()
    return _model_manager_instance
