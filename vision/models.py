from __future__ import annotations
import os
from pathlib import Path

# Disable optional model families we do not use in this project.
# This reduces import-time warnings and avoids loading unnecessary dependencies.


def _best_execution_provider() -> str:
    # CoreML is excluded: it is listed as available on Mac but the RF-DETR model
    # architecture used here triggers an unsupported-node error at runtime.
    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in available:
            print("[models] Using ONNX execution provider: CUDAExecutionProvider")
            return "CUDAExecutionProvider,CPUExecutionProvider"
    except Exception:
        pass
    return "CPUExecutionProvider"


os.environ.setdefault("ONNXRUNTIME_EXECUTION_PROVIDERS", _best_execution_provider())
os.environ.setdefault("API_CALLS_TIMEOUT", "30")
os.environ.setdefault("API_CALLS_MAX_TRIES", "8")
os.environ.setdefault(
    "INFERENCE_HOME",
    str(Path(__file__).resolve().parents[1] / ".inference-cache"),
)
os.environ.setdefault("CORE_MODEL_SAM_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_SAM2_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_SAM3_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_GAZE_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_YOLO_WORLD_ENABLED", "False")
os.environ.setdefault("PALIGEMMA_ENABLED", "False")
os.environ.setdefault("FLORENCE2_ENABLED", "False")
os.environ.setdefault("QWEN_2_5_ENABLED", "False")
os.environ.setdefault("QWEN_3_ENABLED", "False")
os.environ.setdefault("SMOLVLM2_ENABLED", "False")
os.environ.setdefault("DEPTH_ESTIMATION_ENABLED", "False")
os.environ.setdefault("MOONDREAM2_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_TROCR_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_GROUNDINGDINO_ENABLED", "False")

from inference import get_model


def load_roboflow_models(api_key: str, player_model_id: str, field_model_id: str):
    player_model = get_model(model_id=player_model_id, api_key=api_key)
    field_model = get_model(model_id=field_model_id, api_key=api_key)
    return player_model, field_model
