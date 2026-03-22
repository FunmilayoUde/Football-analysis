from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ROBOFLOW_API_KEY: str = Field(..., description="Roboflow API key for hosted inference")
    HF_TOKEN: str | None = Field(default=None, description="Hugging Face token (optional)")

    PLAYER_MODEL_ID: str = "football-players-detection-3zvbc/11"
    FIELD_MODEL_ID: str = "football-field-detection-f07vi/14"

    # Runtime
    DEVICE: str = "cpu"

    DET_CONF: float = 0.30
    FIELD_CONF: float = 0.30
    KP_CONF: float = 0.40
    BALL_PAD_PX: int = 10

    # Tiny box filtering (ratio relative to frame area)
    MIN_AREA_RATIO_PEOPLE: float = 0.00008
    MIN_AREA_RATIO_BALL: float = 0.00001

    # Tracker (ByteTrack)
    TRACK_ACTIVATION_THRESHOLD: float = 0.25
    LOST_TRACK_BUFFER: int = 60
    MINIMUM_MATCHING_THRESHOLD: float = 0.80
    TRACK_FRAME_RATE: int = 30

    # ReID
    REID_ENABLED: bool = True
    REID_MATCH_THRESHOLD: float = 0.40
    REID_GALLERY_ALPHA: float = 0.90
    REID_LOST_BUFFER: int = 90

    # Homography estimation
    H_EMA_ALPHA: float = 0.80
    RANSAC_REPROJ_THRESH: float = 3.0
    MIN_KP: int = 5
    MIN_INLIER_RATIO: float = 0.35
    MAX_REPROJ_ERR: float = 10.0
    H_JUMP_GATE_PX: float = 50.0
    H_FALLBACK_EXPIRY: int = 30

    # Keypoint skip / adaptive scheduling
    KP_SKIP_FRAMES: int = 3
    KP_MOTION_THRESHOLD: float = 8.0

    # ReID inference
    REID_FP16: bool = True


def load_settings() -> Settings:
    s = Settings()
    if s.HF_TOKEN:
        import os
        os.environ["HF_TOKEN"] = s.HF_TOKEN
        os.environ["HUGGINGFACE_HUB_TOKEN"] = s.HF_TOKEN
    return s
