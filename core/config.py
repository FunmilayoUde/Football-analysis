from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Keys
    ROBOFLOW_API_KEY: str = Field(..., description="Roboflow API key for hosted inference")
    HF_TOKEN: str | None = Field(default=None, description="Hugging Face token (optional)")

    # Models (from the notebook)
    PLAYER_MODEL_ID: str = "football-players-detection-3zvbc/11"
    FIELD_MODEL_ID: str = "football-field-detection-f07vi/14"

    # Runtime
    DEVICE: str = "cpu"

    # Detection thresholds
    DET_CONF: float = 0.30
    FIELD_CONF: float = 0.30
    KP_CONF: float = 0.50
    BALL_PAD_PX: int = 10

    # Tiny box filtering (ratio relative to frame area)
    MIN_AREA_RATIO_PEOPLE: float = 0.00008
    MIN_AREA_RATIO_BALL: float = 0.00001

    # Homography estimation
    H_EMA_ALPHA: float = 0.80
    RANSAC_REPROJ_THRESH: float = 3.0
    MIN_KP: int = 6
    MIN_INLIER_RATIO: float = 0.55
    MAX_REPROJ_ERR: float = 8.0


def load_settings() -> Settings:
    s = Settings()
    # Make HF token available to huggingface/transformers if provided
    if s.HF_TOKEN:
        import os
        os.environ["HF_TOKEN"] = s.HF_TOKEN
        os.environ["HUGGINGFACE_HUB_TOKEN"] = s.HF_TOKEN
    return s
