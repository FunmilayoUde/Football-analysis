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
    FAST_MODE: bool = True
    MAX_FRAMES: int = 0  # 0 = all frames
    DETECT_EVERY_N: int = 2
    HOMOGRAPHY_EVERY_N: int = 2
    TEAM_UPDATE_EVERY_N: int = 3
    PREPROCESS_ENABLED: bool = False
    TEAM_MODE: str = "color"  # color | embedding
    SIDE_BY_SIDE_VIEW: bool = True
    LEFT_VIEW_RATIO: float = 0.62

    # Tracker tuning (applied when supported by installed supervision version)
    TRACKER_TYPE: str = "bytetrack"  # bytetrack | botsort
    TRACK_ACTIVATION_THRESHOLD: float = 0.25
    TRACK_LOW_THRESHOLD: float = 0.10
    TRACK_MIN_MATCHING_THRESHOLD: float = 0.80
    TRACK_LOST_BUFFER: int = 90
    TRACK_MIN_CONSEC_FRAMES: int = 2
    TRACK_STALE_FRAMES: int = 2
    BOTSORT_GMC_METHOD: str = "sparseOptFlow"
    BOTSORT_WITH_REID: bool = False
    BOTSORT_REID_MODEL: str = "yolo11n-cls.pt"
    BOTSORT_PROXIMITY_THRESH: float = 0.5
    BOTSORT_APPEARANCE_THRESH: float = 0.55
    ID_RELINK_FRAMES: int = 120
    ID_MEMORY_FRAMES: int = 90
    ID_RELINK_PX: float = 95.0
    ID_APP_WEIGHT: float = 0.80
    ID_AMBIGUITY_MARGIN: float = 0.08
    ID_MAX_RELINK_COST: float = 1.45
    ID_UNKNOWN_APP_PENALTY: float = 0.60
    ID_UNKNOWN_APP_MAX_GAP: int = 12
    ID_UNKNOWN_APP_MAX_SPACE_FRAC: float = 0.85
    ID_GALLERY_SIZE: int = 36
    ID_GALLERY_MIN_ADD_DIST: float = 0.10
    ID_FRAME_CONTINUITY_IOU: float = 0.45
    ID_FRAME_CONTINUITY_MARGIN: float = 0.03
    ID_SPATIAL_RESCUE_MAX_GAP: int = 220
    ID_SPATIAL_RESCUE_MARGIN: float = 0.15
    ID_SPATIAL_RESCUE_RATIO: float = 0.70
    ID_MAX_PLAYER_IDS: int = 22
    ID_MAX_GOALKEEPER_IDS: int = 2
    ID_MAX_REFEREE_IDS: int = 2
    ID_HARD_REUSE_WHEN_CAPPED: bool = True
    ID_LOCK_FRAMES: int = 12
    ID_CAPPED_REUSE_MIN_GAP: int = 8

    # Color-team classifier
    TEAM_COLOR_INIT_SAMPLES: int = 30
    TEAM_COLOR_LR: float = 0.05
    TEAM_COLOR_MIN_MARGIN: float = 0.08

    DET_CONF: float = 0.30
    DET_CONF_PLAYER: float = 0.24
    DET_CONF_REFEREE: float = 0.24
    DET_CONF_GOALKEEPER: float = 0.22
    DET_CONF_BALL: float = 0.10
    DETECT_UPSCALE: float = 1.35
    FIELD_CONF: float = 0.30
    KP_CONF: float = 0.30
    BALL_PAD_PX: int = 10
    BALL_MAX_MISSING: int = 10
    BALL_MAX_INTERP_FRAMES: int = 8
    BALL_MAX_JUMP_PX: float = 140.0
    BALL_MIN_CONF: float = 0.12

    # Tiny box filtering (ratio relative to frame area)
    MIN_AREA_RATIO_PEOPLE: float = 0.00008
    MIN_AREA_RATIO_BALL: float = 0.00001

    # Homography estimation
    H_EMA_ALPHA: float = 0.50
    RANSAC_REPROJ_THRESH: float = 25.0
    MIN_KP: int = 6
    MIN_INLIER_RATIO: float = 0.40
    MAX_REPROJ_ERR: float = 80.0
    H_HOLD_MAX_FRAMES: int = 10
    H_REINIT_FRAMES: int = 30


def load_settings() -> Settings:
    s = Settings()
    if s.HF_TOKEN:
        import os
        os.environ["HF_TOKEN"] = s.HF_TOKEN
        os.environ["HUGGINGFACE_HUB_TOKEN"] = s.HF_TOKEN
    return s
