from __future__ import annotations
import logging
import cv2
import numpy as np

logger = logging.getLogger("fda")


def normalize_h(H: np.ndarray) -> np.ndarray:
    """Normalize homography so H[2,2] == 1 when possible."""
    H = H.astype(np.float64)
    if abs(H[2, 2]) > 1e-9:
        H = H / H[2, 2]
    return H


def ema_matrix(prev: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
    """
    EMA smoothing: higher alpha means more weight on previous (more smoothing).
    H_smooth = alpha*prev + (1-alpha)*new
    """
    return alpha * prev + (1.0 - alpha) * new


def detect_camera_motion(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    threshold: float = 8.0,
) -> bool:
    """Fast camera motion check via downsampled mean absolute difference."""
    small_prev = cv2.resize(prev_frame, (160, 90))
    small_curr = cv2.resize(curr_frame, (160, 90))
    diff = np.mean(np.abs(small_curr.astype(np.float32) - small_prev.astype(np.float32)))
    return float(diff) > threshold


def log_frame_diagnostics(
    frame_idx: int,
    tracker_stats: dict,
    hres,
    log_every: int = 50,
) -> None:
    """Log tracker and homography diagnostics every N frames."""
    if frame_idx % log_every != 0:
        return

    t = tracker_stats
    logger.info(
        "frame=%d | trk: in=%d matched=%d unmatched=%d reid_recover=%d lost=%d | "
        "hom: ok=%s kp=%d inlier=%.2f reproj=%.1f fallback_age=%d degraded=%s",
        frame_idx,
        t.get("input_detections", 0),
        t.get("matched_tracks", 0),
        t.get("unmatched_detections", 0),
        t.get("reid_recoveries", 0),
        t.get("lost_tracks", 0),
        hres.ok,
        hres.n_points,
        hres.inlier_ratio,
        hres.reproj_err,
        hres.fallback_age,
        hres.degraded,
    )
