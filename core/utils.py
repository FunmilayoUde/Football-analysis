from __future__ import annotations
import numpy as np


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
