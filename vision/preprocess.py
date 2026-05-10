from __future__ import annotations

import cv2
import numpy as np


def enhance_frame(frame: np.ndarray, enabled: bool = True) -> np.ndarray:
    """
    Low-cost enhancement for low-quality broadcast frames.
    """
    if not enabled:
        return frame

    # Keep this lightweight so it does not dominate runtime.
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    out = cv2.filter2D(out, -1, kernel)
    return out
