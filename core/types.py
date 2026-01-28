from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class HomographyResult:
    H: np.ndarray | None
    ok: bool
    n_points: int
    inlier_ratio: float
    reproj_err: float
