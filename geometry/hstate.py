from __future__ import annotations

import numpy as np

from core.types import HomographyResult
from geometry.homography import HomographyEstimator


class HomographyStateMachine:
    """
    Controls homography acceptance/hold/reinit so jitter and long stale holds are reduced.
    """
    def __init__(
        self,
        estimator: HomographyEstimator,
        hold_max_frames: int = 45,
        reinit_frames: int = 90,
    ):
        self.estimator = estimator
        self.hold_max_frames = int(hold_max_frames)
        self.reinit_frames = int(reinit_frames)
        self.H_current: np.ndarray | None = None
        self.fail_streak = 0
        self.ok_count = 0

    def update(self, keypoints) -> tuple[np.ndarray | None, bool, HomographyResult]:
        hres = self.estimator.estimate(keypoints)
        if hres.ok and hres.H is not None:
            self.H_current = hres.H
            self.fail_streak = 0
            self.ok_count += 1
            return self.H_current, True, hres

        self.fail_streak += 1
        if self.fail_streak > self.reinit_frames:
            self.estimator.reset()
            self.H_current = None
            self.fail_streak = 0
            return None, False, hres

        if self.H_current is not None and self.fail_streak <= self.hold_max_frames:
            return self.H_current, False, hres

        return None, False, hres
