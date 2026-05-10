from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import supervision as sv


@dataclass
class BallTrackState:
    center: np.ndarray | None = None
    box_wh: np.ndarray | None = None
    missing: int = 0


class BallSmoother:
    def __init__(
        self,
        ball_id: int,
        max_missing: int = 8,
        max_interp_frames: int = 6,
        max_jump_px: float = 120.0,
        min_conf: float = 0.10,
    ):
        self.ball_id = int(ball_id)
        self.max_missing = int(max_missing)
        self.max_interp_frames = int(max_interp_frames)
        self.max_jump_px = float(max_jump_px)
        self.min_conf = float(min_conf)
        self.state = BallTrackState()

    @staticmethod
    def _centers_xyxy(xyxy: np.ndarray) -> np.ndarray:
        cx = 0.5 * (xyxy[:, 0] + xyxy[:, 2])
        cy = 0.5 * (xyxy[:, 1] + xyxy[:, 3])
        return np.stack([cx, cy], axis=1)

    def _make_detection(self, center: np.ndarray, wh: np.ndarray) -> sv.Detections:
        half = 0.5 * wh
        x1y1 = center - half
        x2y2 = center + half
        xyxy = np.array([[x1y1[0], x1y1[1], x2y2[0], x2y2[1]]], dtype=np.float32)
        return sv.Detections(
            xyxy=xyxy,
            confidence=np.array([1.0], dtype=np.float32),
            class_id=np.array([self.ball_id], dtype=np.int32),
        )

    def update(self, ball_det: sv.Detections) -> tuple[sv.Detections, bool]:
        if len(ball_det) > 0:
            centers = self._centers_xyxy(ball_det.xyxy)
            widths = ball_det.xyxy[:, 2] - ball_det.xyxy[:, 0]
            heights = ball_det.xyxy[:, 3] - ball_det.xyxy[:, 1]
            wh = np.stack([widths, heights], axis=1)

            if self.state.center is None:
                idx = int(np.argmax(ball_det.confidence)) if ball_det.confidence is not None else 0
            else:
                d = np.linalg.norm(centers - self.state.center.reshape(1, 2), axis=1)
                idx = int(np.argmin(d))
                if d[idx] > self.max_jump_px:
                    idx = int(np.argmax(ball_det.confidence)) if ball_det.confidence is not None else idx

            chosen = ball_det[idx:idx + 1]
            if chosen.confidence is not None and float(chosen.confidence[0]) < self.min_conf:
                chosen = sv.Detections(
                    xyxy=np.zeros((0, 4), dtype=np.float32),
                    confidence=np.zeros((0,), dtype=np.float32),
                    class_id=np.zeros((0,), dtype=np.int32),
                )
            if len(chosen) == 0:
                # treat very low confidence as a miss
                self.state.missing += 1
                if self.state.center is None or self.state.missing > self.max_missing:
                    self.state = BallTrackState()
                    return chosen, False
                if self.state.missing <= self.max_interp_frames:
                    return self._make_detection(self.state.center, self.state.box_wh), True
                return chosen, False

            self.state.center = centers[idx].astype(np.float32)
            self.state.box_wh = np.maximum(wh[idx].astype(np.float32), 2.0)
            self.state.missing = 0
            return chosen, False

        if self.state.center is None:
            return ball_det, False

        self.state.missing += 1
        if self.state.missing > self.max_missing:
            self.state = BallTrackState()
            return ball_det, False

        if self.state.missing <= self.max_interp_frames:
            return self._make_detection(self.state.center, self.state.box_wh), True
        return ball_det, False
