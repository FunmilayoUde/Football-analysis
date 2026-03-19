from __future__ import annotations

import numpy as np
import cv2
import supervision as sv

from core.types import HomographyResult
from core.utils import normalize_h
from sports.configs.soccer import SoccerPitchConfiguration

GATE_CONTROL_POINTS = np.array([
    [0.0, 0.0],
    [105.0, 0.0],
    [105.0, 68.0],
    [0.0, 68.0],
], dtype=np.float32)

SMOOTH_REFERENCE_POINTS = np.array([
    [0.0, 0.0], [52.5, 0.0], [105.0, 0.0],
    [0.0, 34.0], [52.5, 34.0], [105.0, 34.0],
    [0.0, 68.0], [52.5, 68.0], [105.0, 68.0],
], dtype=np.float32)


class HomographyEstimator:
    def __init__(
        self,
        config: SoccerPitchConfiguration,
        kp_conf: float,
        ema_alpha: float,
        ransac_reproj_thresh: float,
        min_kp: int,
        min_inlier_ratio: float,
        max_reproj_err: float,
        jump_gate_px: float = 50.0,
        fallback_expiry: int = 30,
    ):
        self.config = config
        self.kp_conf = kp_conf
        self.ema_alpha = ema_alpha
        self.ransac_reproj_thresh = ransac_reproj_thresh
        self.min_kp = min_kp
        self.min_inlier_ratio = min_inlier_ratio
        self.max_reproj_err = max_reproj_err
        self.jump_gate_px = jump_gate_px
        self.fallback_expiry = fallback_expiry

        self._H_prev: np.ndarray | None = None
        self._fallback_count: int = 0
        self._ref_image_pts: np.ndarray | None = None

    def reset(self) -> None:
        self._H_prev = None
        self._fallback_count = 0
        self._ref_image_pts = None

    def _jump_gate(self, H_new: np.ndarray) -> bool:
        """Reject H_new if projected control points shift too far from H_prev."""
        if self._H_prev is None:
            return True

        pts = GATE_CONTROL_POINTS.reshape(-1, 1, 2)
        try:
            H_prev_inv = np.linalg.inv(self._H_prev)
            H_new_inv = np.linalg.inv(H_new)
        except np.linalg.LinAlgError:
            return False

        proj_prev = cv2.perspectiveTransform(pts, H_prev_inv).reshape(-1, 2)
        proj_new = cv2.perspectiveTransform(pts, H_new_inv).reshape(-1, 2)

        max_displacement = np.linalg.norm(proj_new - proj_prev, axis=1).max()
        return float(max_displacement) <= self.jump_gate_px

    def _landmark_smooth(self, H_new: np.ndarray) -> np.ndarray:
        """Smooth in image-space via reference landmarks, then refit H.

        Unlike raw matrix EMA, this always produces a geometrically valid
        homography because the output is fitted from point correspondences.
        """
        ref_pitch = SMOOTH_REFERENCE_POINTS.reshape(-1, 1, 2)
        try:
            H_new_inv = np.linalg.inv(H_new)
        except np.linalg.LinAlgError:
            return H_new

        ref_image_new = cv2.perspectiveTransform(ref_pitch, H_new_inv).reshape(-1, 2)

        if self._ref_image_pts is None:
            self._ref_image_pts = ref_image_new
        else:
            self._ref_image_pts = (
                self.ema_alpha * self._ref_image_pts
                + (1.0 - self.ema_alpha) * ref_image_new
            )

        H_smooth, _ = cv2.findHomography(
            self._ref_image_pts.astype(np.float32),
            SMOOTH_REFERENCE_POINTS,
            method=0,
        )

        if H_smooth is None:
            return H_new

        return normalize_h(H_smooth)

    def _make_fallback_result(
        self, n: int, inlier_ratio: float = 0.0, reproj_err: float = 1e9
    ) -> HomographyResult:
        self._fallback_count += 1
        degraded = self._fallback_count >= self.fallback_expiry
        H = None if degraded else self._H_prev
        return HomographyResult(
            H=H,
            ok=False,
            n_points=n,
            inlier_ratio=inlier_ratio,
            reproj_err=reproj_err,
            degraded=degraded,
            fallback_age=self._fallback_count,
        )

    def estimate(self, keypoints: sv.KeyPoints) -> HomographyResult:
        conf = keypoints.confidence[0]
        keep = conf > self.kp_conf

        frame_pts = keypoints.xy[0][keep].astype(np.float32)
        pitch_pts = np.array(self.config.vertices, dtype=np.float32)[keep]

        n = frame_pts.shape[0]
        if n < 4 or n < self.min_kp:
            return self._make_fallback_result(n)

        H, inliers = cv2.findHomography(
            frame_pts,
            pitch_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(self.ransac_reproj_thresh),
        )

        if H is None or inliers is None:
            return self._make_fallback_result(n)

        inliers = inliers.reshape(-1).astype(bool)
        inlier_ratio = float(inliers.mean()) if len(inliers) else 0.0
        if inlier_ratio < self.min_inlier_ratio:
            return self._make_fallback_result(n, inlier_ratio=inlier_ratio)

        frame_in = frame_pts[inliers]
        pitch_in = pitch_pts[inliers]
        proj = cv2.perspectiveTransform(frame_in.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = np.linalg.norm(proj - pitch_in, axis=1)
        reproj_err = float(err.mean()) if err.size else 1e9
        if reproj_err > self.max_reproj_err:
            return self._make_fallback_result(
                n, inlier_ratio=inlier_ratio, reproj_err=reproj_err
            )

        H = normalize_h(H)

        if not self._jump_gate(H):
            return self._make_fallback_result(
                n, inlier_ratio=inlier_ratio, reproj_err=reproj_err
            )

        H_smooth = self._landmark_smooth(H)

        self._H_prev = H_smooth
        self._fallback_count = 0
        return HomographyResult(
            H=H_smooth,
            ok=True,
            n_points=n,
            inlier_ratio=inlier_ratio,
            reproj_err=reproj_err,
            degraded=False,
            fallback_age=0,
        )

    @staticmethod
    def transform_points(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        if H is None or points_xy is None or points_xy.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        pts = points_xy.astype(np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        return out.astype(np.float32)
