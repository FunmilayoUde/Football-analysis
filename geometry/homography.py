from __future__ import annotations

import numpy as np
import cv2
import supervision as sv

from core.types import HomographyResult
from core.utils import normalize_h, ema_matrix
from sports.configs.soccer import SoccerPitchConfiguration


class HomographyEstimator:
    """
    Estimates image->pitch homography using Roboflow field keypoints.
    - Filters keypoints by confidence
    - Uses RANSAC homography
    - Validates via inlier ratio and mean reprojection error
    - EMA smooths H
    - Falls back to last good H (self._H_prev) when current estimate fails
    """
    def __init__(
        self,
        config: SoccerPitchConfiguration,
        kp_conf: float,
        ema_alpha: float,
        ransac_reproj_thresh: float,
        min_kp: int,
        min_inlier_ratio: float,
        max_reproj_err: float,
    ):
        self.config = config
        self.kp_conf = kp_conf
        self.ema_alpha = ema_alpha
        self.ransac_reproj_thresh = ransac_reproj_thresh
        self.min_kp = min_kp
        self.min_inlier_ratio = min_inlier_ratio
        self.max_reproj_err = max_reproj_err
        self._H_prev: np.ndarray | None = None
        self._warned_kp_mismatch = False
        self._fail_counts: dict[str, int] = {}
        self._diag_max_per_reason = 5

        # The keypoint-detection model emits class_ids in the order of config.labels,
        # NOT in the order of config.vertices. Specifically labels "14" and "19" are
        # at the END of the labels list (indices 30, 31), so a naive vertices[class_id]
        # lookup is wrong for any class_id >= 13. Build a class_id -> vertex_index map
        # using the label numerical name (vertex "15" -> index 14, etc.).
        try:
            self._class_id_to_vertex_idx = np.array(
                [int(label) - 1 for label in config.labels],
                dtype=np.int32,
            )
        except Exception:
            self._class_id_to_vertex_idx = None

    def _diag(self, reason: str, **kwargs) -> None:
        n = self._fail_counts.get(reason, 0) + 1
        self._fail_counts[reason] = n
        if n <= self._diag_max_per_reason:
            extras = " ".join(f"{k}={v}" for k, v in kwargs.items())
            print(f"[homography] reject reason={reason} ({extras})")

    def report_failure_summary(self) -> None:
        if not self._fail_counts:
            return
        print("[homography] failure summary:", dict(self._fail_counts))

    def reset(self) -> None:
        self._H_prev = None

    def estimate(self, keypoints: sv.KeyPoints) -> HomographyResult:
        if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
            return HomographyResult(H=self._H_prev, ok=False, n_points=0, inlier_ratio=0.0, reproj_err=1e9)

        frame_all = np.asarray(keypoints.xy[0], dtype=np.float32)
        if frame_all.ndim != 2 or frame_all.shape[1] != 2 or frame_all.shape[0] == 0:
            return HomographyResult(H=self._H_prev, ok=False, n_points=0, inlier_ratio=0.0, reproj_err=1e9)

        if keypoints.confidence is not None and len(keypoints.confidence) > 0:
            conf_all = np.asarray(keypoints.confidence[0], dtype=np.float32)
        else:
            conf_all = np.ones((frame_all.shape[0],), dtype=np.float32)

        vertices = np.asarray(self.config.vertices, dtype=np.float32)

        # Prefer explicit keypoint ids from detector output (if available), then fallback to aligned prefix.
        pitch_all = None
        kp_idx = None
        if hasattr(keypoints, "class_id") and keypoints.class_id is not None:
            kp_idx = np.asarray(keypoints.class_id)
            if kp_idx.ndim > 1:
                kp_idx = kp_idx[0]
            kp_idx = kp_idx.astype(np.int32)

        if kp_idx is not None and kp_idx.shape[0] == frame_all.shape[0]:
            valid_idx = (kp_idx >= 0) & (kp_idx < len(vertices))
            if np.any(valid_idx):
                frame_all = frame_all[valid_idx]
                conf_all = conf_all[valid_idx]
                pitch_all = vertices[kp_idx[valid_idx]]

        if pitch_all is None:
            n_common = min(frame_all.shape[0], conf_all.shape[0], len(vertices))
            if n_common == 0:
                return HomographyResult(H=self._H_prev, ok=False, n_points=0, inlier_ratio=0.0, reproj_err=1e9)

            if not self._warned_kp_mismatch and frame_all.shape[0] != len(vertices):
                print(
                    f"[WARN] Keypoint/vertex count mismatch (kp={frame_all.shape[0]}, pitch={len(vertices)}). "
                    f"Using first {n_common} pairs."
                )
                self._warned_kp_mismatch = True

            frame_all = frame_all[:n_common]
            conf_all = conf_all[:n_common]
            pitch_all = vertices[:n_common]

        keep = conf_all > self.kp_conf
        frame_pts = frame_all[keep]
        pitch_pts = pitch_all[keep]

        n = frame_pts.shape[0]
        if n < 4 or n < self.min_kp:
            self._diag("too_few_kp", n=n, min_kp=self.min_kp)
            return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=0.0, reproj_err=1e9)

        H, inliers = cv2.findHomography(
            frame_pts,
            pitch_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(self.ransac_reproj_thresh),
        )

        if H is None or inliers is None:
            self._diag("ransac_returned_none", n=n)
            return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=0.0, reproj_err=1e9)

        inliers = inliers.reshape(-1).astype(bool)
        inlier_ratio = float(inliers.mean()) if len(inliers) else 0.0
        if inlier_ratio < self.min_inlier_ratio:
            self._diag("inlier_ratio_low", n=n, ratio=f"{inlier_ratio:.2f}", min=self.min_inlier_ratio)
            return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=inlier_ratio, reproj_err=1e9)

        # Reject ill-conditioned H — a degenerate matrix projects most of the image to infinity
        # even when the RANSAC inlier set has zero reprojection error.
        cond = float(np.linalg.cond(H))
        if not np.isfinite(cond) or cond > 1e9:
            self._diag("ill_conditioned", n=n, cond=f"{cond:.2e}")
            return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=inlier_ratio, reproj_err=1e9)

        # Full-keypoint reprojection check: RANSAC inlier error is circular (RANSAC minimized it),
        # so check ALL confident keypoints. A minimal 4-point H that diverges everywhere else
        # will fail this test. Use a generous soft threshold to stay robust to genuine outliers.
        all_proj = cv2.perspectiveTransform(frame_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        if not np.all(np.isfinite(all_proj)):
            self._diag("nonfinite_proj", n=n)
            return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=inlier_ratio, reproj_err=1e9)
        all_err = np.linalg.norm(all_proj - pitch_pts, axis=1)
        soft_thresh = self.ransac_reproj_thresh * 6.0
        soft_inlier_ratio = float(np.mean(all_err <= soft_thresh))
        if soft_inlier_ratio < (self.min_inlier_ratio - 0.10):
            self._diag(
                "soft_inlier_low",
                n=n,
                soft=f"{soft_inlier_ratio:.2f}",
                med_err=f"{float(np.median(all_err)):.1f}",
            )
            return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=inlier_ratio, reproj_err=float(np.median(all_err)))

        # Reprojection error on RANSAC inliers only (for logging/threshold).
        frame_in = frame_pts[inliers]
        pitch_in = pitch_pts[inliers]
        proj = cv2.perspectiveTransform(frame_in.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = np.linalg.norm(proj - pitch_in, axis=1)
        reproj_err = float(err.mean()) if err.size else 1e9
        if reproj_err > self.max_reproj_err:
            self._diag("reproj_err_high", n=n, err=f"{reproj_err:.1f}", max=self.max_reproj_err)
            return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=inlier_ratio, reproj_err=reproj_err)

        H = normalize_h(H)

        # EMA smoothing (note: alpha here means "weight of previous")
        if self._H_prev is None:
            H_smooth = H
        else:
            H_smooth = ema_matrix(self._H_prev, H, alpha=self.ema_alpha)
            H_smooth = normalize_h(H_smooth)

        self._H_prev = H_smooth
        return HomographyResult(H=H_smooth, ok=True, n_points=n, inlier_ratio=inlier_ratio, reproj_err=reproj_err)

    @staticmethod
    def transform_points(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        """
        Apply homography H to Nx2 points (image coords) -> Nx2 points (pitch coords).
        """
        if H is None or points_xy is None or points_xy.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        pts = points_xy.astype(np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        return out.astype(np.float32)
