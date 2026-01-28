# from __future__ import annotations

# import numpy as np
# import cv2
# import supervision as sv
# from sports.common.view import ViewTransformer
# from sports.configs.soccer import SoccerPitchConfiguration

# from core.types import HomographyResult
# from core.utils import normalize_h, ema_matrix


# class HomographyEstimator:
#     def __init__(
#         self,
#         config: SoccerPitchConfiguration,
#         kp_conf: float,
#         ema_alpha: float,
#         ransac_reproj_thresh: float,
#         min_kp: int,
#         min_inlier_ratio: float,
#         max_reproj_err: float,
#     ):
#         self.config = config
#         self.kp_conf = kp_conf
#         self.ema_alpha = ema_alpha
#         self.ransac_reproj_thresh = ransac_reproj_thresh
#         self.min_kp = min_kp
#         self.min_inlier_ratio = min_inlier_ratio
#         self.max_reproj_err = max_reproj_err

#         self._H_prev: np.ndarray | None = None

#     def reset(self) -> None:
#         self._H_prev = None

#     def estimate(self, keypoints: sv.KeyPoints) -> HomographyResult:
#         """
#         keypoints: sv.KeyPoints from FIELD_DETECTION_MODEL.infer
#         Uses notebook logic for selecting confident points, but computes H with RANSAC
#         and then EMA smooths it.
#         """
#         # keypoints.xy shape: (1, K, 2), keypoints.confidence shape: (1, K)
#         conf = keypoints.confidence[0]
#         keep = conf > self.kp_conf

#         frame_pts = keypoints.xy[0][keep].astype(np.float32)             # Nx2
#         pitch_pts = np.array(self.config.vertices, dtype=np.float32)[keep]  # Nx2

#         n = frame_pts.shape[0]
#         if n < 4 or n < self.min_kp:
#             return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=0.0, reproj_err=1e9)

#         H, inliers = cv2.findHomography(
#             frame_pts,
#             pitch_pts,
#             method=cv2.RANSAC,
#             ransacReprojThreshold=float(self.ransac_reproj_thresh),
#         )

#         if H is None or inliers is None:
#             return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=0.0, reproj_err=1e9)

#         inliers = inliers.reshape(-1).astype(bool)
#         inlier_ratio = float(inliers.mean()) if len(inliers) else 0.0
#         if inlier_ratio < self.min_inlier_ratio:
#             return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=inlier_ratio, reproj_err=1e9)

#         # reprojection error on inliers
#         frame_in = frame_pts[inliers]
#         pitch_in = pitch_pts[inliers]
#         proj = cv2.perspectiveTransform(frame_in.reshape(-1, 1, 2), H).reshape(-1, 2)
#         err = np.linalg.norm(proj - pitch_in, axis=1)
#         reproj_err = float(err.mean()) if err.size else 1e9
#         if reproj_err > self.max_reproj_err:
#             return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=inlier_ratio, reproj_err=reproj_err)

#         H = normalize_h(H)

#         # EMA smooth
#         if self._H_prev is None:
#             H_smooth = H
#         else:
#             H_smooth = ema_matrix(self._H_prev, H, alpha=self.ema_alpha)
#             H_smooth = normalize_h(H_smooth)

#         self._H_prev = H_smooth
#         return HomographyResult(H=H_smooth, ok=True, n_points=n, inlier_ratio=inlier_ratio, reproj_err=reproj_err)

#     def make_transformer(self, H: np.ndarray) -> ViewTransformer:
#         # ViewTransformer accepts points, but we can directly override its matrix.
#         vt = ViewTransformer(source=np.zeros((4, 2), dtype=np.float32), target=np.zeros((4, 2), dtype=np.float32))
#         vt.m = H
#         return vt


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

    def reset(self) -> None:
        self._H_prev = None

    def estimate(self, keypoints: sv.KeyPoints) -> HomographyResult:
        conf = keypoints.confidence[0]
        keep = conf > self.kp_conf

        frame_pts = keypoints.xy[0][keep].astype(np.float32)
        pitch_pts = np.array(self.config.vertices, dtype=np.float32)[keep]

        n = frame_pts.shape[0]
        if n < 4 or n < self.min_kp:
            return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=0.0, reproj_err=1e9)

        H, inliers = cv2.findHomography(
            frame_pts,
            pitch_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(self.ransac_reproj_thresh),
        )

        if H is None or inliers is None:
            return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=0.0, reproj_err=1e9)

        inliers = inliers.reshape(-1).astype(bool)
        inlier_ratio = float(inliers.mean()) if len(inliers) else 0.0
        if inlier_ratio < self.min_inlier_ratio:
            return HomographyResult(H=self._H_prev, ok=False, n_points=n, inlier_ratio=inlier_ratio, reproj_err=1e9)

        # mean reprojection error (in pitch coord space)
        frame_in = frame_pts[inliers]
        pitch_in = pitch_pts[inliers]
        proj = cv2.perspectiveTransform(frame_in.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = np.linalg.norm(proj - pitch_in, axis=1)
        reproj_err = float(err.mean()) if err.size else 1e9
        if reproj_err > self.max_reproj_err:
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

