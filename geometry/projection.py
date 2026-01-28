# from __future__ import annotations

# import numpy as np
# import supervision as sv
# from sports.common.view import ViewTransformer


# def project_anchors_to_pitch(
#     transformer: ViewTransformer,
#     detections: sv.Detections,
#     anchor: sv.Position = sv.Position.BOTTOM_CENTER
# ) -> np.ndarray:
#     """
#     Returns Nx2 pitch coordinates for each detection anchor point.
#     """
#     if len(detections) == 0:
#         return np.zeros((0, 2), dtype=np.float32)
#     pts = detections.get_anchors_coordinates(anchor)
#     pitch_pts = transformer.transform_points(points=pts)
#     return pitch_pts.astype(np.float32)  

from __future__ import annotations

import numpy as np
import supervision as sv
from geometry.homography import HomographyEstimator


def project_anchors_to_pitch(H: np.ndarray, detections: sv.Detections, anchor: sv.Position) -> np.ndarray:
    """
    Extract anchor points from detections and project them onto pitch coordinates using homography H.
    """
    if H is None or len(detections) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    pts = detections.get_anchors_coordinates(anchor)
    return HomographyEstimator.transform_points(H, pts)

