from __future__ import annotations

import numpy as np
import supervision as sv


# Notebook class IDs (IMPORTANT)
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3


def tiny_box_filter(
    detections: sv.Detections,
    frame_w: int,
    frame_h: int,
    min_area_ratio_people: float,
    min_area_ratio_ball: float,
) -> sv.Detections:
    if len(detections) == 0:
        return detections

    areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (detections.xyxy[:, 3] - detections.xyxy[:, 1])
    frame_area = float(frame_w * frame_h)
    area_ratio = areas / max(frame_area, 1.0)

    thr = np.where(detections.class_id == BALL_ID, min_area_ratio_ball, min_area_ratio_people)
    keep = area_ratio >= thr
    return detections[keep]


def infer_players_and_ball(player_model, frame: np.ndarray, conf: float) -> sv.Detections:
    result = player_model.infer(frame, confidence=conf)[0]
    det = sv.Detections.from_inference(result)
    return det


def infer_field_keypoints(field_model, frame: np.ndarray, conf: float) -> sv.KeyPoints:
    result = field_model.infer(frame, confidence=conf)[0]
    kp = sv.KeyPoints.from_inference(result)
    return kp
