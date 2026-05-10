from __future__ import annotations
import cv2
import numpy as np
import supervision as sv

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


def infer_players_and_ball_upscaled(
    player_model,
    frame: np.ndarray,
    conf: float,
    upscale: float = 1.0,
) -> sv.Detections:
    upscale = float(max(1.0, upscale))
    if upscale <= 1.01:
        return infer_players_and_ball(player_model, frame, conf=conf)

    h, w = frame.shape[:2]
    up = cv2.resize(frame, (int(w * upscale), int(h * upscale)), interpolation=cv2.INTER_LINEAR)
    det = infer_players_and_ball(player_model, up, conf=conf)
    if len(det) > 0:
        det.xyxy = det.xyxy / upscale
    return det


_KP_CORRESPONDENCE_LOGGED = False


def infer_field_keypoints(field_model, frame: np.ndarray, conf: float) -> sv.KeyPoints:
    global _KP_CORRESPONDENCE_LOGGED
    result = field_model.infer(frame, confidence=conf)[0]
    kp = sv.KeyPoints.from_inference(result)
    # sv.KeyPoints.from_inference only stores the global detection class_id, not
    # the per-keypoint class_ids that encode which pitch vertex each point corresponds to.
    # Inject the per-keypoint ids so the homography estimator uses the correct mapping.
    try:
        pred = result.predictions[0]
        per_kp_ids = np.array(
            [kpt.class_id for kpt in pred.keypoints],
            dtype=np.int32,
        )
        kp.class_id = per_kp_ids.reshape(1, -1)
        if not _KP_CORRESPONDENCE_LOGGED:
            print(
                f"[detect] keypoint correspondence injected: "
                f"n={per_kp_ids.shape[0]} ids[0:8]={per_kp_ids[:8].tolist()}"
            )
            try:
                pairs = [
                    (int(kpt.class_id), getattr(kpt, "class_name", "?"))
                    for kpt in pred.keypoints[:12]
                ]
                print(f"[detect] kp class_id <-> class_name pairs: {pairs}")
            except Exception:
                pass
            _KP_CORRESPONDENCE_LOGGED = True
    except Exception as exc:
        if not _KP_CORRESPONDENCE_LOGGED:
            print(f"[detect] WARNING: keypoint correspondence injection FAILED: {exc!r}")
            _KP_CORRESPONDENCE_LOGGED = True
    return kp


def class_conf_filter(
    detections: sv.Detections,
    player_conf: float,
    referee_conf: float,
    goalkeeper_conf: float,
    ball_conf: float,
) -> sv.Detections:
    if len(detections) == 0 or detections.confidence is None or detections.class_id is None:
        return detections

    cls = detections.class_id.astype(np.int32)
    conf = detections.confidence.astype(np.float32)
    thr = np.full_like(conf, fill_value=player_conf, dtype=np.float32)
    thr = np.where(cls == PLAYER_ID, player_conf, thr)
    thr = np.where(cls == REFEREE_ID, referee_conf, thr)
    thr = np.where(cls == GOALKEEPER_ID, goalkeeper_conf, thr)
    thr = np.where(cls == BALL_ID, ball_conf, thr)
    keep = conf >= thr
    return detections[keep]
