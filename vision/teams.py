from __future__ import annotations

import supervision as sv
import numpy as np

try:
    from sports.common.team import TeamClassifier
except Exception:
    TeamClassifier = None


class TeamClassifierWrapper:
    """
    Notebook-style TeamClassifier.
    Note: This may download weights -> HF_TOKEN can be required depending on your environment.
    """
    def __init__(self, device: str = "cpu"):
        if TeamClassifier is None:
            raise RuntimeError("sports TeamClassifier not available. Install roboflow/sports and transformers deps.")
        self.device = device
        self.clf = TeamClassifier(device=device)
        self.is_fit = False

    def fit_from_video_frames(
        self,
        frame_iter,
        player_model,
        det_conf: float,
        player_class_id: int,
        stride_note: str = "Use a generator with stride outside",
        max_frames: int | None = None,
    ) -> None:
        crops: list[np.ndarray] = []
        count = 0
        for frame in frame_iter:
            det = sv.Detections.from_inference(player_model.infer(frame, confidence=det_conf)[0])
            players = det[det.class_id == player_class_id]
            for xyxy in players.xyxy:
                crops.append(sv.crop_image(frame, xyxy))
            count += 1
            if max_frames and count >= max_frames:
                break

        if len(crops) == 0:
            raise RuntimeError("No player crops collected for team classifier fitting.")
        self.clf.fit(crops)
        self.is_fit = True

    def predict_team_ids(self, frame: np.ndarray, player_detections: sv.Detections) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError("TeamClassifier not fit yet. Call fit_from_video_frames first.")
        crops = [sv.crop_image(frame, xyxy) for xyxy in player_detections.xyxy]
        if len(crops) == 0:
            return np.array([], dtype=np.int32)
        team_ids = self.clf.predict(crops)
        return np.asarray(team_ids, dtype=np.int32)
