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

    @staticmethod
    def _torso_crop(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in xyxy.tolist()]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None

        bw = x2 - x1
        bh = y2 - y1
        if bw < 6 or bh < 10:
            return None

        # Center-x and upper-body emphasis.
        tx1 = x1 + int(0.20 * bw)
        tx2 = x1 + int(0.80 * bw)
        ty1 = y1 + int(0.10 * bh)
        ty2 = y1 + int(0.58 * bh)

        tx1 = max(0, min(w - 1, tx1))
        tx2 = max(0, min(w, tx2))
        ty1 = max(0, min(h - 1, ty1))
        ty2 = max(0, min(h, ty2))

        if tx2 - tx1 < 4 or ty2 - ty1 < 6:
            return None
        return frame[ty1:ty2, tx1:tx2]

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
                c = self._torso_crop(frame, xyxy)
                if c is not None:
                    crops.append(c)
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
        crops = []
        valid_idx = []
        for i, xyxy in enumerate(player_detections.xyxy):
            c = self._torso_crop(frame, xyxy)
            if c is not None:
                crops.append(c)
                valid_idx.append(i)

        if len(crops) == 0:
            return np.array([], dtype=np.int32)
        pred = np.asarray(self.clf.predict(crops), dtype=np.int32)
        out = np.full((len(player_detections),), -1, dtype=np.int32)
        for i, p in zip(valid_idx, pred):
            out[i] = int(p)
        return out
