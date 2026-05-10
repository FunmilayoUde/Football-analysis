from __future__ import annotations

import numpy as np
import supervision as sv
import cv2


class ColorTeamClassifier:
    """
    Lightweight team classifier using torso HSV features + online 2-cluster centroids.
    Designed for low-quality video where heavy embedding models can be unstable/slow.
    """
    def __init__(
        self,
        init_samples: int = 40,
        lr: float = 0.05,
        min_margin: float = 0.12,
        track_feat_alpha: float = 0.8,
    ):
        self.init_samples = int(init_samples)
        self.lr = float(lr)
        self.min_margin = float(min_margin)
        self.track_feat_alpha = float(track_feat_alpha)
        self.centroids: np.ndarray | None = None  # (2, D)
        self._buffer: list[np.ndarray] = []
        self._track_feat: dict[int, np.ndarray] = {}

    @staticmethod
    def _torso_crop(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in xyxy.tolist()]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None

        bw = x2 - x1
        bh = y2 - y1
        if bw < 8 or bh < 12:
            return None

        tx1 = x1 + int(0.22 * bw)
        tx2 = x1 + int(0.78 * bw)
        ty1 = y1 + int(0.10 * bh)
        ty2 = y1 + int(0.58 * bh)
        if tx2 - tx1 < 6 or ty2 - ty1 < 8:
            return None
        return frame[ty1:ty2, tx1:tx2]

    @staticmethod
    def _feature_from_crop(crop: np.ndarray) -> np.ndarray | None:
        if crop is None or crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Focus on colored jersey signal, reduce influence of dark shadows.
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask = (sat > 35) & (val > 30)
        if np.count_nonzero(mask) < 20:
            mask = np.ones_like(sat, dtype=bool)

        h_ch = hsv[:, :, 0][mask]
        s_ch = hsv[:, :, 1][mask]
        if h_ch.size == 0:
            return None

        hist2d, _, _ = np.histogram2d(
            h_ch.astype(np.float32),
            s_ch.astype(np.float32),
            bins=(12, 8),
            range=((0, 180), (0, 256)),
        )
        hist = hist2d.flatten().astype(np.float32)
        hist /= max(float(hist.sum()), 1e-6)

        mean_h = np.array([float(np.mean(h_ch)) / 180.0], dtype=np.float32)
        mean_s = np.array([float(np.mean(s_ch)) / 255.0], dtype=np.float32)
        feat = np.concatenate([hist, mean_h, mean_s], axis=0).astype(np.float32)
        feat /= max(float(np.linalg.norm(feat)), 1e-6)
        return feat

    @staticmethod
    def _init_centroids(features: np.ndarray) -> np.ndarray:
        n = features.shape[0]
        if n < 2:
            return np.stack([features[0], features[0]], axis=0)

        # Farthest-pair seeding.
        best_i, best_j, best_d = 0, 1, -1.0
        for i in range(n):
            d = np.linalg.norm(features[i + 1:] - features[i], axis=1) if i + 1 < n else np.array([])
            if d.size == 0:
                continue
            j_rel = int(np.argmax(d))
            if float(d[j_rel]) > best_d:
                best_d = float(d[j_rel])
                best_i = i
                best_j = i + 1 + j_rel

        c = np.stack([features[best_i], features[best_j]], axis=0).astype(np.float32)
        for _ in range(8):
            d0 = np.linalg.norm(features - c[0], axis=1)
            d1 = np.linalg.norm(features - c[1], axis=1)
            a = d1 < d0
            if np.any(~a):
                c[0] = features[~a].mean(axis=0)
            if np.any(a):
                c[1] = features[a].mean(axis=0)
            c[0] /= max(float(np.linalg.norm(c[0])), 1e-6)
            c[1] /= max(float(np.linalg.norm(c[1])), 1e-6)
        return c

    def _maybe_fit(self) -> None:
        if self.centroids is not None:
            return
        if len(self._buffer) < self.init_samples:
            return
        feats = np.asarray(self._buffer, dtype=np.float32)
        self.centroids = self._init_centroids(feats)

    def _assign(self, feat: np.ndarray) -> tuple[int, float]:
        d0 = float(np.linalg.norm(feat - self.centroids[0]))
        d1 = float(np.linalg.norm(feat - self.centroids[1]))
        margin = abs(d0 - d1)
        team = 0 if d0 < d1 else 1
        return team, margin

    def update_players(
        self,
        frame: np.ndarray,
        player_detections: sv.Detections,
        stable_player_ids: np.ndarray,
    ) -> dict[int, int]:
        out: dict[int, int] = {}
        if len(player_detections) == 0 or len(stable_player_ids) == 0:
            return out

        for i, tid in enumerate(stable_player_ids):
            if i >= len(player_detections):
                break
            xyxy = player_detections.xyxy[i]
            crop = self._torso_crop(frame, xyxy)
            feat = self._feature_from_crop(crop)
            if feat is None:
                continue

            tid = int(tid)
            prev = self._track_feat.get(tid)
            if prev is not None and prev.shape == feat.shape:
                feat = self.track_feat_alpha * prev + (1.0 - self.track_feat_alpha) * feat
                feat /= max(float(np.linalg.norm(feat)), 1e-6)
            self._track_feat[tid] = feat

            if self.centroids is None:
                self._buffer.append(feat)
                continue

            team, margin = self._assign(feat)
            if margin < self.min_margin:
                continue

            out[tid] = int(team)
            # Avoid centroid drift from low-quality assignments; update only on high-confidence margin.
            if margin >= (self.min_margin * 2.5):
                self.centroids[team] = (1.0 - self.lr) * self.centroids[team] + self.lr * feat
                self.centroids[team] /= max(float(np.linalg.norm(self.centroids[team])), 1e-6)

        self._maybe_fit()
        return out

    @property
    def is_ready(self) -> bool:
        return self.centroids is not None
