from __future__ import annotations

from collections import defaultdict
import numpy as np
import supervision as sv

from vision.reid import ReIDExtractor


class EnhancedTracker:
    """ByteTrack + ReID recovery + class stabilization.

    ByteTrack handles primary IoU/motion matching. When a track is lost,
    its appearance embedding is stored. If a new ByteTrack ID appears that
    visually matches a lost track, the old stable ID is recovered instead
    of minting a new one — eliminating the most common ID switches.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 60,
        minimum_matching_threshold: float = 0.80,
        frame_rate: int = 30,
        class_lock_threshold: int = 5,
        reid_extractor: ReIDExtractor | None = None,
        reid_match_threshold: float = 0.40,
        reid_gallery_alpha: float = 0.90,
        reid_lost_buffer: int = 90,
    ):
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        self.byte_tracker.reset()

        self.reid = reid_extractor
        self.reid_match_threshold = reid_match_threshold
        self.reid_gallery_alpha = reid_gallery_alpha
        self.reid_lost_buffer = reid_lost_buffer

        self._id_map: dict[int, int] = {}
        self._next_stable_id = 1

        self._gallery: dict[int, np.ndarray] = {}
        self._lost: dict[int, dict] = {}

        self._class_lock_threshold = class_lock_threshold
        self._class_votes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

        self.last_stats: dict = {}

    def reset(self) -> None:
        self.byte_tracker.reset()
        self._id_map.clear()
        self._next_stable_id = 1
        self._gallery.clear()
        self._lost.clear()
        self._class_votes.clear()

    # ------------------------------------------------------------------
    # ReID helpers
    # ------------------------------------------------------------------

    def _match_lost(self, embedding: np.ndarray, class_id: int) -> int | None:
        """Find best-matching lost track via cosine similarity, respecting class."""
        if not self._lost:
            return None

        best_sid: int | None = None
        best_sim = self.reid_match_threshold

        for sid, info in self._lost.items():
            if info["class_id"] is not None and info["class_id"] != class_id:
                continue
            sim = float(np.dot(embedding, info["embedding"]))
            if sim > best_sim:
                best_sim = sim
                best_sid = sid

        return best_sid

    def _age_lost_tracks(self) -> None:
        for sid in list(self._lost.keys()):
            self._lost[sid]["age"] += 1
            if self._lost[sid]["age"] > self.reid_lost_buffer:
                del self._lost[sid]

    def _move_to_lost(self, bt_id: int) -> None:
        stable_id = self._id_map.pop(bt_id, None)
        if stable_id is None:
            return
        emb = self._gallery.pop(stable_id, None)
        if emb is not None:
            self._lost[stable_id] = {
                "embedding": emb,
                "age": 0,
                "class_id": self._get_majority_class(stable_id),
            }

    def _update_gallery(self, stable_id: int, embedding: np.ndarray) -> None:
        if stable_id in self._gallery:
            updated = (
                self.reid_gallery_alpha * self._gallery[stable_id]
                + (1.0 - self.reid_gallery_alpha) * embedding
            )
            self._gallery[stable_id] = updated / np.linalg.norm(updated)
        else:
            self._gallery[stable_id] = embedding

    # ------------------------------------------------------------------
    # Class stabilization
    # ------------------------------------------------------------------

    def _get_majority_class(self, stable_id: int) -> int | None:
        votes = self._class_votes.get(stable_id)
        if not votes:
            return None
        return max(votes, key=votes.get)

    def _stabilize_classes(self, detections: sv.Detections) -> sv.Detections:
        if detections.tracker_id is None or len(detections) == 0:
            return detections

        stable_classes = detections.class_id.copy()
        for i, tid in enumerate(detections.tracker_id):
            tid = int(tid)
            det_cls = int(detections.class_id[i])
            self._class_votes[tid][det_cls] += 1
            total = sum(self._class_votes[tid].values())
            if total >= self._class_lock_threshold:
                stable_classes[i] = max(
                    self._class_votes[tid], key=self._class_votes[tid].get
                )

        detections.class_id = stable_classes
        return detections

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(
        self, detections: sv.Detections, frame: np.ndarray | None = None
    ) -> sv.Detections:
        n_input = len(detections)
        tracks = self.byte_tracker.update_with_detections(detections)

        reid_recoveries = 0

        if tracks.tracker_id is None or len(tracks) == 0:
            if self.reid is not None:
                for bt_id in list(self._id_map.keys()):
                    self._move_to_lost(bt_id)
                self._age_lost_tracks()

            self.last_stats = self._make_stats(n_input, 0, reid_recoveries)
            return tracks

        # --- ReID-enhanced ID management ---
        if self.reid is not None and frame is not None:
            embeddings = self.reid.extract(frame, tracks.xyxy)
            current_bt_ids = set(int(x) for x in tracks.tracker_id)

            for bt_id in list(self._id_map.keys()):
                if bt_id not in current_bt_ids:
                    self._move_to_lost(bt_id)

            stable_ids = np.zeros(len(tracks), dtype=np.int64)

            for i in range(len(tracks)):
                bt_id = int(tracks.tracker_id[i])

                if bt_id in self._id_map:
                    sid = self._id_map[bt_id]
                    stable_ids[i] = sid
                    self._update_gallery(sid, embeddings[i])
                else:
                    matched_sid = self._match_lost(
                        embeddings[i], int(tracks.class_id[i])
                    )
                    if matched_sid is not None:
                        stable_ids[i] = matched_sid
                        self._id_map[bt_id] = matched_sid
                        self._gallery[matched_sid] = embeddings[i]
                        del self._lost[matched_sid]
                        reid_recoveries += 1
                    else:
                        sid = self._next_stable_id
                        self._next_stable_id += 1
                        stable_ids[i] = sid
                        self._id_map[bt_id] = sid
                        self._gallery[sid] = embeddings[i]

            tracks.tracker_id = stable_ids
            self._age_lost_tracks()

        tracks = self._stabilize_classes(tracks)

        n_matched = len(tracks)
        self.last_stats = self._make_stats(n_input, n_matched, reid_recoveries)
        return tracks

    def _make_stats(self, n_input: int, n_matched: int, reid_recoveries: int) -> dict:
        return {
            "input_detections": n_input,
            "matched_tracks": n_matched,
            "unmatched_detections": max(0, n_input - n_matched),
            "active_track_ids": [],
            "reid_recoveries": reid_recoveries,
            "lost_tracks": len(self._lost),
        }
