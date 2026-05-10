from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import supervision as sv
import cv2
try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - runtime fallback
    linear_sum_assignment = None


@dataclass
class StableTrack:
    stable_id: int
    class_id: int
    center: np.ndarray
    last_seen: int
    velocity: np.ndarray
    raw_id: int | None = None
    raw_key: tuple[int, int] | None = None
    appearance: np.ndarray | None = None
    appearance_gallery: list[np.ndarray] | None = None
    lock_until: int = -1
    team_id: int = -1
    team_votes: int = 0


class IDStabilizer:
    """
    Remaps raw tracker IDs to more stable IDs by reconnecting short-lived ID switches
    using both position and torso appearance.
    """
    def __init__(
        self,
        max_relink_frames: int = 30,
        memory_frames: int = 45,
        max_relink_px: float = 80.0,
        appearance_weight: float = 0.65,
        appearance_alpha: float = 0.8,
        max_relink_cost: float = 1.30,
        ambiguity_margin: float = 0.06,
        max_appearance_distance: float = 0.90,
        unknown_appearance_penalty: float = 0.70,
        unknown_appearance_max_gap: int = 12,
        unknown_appearance_max_space_frac: float = 0.90,
        gallery_size: int = 36,
        gallery_min_add_dist: float = 0.10,
        frame_continuity_iou: float = 0.62,
        frame_continuity_margin: float = 0.08,
        spatial_rescue_max_gap: int = 120,
        spatial_rescue_margin: float = 0.10,
        spatial_rescue_ratio: float = 0.78,
        max_player_ids: int = 22,
        player_class_id: int = 2,
        max_goalkeeper_ids: int = 2,
        goalkeeper_class_id: int = 1,
        max_referee_ids: int = 2,
        referee_class_id: int = 3,
        hard_reuse_when_capped: bool = True,
        lock_frames: int = 12,
        capped_reuse_min_gap: int = 8,
    ):
        self.max_relink_frames = int(max_relink_frames)
        self.memory_frames = int(memory_frames)
        self.max_relink_px = float(max_relink_px)
        self.appearance_weight = float(appearance_weight)
        self.appearance_alpha = float(appearance_alpha)
        self.max_relink_cost = float(max_relink_cost)
        self.ambiguity_margin = float(ambiguity_margin)
        self.max_appearance_distance = float(max_appearance_distance)
        self.unknown_appearance_penalty = float(unknown_appearance_penalty)
        self.unknown_appearance_max_gap = int(unknown_appearance_max_gap)
        self.unknown_appearance_max_space_frac = float(unknown_appearance_max_space_frac)
        self.gallery_size = max(4, int(gallery_size))
        self.gallery_min_add_dist = float(gallery_min_add_dist)
        self.frame_continuity_iou = float(frame_continuity_iou)
        self.frame_continuity_margin = float(frame_continuity_margin)
        self.spatial_rescue_max_gap = int(spatial_rescue_max_gap)
        self.spatial_rescue_margin = float(spatial_rescue_margin)
        self.spatial_rescue_ratio = float(spatial_rescue_ratio)
        self.max_player_ids = max(0, int(max_player_ids))
        self.player_class_id = int(player_class_id)
        self.max_goalkeeper_ids = max(0, int(max_goalkeeper_ids))
        self.goalkeeper_class_id = int(goalkeeper_class_id)
        self.max_referee_ids = max(0, int(max_referee_ids))
        self.referee_class_id = int(referee_class_id)
        self.hard_reuse_when_capped = bool(hard_reuse_when_capped)
        self.lock_frames = max(0, int(lock_frames))
        self.capped_reuse_min_gap = max(0, int(capped_reuse_min_gap))
        self._next_stable_id = 1
        self._raw_to_stable: dict[tuple[int, int], int] = {}
        self._stable_tracks: dict[int, StableTrack] = {}
        self._last_frame_idx: int | None = None
        self._last_boxes: np.ndarray | None = None
        self._last_class_ids: np.ndarray | None = None
        self._last_stable_ids: np.ndarray | None = None

    @staticmethod
    def _centers(xyxy: np.ndarray) -> np.ndarray:
        cx = 0.5 * (xyxy[:, 0] + xyxy[:, 2])
        # Bottom-center is more stable for players than bbox center during jumps/arm motion.
        cy = xyxy[:, 3]
        return np.stack([cx, cy], axis=1)

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
        if bw < 6 or bh < 9:
            return None
        tx1 = x1 + int(0.20 * bw)
        tx2 = x1 + int(0.80 * bw)
        ty1 = y1 + int(0.10 * bh)
        ty2 = y1 + int(0.58 * bh)
        if tx2 - tx1 < 4 or ty2 - ty1 < 6:
            return None
        return frame[ty1:ty2, tx1:tx2]

    @staticmethod
    def _appearance_feature(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
        crop = IDStabilizer._torso_crop(frame, xyxy)
        if crop is None:
            return None
        if crop.shape[0] < 12 or crop.shape[1] < 8:
            return None

        # Improve crop quality before histogram extraction.
        # Use cubic for upscaling tiny crops; area is still preferable for downscaling.
        target_w, target_h = 64, 96
        interp = cv2.INTER_CUBIC if (crop.shape[0] < target_h or crop.shape[1] < target_w) else cv2.INTER_AREA
        crop = cv2.resize(crop, (target_w, target_h), interpolation=interp)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l_ch)
        crop = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

        def block_feature(block: np.ndarray) -> np.ndarray | None:
            hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            mask = (sat > 30) & (val > 25)
            if np.count_nonzero(mask) < 20:
                mask = np.ones_like(sat, dtype=bool)
            h_ch = hsv[:, :, 0][mask]
            s_ch = hsv[:, :, 1][mask]
            if h_ch.size == 0:
                return None
            hist_hs, _, _ = np.histogram2d(
                h_ch.astype(np.float32),
                s_ch.astype(np.float32),
                bins=(12, 8),
                range=((0, 180), (0, 256)),
            )
            lab_eq = cv2.cvtColor(block, cv2.COLOR_BGR2LAB)
            a_vals = lab_eq[:, :, 1][mask]
            b_vals = lab_eq[:, :, 2][mask]
            hist_ab, _, _ = np.histogram2d(
                a_vals.astype(np.float32),
                b_vals.astype(np.float32),
                bins=(8, 8),
                range=((0, 256), (0, 256)),
            )
            return np.concatenate(
                [
                    hist_hs.flatten().astype(np.float32),
                    0.75 * hist_ab.flatten().astype(np.float32),
                ]
            )

        mid = max(1, crop.shape[0] // 2)
        top = crop[:mid, :]
        bottom = crop[mid:, :]
        feat_top = block_feature(top)
        feat_bottom = block_feature(bottom)
        if feat_top is None or feat_bottom is None:
            full = block_feature(crop)
            if full is None:
                return None
            feat = np.concatenate([full, full], axis=0)
        else:
            # Stripe encoding (top/bottom) improves separation for similar jersey colors.
            feat = np.concatenate([feat_top, feat_bottom], axis=0)
        feat /= max(float(feat.sum()), 1e-6)
        feat /= max(float(np.linalg.norm(feat)), 1e-6)
        return feat

    @staticmethod
    def _embedding_distance(a: np.ndarray, b: np.ndarray) -> float:
        # L2-normalized vectors: Euclidean in [0,2], map to [0,1].
        return float(np.linalg.norm(a - b)) * 0.5

    def _update_gallery(self, st: StableTrack, appearance: np.ndarray | None) -> None:
        if appearance is None:
            return
        if st.appearance_gallery is None:
            st.appearance_gallery = []
        if len(st.appearance_gallery) == 0:
            st.appearance_gallery.append(appearance.astype(np.float32))
            return
        d_min = min(self._embedding_distance(appearance, g) for g in st.appearance_gallery)
        if d_min >= self.gallery_min_add_dist or len(st.appearance_gallery) < 4:
            st.appearance_gallery.append(appearance.astype(np.float32))
        if len(st.appearance_gallery) > self.gallery_size:
            st.appearance_gallery = st.appearance_gallery[-self.gallery_size:]

    def _appearance_distance(self, st: StableTrack, appearance: np.ndarray | None) -> float | None:
        if appearance is None:
            return None
        candidates: list[float] = []
        if st.appearance is not None:
            candidates.append(self._embedding_distance(appearance, st.appearance))
        if st.appearance_gallery:
            candidates.extend(self._embedding_distance(appearance, g) for g in st.appearance_gallery)
        if not candidates:
            return None
        best = min(candidates)
        mean = float(sum(candidates) / max(len(candidates), 1))
        # Prefer best historical match but keep a small mean term for regularization.
        return 0.75 * best + 0.25 * mean

    @staticmethod
    def _pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if len(a) == 0 or len(b) == 0:
            return np.zeros((len(a), len(b)), dtype=np.float32)
        ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
        bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        ix1 = np.maximum(ax1, bx1)
        iy1 = np.maximum(ay1, by1)
        ix2 = np.minimum(ax2, bx2)
        iy2 = np.minimum(ay2, by2)
        iw = np.maximum(0.0, ix2 - ix1)
        ih = np.maximum(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = np.maximum(0.0, (ax2 - ax1)) * np.maximum(0.0, (ay2 - ay1))
        area_b = np.maximum(0.0, (bx2 - bx1)) * np.maximum(0.0, (by2 - by1))
        union = area_a + area_b - inter + 1e-6
        return (inter / union).astype(np.float32)

    def _apply_frame_continuity(
        self,
        frame_idx: int,
        tracks: sv.Detections,
        class_ids: np.ndarray,
        raw_ids: np.ndarray,
        raw_keys: list[tuple[int, int]],
        centers: np.ndarray,
        appearances: list[np.ndarray | None],
        stable_ids: np.ndarray,
        unmatched_idx: set[int],
        assigned_sid: set[int],
        det_team_ids: np.ndarray | None = None,
    ) -> None:
        # Only use immediate previous frame to suppress one-frame ID ping-pong.
        if self._last_frame_idx is None or self._last_frame_idx != (int(frame_idx) - 1):
            return
        if self._last_boxes is None or self._last_class_ids is None or self._last_stable_ids is None:
            return
        if len(self._last_boxes) == 0 or len(unmatched_idx) == 0:
            return

        for class_id in np.unique(class_ids):
            cur_idx = [i for i in unmatched_idx if int(class_ids[i]) == int(class_id)]
            if not cur_idx:
                continue
            prev_mask = self._last_class_ids == int(class_id)
            if not np.any(prev_mask):
                continue
            prev_boxes = self._last_boxes[prev_mask]
            prev_sid = self._last_stable_ids[prev_mask]
            if len(prev_boxes) == 0:
                continue
            iou = self._pairwise_iou(tracks.xyxy[np.array(cur_idx)], prev_boxes)

            candidate_pairs: list[tuple[float, int, int]] = []
            for r, det_i in enumerate(cur_idx):
                vals = iou[r]
                if vals.size == 0:
                    continue
                order = np.argsort(vals)[::-1]
                best_j = int(order[0])
                best_iou = float(vals[best_j])
                if best_iou < self.frame_continuity_iou:
                    continue
                second_iou = float(vals[int(order[1])]) if len(order) > 1 else -1.0
                if (best_iou - second_iou) < self.frame_continuity_margin:
                    continue
                sid = int(prev_sid[best_j])
                if sid in assigned_sid or sid not in self._stable_tracks:
                    continue
                st_cand = self._stable_tracks.get(sid)
                # Team color veto: don't lock when the candidate stable_id's known team differs.
                if (
                    st_cand is not None
                    and det_team_ids is not None
                    and self._team_mismatch(st_cand, int(det_team_ids[det_i]))
                ):
                    continue
                # Appearance veto: don't lock positionally when jerseys clearly differ.
                if appearances[det_i] is not None and st_cand is not None:
                    d_app = self._appearance_distance(st_cand, appearances[det_i])
                    if d_app is not None and d_app > 0.55:
                        continue
                candidate_pairs.append((best_iou, det_i, sid))

            # Greedy by strongest overlap first to preserve one-to-one mapping.
            candidate_pairs.sort(key=lambda x: x[0], reverse=True)
            used_det: set[int] = set()
            used_sid: set[int] = set()
            for _, det_i, sid in candidate_pairs:
                if det_i in used_det or sid in used_sid:
                    continue
                self._update_stable(
                    sid=sid,
                    class_id=int(class_ids[det_i]),
                    center=centers[det_i],
                    frame_idx=frame_idx,
                    raw_id=int(raw_ids[det_i]),
                    raw_key=raw_keys[det_i],
                    appearance=appearances[det_i],
                    team_id=int(det_team_ids[det_i]) if det_team_ids is not None else -1,
                )
                stable_ids[det_i] = sid
                assigned_sid.add(sid)
                unmatched_idx.discard(det_i)
                used_det.add(det_i)
                used_sid.add(sid)

    def _apply_spatial_rescue(
        self,
        frame_idx: int,
        class_ids: np.ndarray,
        raw_ids: np.ndarray,
        raw_keys: list[tuple[int, int]],
        centers: np.ndarray,
        appearances: list[np.ndarray | None],
        stable_ids: np.ndarray,
        unmatched_idx: set[int],
        assigned_sid: set[int],
        det_team_ids: np.ndarray | None = None,
    ) -> None:
        # Last-chance relink based on motion geometry only, to reduce new-ID fragmentation.
        if len(unmatched_idx) == 0:
            return
        relink_window = max(self.max_relink_frames, self.memory_frames, self.spatial_rescue_max_gap)
        for class_id in np.unique(class_ids):
            det_idx = [i for i in unmatched_idx if int(class_ids[i]) == int(class_id)]
            if not det_idx:
                continue
            for det_i in det_idx:
                cands: list[tuple[float, int]] = []
                for sid, st in self._stable_tracks.items():
                    if sid in assigned_sid or st.class_id != int(class_id):
                        continue
                    gap = int(frame_idx - st.last_seen)
                    if gap <= 0 or gap > relink_window:
                        continue
                    if frame_idx <= st.lock_until:
                        continue
                    if det_team_ids is not None and self._team_mismatch(
                        st, int(det_team_ids[det_i])
                    ):
                        continue
                    # Decay velocity for long gaps: a player who left the frame stops
                    # accelerating away, so anchoring near the exit point is more accurate
                    # than extrapolating at full speed for tens of seconds.
                    vel_weight = max(0.0, 1.0 - float(gap) / 40.0)
                    predicted = st.center + st.velocity * vel_weight * float(min(gap, 40))
                    d_space_px = float(np.linalg.norm(centers[det_i] - predicted))
                    max_space = self.max_relink_px * min(3.2, 1.0 + 0.25 * max(gap, 1))
                    if d_space_px > max_space:
                        continue
                    cands.append((d_space_px / max(max_space, 1e-6), sid))

                if not cands:
                    continue
                cands.sort(key=lambda x: x[0])
                best_norm, best_sid = cands[0]
                take = False
                if len(cands) == 1:
                    take = True
                else:
                    second_norm = cands[1][0]
                    if (second_norm - best_norm) >= self.spatial_rescue_margin:
                        take = True
                    elif second_norm > 1e-6 and (best_norm / second_norm) <= self.spatial_rescue_ratio:
                        take = True
                if not take:
                    continue

                self._update_stable(
                    sid=int(best_sid),
                    class_id=int(class_ids[det_i]),
                    center=centers[det_i],
                    frame_idx=frame_idx,
                    raw_id=int(raw_ids[det_i]),
                    raw_key=raw_keys[det_i],
                    appearance=appearances[det_i],
                    team_id=int(det_team_ids[det_i]) if det_team_ids is not None else -1,
                )
                stable_ids[det_i] = int(best_sid)
                assigned_sid.add(int(best_sid))
                unmatched_idx.discard(det_i)

    def _count_stable_for_class(self, class_id: int) -> int:
        return sum(1 for st in self._stable_tracks.values() if int(st.class_id) == int(class_id))

    def _cap_for_class(self, class_id: int) -> int:
        cid = int(class_id)
        if cid == self.player_class_id:
            return self.max_player_ids
        if cid == self.goalkeeper_class_id:
            return self.max_goalkeeper_ids
        if cid == self.referee_class_id:
            return self.max_referee_ids
        return 0

    def _select_reuse_sid_for_capped(
        self,
        class_id: int,
        frame_idx: int,
        center: np.ndarray,
        appearance: np.ndarray | None,
        assigned_sid: set[int],
    ) -> int | None:
        candidates: list[tuple[float, int]] = []
        for sid, st in self._stable_tracks.items():
            if sid in assigned_sid or int(st.class_id) != int(class_id):
                continue
            gap = max(1, int(frame_idx - st.last_seen))
            if gap < self.capped_reuse_min_gap:
                continue
            predicted = st.center + st.velocity * float(gap)
            d_space_px = float(np.linalg.norm(center - predicted))
            max_space = self.max_relink_px * min(4.5, 1.0 + 0.35 * gap)
            d_space = d_space_px / max(max_space, 1e-6)
            d_app_opt = self._appearance_distance(st, appearance)
            d_app = float(d_app_opt) if d_app_opt is not None else float(self.unknown_appearance_penalty)
            score = d_space + 0.35 * d_app + 0.008 * gap
            candidates.append((score, sid))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return int(candidates[0][1])

    def _select_oldest_sid_for_class(self, class_id: int, assigned_sid: set[int]) -> int | None:
        oldest_sid: int | None = None
        oldest_seen = 10**9
        for sid, st in self._stable_tracks.items():
            if sid in assigned_sid or int(st.class_id) != int(class_id):
                continue
            if int(st.last_seen) < oldest_seen:
                oldest_seen = int(st.last_seen)
                oldest_sid = int(sid)
        return oldest_sid

    def _new_stable(
        self,
        class_id: int,
        center: np.ndarray,
        frame_idx: int,
        raw_id: int,
        raw_key: tuple[int, int],
        appearance: np.ndarray | None,
        team_id: int = -1,
    ) -> int:
        sid = self._next_stable_id
        self._next_stable_id += 1
        self._stable_tracks[sid] = StableTrack(
            stable_id=sid,
            class_id=int(class_id),
            center=center.astype(np.float32),
            last_seen=int(frame_idx),
            velocity=np.zeros((2,), dtype=np.float32),
            raw_id=int(raw_id),
            raw_key=raw_key,
            appearance=appearance,
            appearance_gallery=([appearance.astype(np.float32)] if appearance is not None else []),
            lock_until=int(frame_idx + self.lock_frames),
            team_id=int(team_id),
            team_votes=1 if int(team_id) >= 0 else 0,
        )
        return sid

    def _update_team(self, st: StableTrack, team_id: int) -> None:
        if int(team_id) < 0:
            return
        if st.team_id < 0:
            st.team_id = int(team_id)
            st.team_votes = 1
            return
        if int(team_id) == st.team_id:
            st.team_votes = min(st.team_votes + 1, 999)
        else:
            # Decay confidence; only flip team after sustained disagreement
            st.team_votes = max(0, st.team_votes - 1)
            if st.team_votes == 0:
                st.team_id = int(team_id)
                st.team_votes = 1

    def _update_stable(
        self,
        sid: int,
        class_id: int,
        center: np.ndarray,
        frame_idx: int,
        raw_id: int,
        raw_key: tuple[int, int],
        appearance: np.ndarray | None,
        team_id: int = -1,
    ) -> None:
        st = self._stable_tracks[sid]
        old_key = st.raw_key
        if old_key is not None and old_key != raw_key:
            self._raw_to_stable.pop(old_key, None)

        dt = max(int(frame_idx - st.last_seen), 1)
        displacement = center.astype(np.float32) - st.center
        instant_v = displacement / float(dt)
        st.velocity = 0.7 * st.velocity + 0.3 * instant_v
        st.center = center.astype(np.float32)
        st.last_seen = int(frame_idx)
        st.class_id = int(class_id)
        st.raw_id = int(raw_id)
        st.raw_key = raw_key
        st.lock_until = int(frame_idx + self.lock_frames)
        self._update_team(st, int(team_id))
        self._update_gallery(st, appearance)
        if appearance is not None:
            if st.appearance is None:
                st.appearance = appearance
            else:
                st.appearance = (
                    self.appearance_alpha * st.appearance
                    + (1.0 - self.appearance_alpha) * appearance
                )
                st.appearance /= max(float(np.linalg.norm(st.appearance)), 1e-6)
        self._raw_to_stable[raw_key] = sid

    def _compute_match_cost(
        self,
        st: StableTrack,
        center: np.ndarray,
        appearance: np.ndarray | None,
        raw_key: tuple[int, int],
        frame_idx: int,
    ) -> float:
        gap = int(frame_idx - st.last_seen)
        if gap < 0 or gap > self.max_relink_frames:
            return float("inf")

        predicted = st.center + st.velocity * float(max(gap, 1))
        d_space_px = float(np.linalg.norm(center - predicted))
        max_space = self.max_relink_px * min(2.6, 1.0 + 0.18 * max(gap, 1))
        if d_space_px > max_space:
            return float("inf")
        d_space = d_space_px / max(max_space, 1e-6)

        d_app_opt = self._appearance_distance(st, appearance)
        if d_app_opt is not None:
            d_app = float(d_app_opt)
            if d_app > self.max_appearance_distance:
                return float("inf")
        else:
            # Missing appearance is still allowed for short/medium gaps when motion is strong.
            if gap > self.unknown_appearance_max_gap:
                return float("inf")
            if d_space_px > self.unknown_appearance_max_space_frac * self.max_relink_px:
                return float("inf")
            d_app = self.unknown_appearance_penalty

        lock_penalty = 0.0
        if st.raw_key is not None and st.raw_key != raw_key and frame_idx <= st.lock_until:
            lock_left = max(0, int(st.lock_until - frame_idx + 1))
            lock_penalty = 0.35 * (float(lock_left) / float(max(self.lock_frames, 1)))

        raw_penalty = -0.20 if st.raw_key == raw_key else 0.04
        gap_penalty = 0.02 * gap
        return d_space + self.appearance_weight * d_app + raw_penalty + gap_penalty + lock_penalty

    def _team_mismatch(self, st: StableTrack, det_team: int) -> bool:
        """Hard veto when both teams are confidently known and differ."""
        if int(det_team) < 0 or st.team_id < 0:
            return False
        if int(st.team_votes) < 3:
            return False
        return int(det_team) != int(st.team_id)

    def update(
        self,
        frame_idx: int,
        frame: np.ndarray,
        tracks: sv.Detections,
        det_team_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        if len(tracks) == 0 or tracks.tracker_id is None:
            return np.zeros((0,), dtype=np.int32)

        centers = self._centers(tracks.xyxy)
        stable_ids = np.full((len(tracks),), -1, dtype=np.int32)
        raw_ids = tracks.tracker_id.astype(np.int32)
        class_ids = tracks.class_id.astype(np.int32)
        raw_keys = [(int(class_ids[i]), int(raw_ids[i])) for i in range(len(tracks))]
        appearances = [self._appearance_feature(frame, tracks.xyxy[i]) for i in range(len(tracks))]
        if det_team_ids is None or len(det_team_ids) != len(tracks):
            det_team_ids = np.full((len(tracks),), -1, dtype=np.int32)
        else:
            det_team_ids = np.asarray(det_team_ids, dtype=np.int32)
        unmatched_idx = set(range(len(tracks)))
        assigned_sid: set[int] = set()

        # Pass 1: immediate frame-to-frame continuity lock to stop adjacent-frame ID flips.
        self._apply_frame_continuity(
            frame_idx=frame_idx,
            tracks=tracks,
            class_ids=class_ids,
            raw_ids=raw_ids,
            raw_keys=raw_keys,
            centers=centers,
            appearances=appearances,
            stable_ids=stable_ids,
            unmatched_idx=unmatched_idx,
            assigned_sid=assigned_sid,
            det_team_ids=det_team_ids,
        )

        # Pass 2: preserve direct raw->stable continuity when still plausible.
        for i in sorted(unmatched_idx):
            raw_key = raw_keys[i]
            sid = self._raw_to_stable.get(raw_key)
            if sid is None or sid in assigned_sid:
                continue
            st = self._stable_tracks.get(sid)
            if st is None or st.class_id != int(class_ids[i]):
                self._raw_to_stable.pop(raw_key, None)
                continue
            # Team color veto: invalidate continuity across confirmed team change.
            if self._team_mismatch(st, int(det_team_ids[i])):
                self._raw_to_stable.pop(raw_key, None)
                continue
            # Appearance veto: if jerseys clearly differ, invalidate raw_key continuity.
            if appearances[i] is not None:
                d_app = self._appearance_distance(st, appearances[i])
                if d_app is not None and d_app > 0.55:
                    self._raw_to_stable.pop(raw_key, None)
                    continue
            continuity_cost = self._compute_match_cost(
                st=st,
                center=centers[i],
                appearance=appearances[i],
                raw_key=raw_key,
                frame_idx=frame_idx,
            )
            if not np.isfinite(continuity_cost) or float(continuity_cost) > (self.max_relink_cost + 0.25):
                self._raw_to_stable.pop(raw_key, None)
                continue
            self._update_stable(
                sid=sid,
                class_id=int(class_ids[i]),
                center=centers[i],
                frame_idx=frame_idx,
                raw_id=int(raw_ids[i]),
                raw_key=raw_key,
                appearance=appearances[i],
                team_id=int(det_team_ids[i]),
            )
            stable_ids[i] = sid
            assigned_sid.add(sid)
            unmatched_idx.discard(i)

        # Pass 3: global one-to-one assignment per class using motion+appearance ReID cost.
        for class_id in np.unique(class_ids):
            class_det_idx = [i for i in unmatched_idx if int(class_ids[i]) == int(class_id)]
            if not class_det_idx:
                continue
            relink_window = max(1, max(self.max_relink_frames, self.memory_frames))
            class_sid = [
                sid
                for sid, st in self._stable_tracks.items()
                if sid not in assigned_sid
                and st.class_id == int(class_id)
                and int(frame_idx - st.last_seen) <= relink_window
            ]
            if not class_sid:
                continue

            cost = np.full((len(class_det_idx), len(class_sid)), np.inf, dtype=np.float32)
            for r, det_i in enumerate(class_det_idx):
                for c, sid in enumerate(class_sid):
                    st = self._stable_tracks[sid]
                    if self._team_mismatch(st, int(det_team_ids[det_i])):
                        continue
                    cost[r, c] = self._compute_match_cost(
                        st=st,
                        center=centers[det_i],
                        appearance=appearances[det_i],
                        raw_key=raw_keys[det_i],
                        frame_idx=frame_idx,
                    )

            finite = np.isfinite(cost)
            if not np.any(finite):
                continue

            if linear_sum_assignment is not None:
                work = np.where(finite, cost, 1e6)
                row_ind, col_ind = linear_sum_assignment(work)
                pairs = list(zip(row_ind.tolist(), col_ind.tolist()))
            else:
                triples = []
                for r in range(cost.shape[0]):
                    for c in range(cost.shape[1]):
                        if np.isfinite(cost[r, c]):
                            triples.append((float(cost[r, c]), r, c))
                triples.sort(key=lambda x: x[0])
                used_r: set[int] = set()
                used_c: set[int] = set()
                pairs = []
                for _, r, c in triples:
                    if r in used_r or c in used_c:
                        continue
                    used_r.add(r)
                    used_c.add(c)
                    pairs.append((r, c))

            for r, c in pairs:
                pair_cost = float(cost[r, c])
                if (not np.isfinite(pair_cost)) or pair_cost > self.max_relink_cost:
                    continue

                # Ambiguity guard: stricter only for longer gaps; short-gap reacquire should be permissive.
                gap = int(frame_idx - self._stable_tracks[class_sid[c]].last_seen)
                margin = self.ambiguity_margin * (1.0 + 0.02 * max(gap, 0))
                row_vals = sorted(float(v) for v in cost[r, :] if np.isfinite(v))
                if gap > 2 and len(row_vals) >= 2 and (row_vals[1] - row_vals[0]) < margin:
                    continue
                col_vals = sorted(float(v) for v in cost[:, c] if np.isfinite(v))
                if gap > 2 and len(col_vals) >= 2 and (col_vals[1] - col_vals[0]) < margin:
                    continue
                det_i = class_det_idx[r]
                sid = class_sid[c]
                self._update_stable(
                    sid=sid,
                    class_id=int(class_ids[det_i]),
                    center=centers[det_i],
                    frame_idx=frame_idx,
                    raw_id=int(raw_ids[det_i]),
                    raw_key=raw_keys[det_i],
                    appearance=appearances[det_i],
                    team_id=int(det_team_ids[det_i]),
                )
                stable_ids[det_i] = sid
                assigned_sid.add(sid)
                unmatched_idx.discard(det_i)

        # Pass 4: geometry-only rescue to avoid unnecessary new IDs after occlusions.
        self._apply_spatial_rescue(
            frame_idx=frame_idx,
            class_ids=class_ids,
            raw_ids=raw_ids,
            raw_keys=raw_keys,
            centers=centers,
            appearances=appearances,
            stable_ids=stable_ids,
            unmatched_idx=unmatched_idx,
            assigned_sid=assigned_sid,
            det_team_ids=det_team_ids,
        )

        # Create new IDs for still-unmatched detections.
        for i in sorted(unmatched_idx):
            class_id_i = int(class_ids[i])
            cap_for_class = self._cap_for_class(class_id_i)
            if (
                self.hard_reuse_when_capped
                and cap_for_class > 0
                and self._count_stable_for_class(class_id_i) >= cap_for_class
            ):
                reuse_sid = self._select_reuse_sid_for_capped(
                    class_id=class_id_i,
                    frame_idx=frame_idx,
                    center=centers[i],
                    appearance=appearances[i],
                    assigned_sid=assigned_sid,
                )
                if reuse_sid is None:
                    reuse_sid = self._select_oldest_sid_for_class(
                        class_id=class_id_i,
                        assigned_sid=assigned_sid,
                    )
                if reuse_sid is not None:
                    self._update_stable(
                        sid=int(reuse_sid),
                        class_id=class_id_i,
                        center=centers[i],
                        frame_idx=frame_idx,
                        raw_id=int(raw_ids[i]),
                        raw_key=raw_keys[i],
                        appearance=appearances[i],
                        team_id=int(det_team_ids[i]),
                    )
                    stable_ids[i] = int(reuse_sid)
                    assigned_sid.add(int(reuse_sid))
                    self._raw_to_stable[raw_keys[i]] = int(reuse_sid)
                    continue

            sid = self._new_stable(
                class_id=class_id_i,
                center=centers[i],
                frame_idx=frame_idx,
                raw_id=int(raw_ids[i]),
                team_id=int(det_team_ids[i]),
                raw_key=raw_keys[i],
                appearance=appearances[i],
            )
            self._raw_to_stable[raw_keys[i]] = sid
            stable_ids[i] = sid

        # Rebuild raw->stable map from recent stable tracks only.
        rebuilt: dict[tuple[int, int], int] = {}
        for sid, st in self._stable_tracks.items():
            if st.raw_key is None:
                continue
            if frame_idx - st.last_seen > self.max_relink_frames * 3:
                continue
            prev_sid = rebuilt.get(st.raw_key)
            if prev_sid is None:
                rebuilt[st.raw_key] = sid
            else:
                prev = self._stable_tracks.get(prev_sid)
                if prev is None or st.last_seen > prev.last_seen:
                    rebuilt[st.raw_key] = sid
        self._raw_to_stable = rebuilt

        # Cache current frame assignments for next-frame continuity.
        self._last_frame_idx = int(frame_idx)
        self._last_boxes = tracks.xyxy.astype(np.float32).copy()
        self._last_class_ids = class_ids.astype(np.int32).copy()
        self._last_stable_ids = stable_ids.astype(np.int32).copy()

        return stable_ids
