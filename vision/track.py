from __future__ import annotations
import inspect
from types import SimpleNamespace
from typing import Protocol
import numpy as np
import supervision as sv


class TrackerLike(Protocol):
    def reset(self) -> None:
        ...

    def update(self, detections: sv.Detections, frame: np.ndarray | None = None) -> sv.Detections:
        ...


class ByteTrackWrapper:
    def __init__(
        self,
        track_activation_threshold: float | None = None,
        minimum_matching_threshold: float | None = None,
        lost_track_buffer: int | None = None,
        minimum_consecutive_frames: int | None = None,
    ):
        params = {
            "track_activation_threshold": track_activation_threshold,
            "minimum_matching_threshold": minimum_matching_threshold,
            "lost_track_buffer": lost_track_buffer,
            "minimum_consecutive_frames": minimum_consecutive_frames,
        }

        # Keep compatibility across supervision versions with different ByteTrack signatures.
        sig = inspect.signature(sv.ByteTrack).parameters
        kwargs = {k: v for k, v in params.items() if v is not None and k in sig}

        self.tracker = sv.ByteTrack(**kwargs)
        self.tracker.reset()

    def reset(self) -> None:
        self.tracker.reset()

    def update(self, detections: sv.Detections, frame: np.ndarray | None = None) -> sv.Detections:
        return self.tracker.update_with_detections(detections)


def empty_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.zeros((0, 4), dtype=np.float32),
        confidence=np.zeros((0,), dtype=np.float32),
        class_id=np.zeros((0,), dtype=np.int32),
    )


def merge_detections(parts: list[sv.Detections]) -> sv.Detections:
    valid = [p for p in parts if p is not None and len(p) > 0]
    if not valid:
        return empty_detections()

    xyxy = np.concatenate([p.xyxy for p in valid], axis=0)
    conf = np.concatenate(
        [p.confidence if p.confidence is not None else np.zeros((len(p),), dtype=np.float32) for p in valid],
        axis=0,
    )
    class_id = np.concatenate(
        [p.class_id if p.class_id is not None else np.zeros((len(p),), dtype=np.int32) for p in valid],
        axis=0,
    )
    out = sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)

    has_tracker = all((p.tracker_id is not None) for p in valid)
    if has_tracker:
        out.tracker_id = np.concatenate([p.tracker_id for p in valid], axis=0).astype(np.int32)
    return out


class BoTSORTWrapper:
    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        minimum_matching_threshold: float = 0.80,
        lost_track_buffer: int = 45,
        frame_rate: int = 30,
        track_low_threshold: float = 0.10,
        gmc_method: str = "sparseOptFlow",
        with_reid: bool = False,
        reid_model: str = "yolo11n-cls.pt",
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.8,
    ):
        try:
            from ultralytics.trackers.bot_sort import BOTSORT
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError("ultralytics/BoT-SORT dependencies are not installed") from e

        model_name = str(reid_model).strip() if reid_model is not None else ""
        if with_reid and (not model_name or model_name.lower() == "auto"):
            # We do not pass native YOLO head features in this pipeline, so "auto" is unsafe here.
            model_name = "yolo11n-cls.pt"

        args = SimpleNamespace(
            tracker_type="botsort",
            track_high_thresh=float(track_activation_threshold),
            track_low_thresh=float(track_low_threshold),
            new_track_thresh=float(track_activation_threshold),
            track_buffer=int(lost_track_buffer),
            match_thresh=float(minimum_matching_threshold),
            fuse_score=True,
            gmc_method=str(gmc_method),
            proximity_thresh=float(proximity_thresh),
            appearance_thresh=float(appearance_thresh),
            with_reid=bool(with_reid),
            model=model_name if with_reid else "auto",
        )

        self.tracker = BOTSORT(args=args, frame_rate=max(1, int(round(frame_rate))))
        self.tracker.reset()

    @staticmethod
    def _to_ultralytics_boxes(detections: sv.Detections, h: int, w: int):
        from ultralytics.engine.results import Boxes

        if len(detections) == 0:
            arr = np.zeros((0, 6), dtype=np.float32)
            return Boxes(arr, (int(h), int(w)))

        conf = (
            detections.confidence.astype(np.float32)
            if detections.confidence is not None
            else np.ones((len(detections),), dtype=np.float32)
        )
        class_id = (
            detections.class_id.astype(np.float32)
            if detections.class_id is not None
            else np.zeros((len(detections),), dtype=np.float32)
        )
        arr = np.concatenate(
            [
                detections.xyxy.astype(np.float32),
                conf.reshape(-1, 1),
                class_id.reshape(-1, 1),
            ],
            axis=1,
        )
        return Boxes(arr, (int(h), int(w)))

    @staticmethod
    def _from_ultralytics_tracks(tracks: np.ndarray | None) -> sv.Detections:
        if tracks is None:
            return empty_detections()
        arr = np.asarray(tracks)
        if arr.size == 0:
            return empty_detections()
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 7:
            return empty_detections()

        out = sv.Detections(
            xyxy=arr[:, 0:4].astype(np.float32),
            confidence=arr[:, 5].astype(np.float32),
            class_id=arr[:, 6].astype(np.int32),
        )
        out.tracker_id = arr[:, 4].astype(np.int32)
        return out

    def reset(self) -> None:
        self.tracker.reset()

    def update(self, detections: sv.Detections, frame: np.ndarray | None = None) -> sv.Detections:
        if frame is None:
            raise ValueError("BoTSORTWrapper.update requires current frame.")
        h, w = frame.shape[:2]
        boxes = self._to_ultralytics_boxes(detections, h=h, w=w)
        tracks = self.tracker.update(boxes, frame)
        return self._from_ultralytics_tracks(tracks)


def create_tracker(
    tracker_type: str = "bytetrack",
    track_activation_threshold: float | None = None,
    minimum_matching_threshold: float | None = None,
    lost_track_buffer: int | None = None,
    minimum_consecutive_frames: int | None = None,
    frame_rate: float | int = 30,
    track_low_threshold: float = 0.10,
    gmc_method: str = "sparseOptFlow",
    with_reid: bool = False,
    reid_model: str = "yolo11n-cls.pt",
    proximity_thresh: float = 0.5,
    appearance_thresh: float = 0.8,
) -> TrackerLike:
    mode = str(tracker_type).strip().lower()
    if mode == "botsort":
        try:
            return BoTSORTWrapper(
                track_activation_threshold=(
                    0.25 if track_activation_threshold is None else float(track_activation_threshold)
                ),
                minimum_matching_threshold=(
                    0.80 if minimum_matching_threshold is None else float(minimum_matching_threshold)
                ),
                lost_track_buffer=(45 if lost_track_buffer is None else int(lost_track_buffer)),
                frame_rate=max(1, int(round(float(frame_rate)))),
                track_low_threshold=float(track_low_threshold),
                gmc_method=str(gmc_method),
                with_reid=bool(with_reid),
                reid_model=str(reid_model),
                proximity_thresh=float(proximity_thresh),
                appearance_thresh=float(appearance_thresh),
            )
        except Exception as e:
            if with_reid:
                print(f"[WARN] BoT-SORT ReID unavailable ({e}). Retrying BoT-SORT without ReID.")
                try:
                    return BoTSORTWrapper(
                        track_activation_threshold=(
                            0.25 if track_activation_threshold is None else float(track_activation_threshold)
                        ),
                        minimum_matching_threshold=(
                            0.80 if minimum_matching_threshold is None else float(minimum_matching_threshold)
                        ),
                        lost_track_buffer=(45 if lost_track_buffer is None else int(lost_track_buffer)),
                        frame_rate=max(1, int(round(float(frame_rate)))),
                        track_low_threshold=float(track_low_threshold),
                        gmc_method=str(gmc_method),
                        with_reid=False,
                        reid_model=str(reid_model),
                        proximity_thresh=float(proximity_thresh),
                        appearance_thresh=float(appearance_thresh),
                    )
                except Exception as e2:
                    print(f"[WARN] BoT-SORT unavailable ({e2}). Falling back to ByteTrack.")
            else:
                print(f"[WARN] BoT-SORT unavailable ({e}). Falling back to ByteTrack.")

    return ByteTrackWrapper(
        track_activation_threshold=track_activation_threshold,
        minimum_matching_threshold=minimum_matching_threshold,
        lost_track_buffer=lost_track_buffer,
        minimum_consecutive_frames=minimum_consecutive_frames,
    )


class TrackManager:
    """
    Wrapper that keeps player/referee IDs stable when detection runs every N frames.
    """
    def __init__(self, tracker: TrackerLike, detect_every_n: int = 1, max_stale_frames: int = 2):
        self.tracker = tracker
        self.detect_every_n = max(1, int(detect_every_n))
        self.max_stale_frames = max(0, int(max_stale_frames))
        self._last_tracks: sv.Detections | None = None
        self._stale = 0

    def reset(self) -> None:
        self.tracker.reset()
        self._last_tracks = None
        self._stale = 0

    def should_detect(self, frame_idx: int) -> bool:
        return frame_idx % self.detect_every_n == 0 or self._last_tracks is None

    def update(
        self,
        frame_idx: int,
        detections: sv.Detections | None,
        frame: np.ndarray | None = None,
    ) -> tuple[sv.Detections, bool]:
        if self.should_detect(frame_idx):
            det = detections if detections is not None else empty_detections()
            tracks = self.tracker.update(det, frame=frame)
            self._last_tracks = tracks
            self._stale = 0
            return tracks, True

        self._stale += 1
        if self._last_tracks is not None and self._stale <= self.max_stale_frames:
            return self._last_tracks, False

        det = detections if detections is not None else empty_detections()
        tracks = self.tracker.update(det, frame=frame)
        self._last_tracks = tracks
        self._stale = 0
        return tracks, True
