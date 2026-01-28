from __future__ import annotations
import supervision as sv


class ByteTrackWrapper:
    def __init__(self):
        self.tracker = sv.ByteTrack()
        self.tracker.reset()

    def reset(self) -> None:
        self.tracker.reset()

    def update(self, detections: sv.Detections) -> sv.Detections:
        # tracker_id will be populated
        return self.tracker.update_with_detections(detections)
