from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Dict, Any, Optional

import numpy as np
import supervision as sv


class CSVWriter:
    def __init__(self, path: str, fieldnames: list[str] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("w", newline="", encoding="utf-8")
        self.w = None
        self.fieldnames = fieldnames
        if self.fieldnames is not None:
            self.w = csv.DictWriter(self.f, fieldnames=self.fieldnames, extrasaction="ignore")
            self.w.writeheader()

    def write_row(self, row: Dict[str, Any]) -> None:
        if self.w is None:
            self.w = csv.DictWriter(self.f, fieldnames=list(row.keys()), extrasaction="ignore")
            self.w.writeheader()
        self.w.writerow(row)

    def close(self) -> None:
        self.f.close()


class VideoWriter:
    def __init__(self, out_path: str, video_info: sv.VideoInfo):
        self.out_path = out_path
        self.sink = sv.VideoSink(target_path=out_path, video_info=video_info)

    def __enter__(self):
        self.sink.__enter__()
        return self

    def write(self, frame):
        self.sink.write_frame(frame)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.sink.__exit__(exc_type, exc_val, exc_tb)
