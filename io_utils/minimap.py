from __future__ import annotations

import cv2
import numpy as np

from vision.detect import PLAYER_ID, REFEREE_ID, GOALKEEPER_ID


TEAM_COLORS = {
    -1: (220, 220, 220),   # unknown
    0: (255, 80, 80),      # team A
    1: (80, 160, 255),     # team B
}


def _pitch_bounds(vertices: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    arr = np.asarray(vertices, dtype=np.float32)
    xmin = float(np.min(arr[:, 0]))
    xmax = float(np.max(arr[:, 0]))
    ymin = float(np.min(arr[:, 1]))
    ymax = float(np.max(arr[:, 1]))
    return xmin, xmax, ymin, ymax


def _to_panel_xy(
    pts: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: int,
    height: int,
    pad: int,
) -> np.ndarray:
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    x = pts[:, 0]
    y = pts[:, 1]
    sx = (width - 2 * pad) / max(xmax - xmin, 1e-6)
    sy = (height - 2 * pad) / max(ymax - ymin, 1e-6)
    px = ((x - xmin) * sx + pad).astype(np.int32)
    py = ((y - ymin) * sy + pad).astype(np.int32)
    py = (height - py).astype(np.int32)  # invert for top-down view
    return np.stack([px, py], axis=1)


def render_side_panel(
    frame: np.ndarray,
    pitch_vertices: list[tuple[float, float]],
    pitch_xy_tracks: np.ndarray,
    track_class_ids: np.ndarray,
    track_ids: np.ndarray,
    team_by_track: dict[int, int],
    pitch_xy_ball: np.ndarray,
    homography_ok: bool,
) -> np.ndarray:
    h, _ = frame.shape[:2]
    panel_w = max(340, int(h * 0.72))
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    panel[:, :] = (24, 84, 24)

    pad = 18
    inner_tl = (pad, pad)
    inner_br = (panel_w - pad, h - pad)
    cv2.rectangle(panel, inner_tl, inner_br, (240, 240, 240), 2)
    mid_x = panel_w // 2
    cv2.line(panel, (mid_x, pad), (mid_x, h - pad), (220, 220, 220), 2)
    cv2.circle(panel, (mid_x, h // 2), max(20, int(h * 0.08)), (220, 220, 220), 2)

    xmin, xmax, ymin, ymax = _pitch_bounds(pitch_vertices)

    # Players / referees
    if pitch_xy_tracks.size > 0:
        mask = np.isfinite(pitch_xy_tracks).all(axis=1)
        good_pts = pitch_xy_tracks[mask]
        good_cls = track_class_ids[mask]
        good_tid = track_ids[mask]
        panel_pts = _to_panel_xy(good_pts, xmin, xmax, ymin, ymax, panel_w, h, pad)
        for (px, py), cls_id, tid in zip(panel_pts, good_cls, good_tid):
            if cls_id in (PLAYER_ID, GOALKEEPER_ID):
                color = TEAM_COLORS.get(team_by_track.get(int(tid), -1), TEAM_COLORS[-1])
                radius = 5
            elif cls_id == REFEREE_ID:
                color = (30, 30, 30)
                radius = 5
            else:
                color = (180, 180, 180)
                radius = 4
            cv2.circle(panel, (int(px), int(py)), radius, color, -1)

    # Ball
    if pitch_xy_ball.size > 0:
        bmask = np.isfinite(pitch_xy_ball).all(axis=1)
        bpts = pitch_xy_ball[bmask]
        bpanel = _to_panel_xy(bpts, xmin, xmax, ymin, ymax, panel_w, h, pad)
        for px, py in bpanel:
            cv2.circle(panel, (int(px), int(py)), 4, (0, 200, 255), -1)

    status = "H: OK" if homography_ok else "H: HOLD"
    cv2.putText(panel, status, (pad + 4, pad + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(panel, status, (pad + 4, pad + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def compose_side_by_side(frame: np.ndarray, panel: np.ndarray, left_ratio: float = 0.62) -> np.ndarray:
    """
    Keep final frame size unchanged while showing a clean side-by-side layout.
    """
    h, w = frame.shape[:2]
    left_w = int(np.clip(left_ratio, 0.45, 0.80) * w)
    right_w = max(1, w - left_w)

    left = cv2.resize(frame, (left_w, h), interpolation=cv2.INTER_AREA)
    right = cv2.resize(panel, (right_w, h), interpolation=cv2.INTER_AREA)

    out = np.zeros_like(frame)
    out[:, :left_w] = left
    out[:, left_w:] = right
    return out
