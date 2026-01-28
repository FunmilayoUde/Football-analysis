from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import supervision as sv
from tqdm import tqdm
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch

from core.config import load_settings
from vision.models import load_roboflow_models
from vision.detect import (
    infer_players_and_ball,
    infer_field_keypoints,
    tiny_box_filter,
    BALL_ID, PLAYER_ID, REFEREE_ID, GOALKEEPER_ID
)
from vision.track import ByteTrackWrapper
from vision.teams import TeamClassifierWrapper
from geometry.homography import HomographyEstimator
from geometry.projection import project_anchors_to_pitch
from io_utils.writers import CSVWriter, VideoWriter


def main(
    source_video: str,
    out_dir: str = "outputs",
    fit_team_stride: int = 30,
    fit_team_max_frames: int = 120,  # ~ few mins depending on stride; tune
    enable_team: bool = True,
):
    s = load_settings()

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # Models
    player_model, field_model = load_roboflow_models(
        api_key=s.ROBOFLOW_API_KEY,
        player_model_id=s.PLAYER_MODEL_ID,
        field_model_id=s.FIELD_MODEL_ID
    )

    # Video
    video_info = sv.VideoInfo.from_video_path(source_video)
    frames = sv.get_video_frames_generator(source_video)

    # Components
    tracker = ByteTrackWrapper()
    pitch_cfg = SoccerPitchConfiguration()

    h_est = HomographyEstimator(
        config=pitch_cfg,
        kp_conf=s.KP_CONF,
        ema_alpha=s.H_EMA_ALPHA,
        ransac_reproj_thresh=s.RANSAC_REPROJ_THRESH,
        min_kp=s.MIN_KP,
        min_inlier_ratio=s.MIN_INLIER_RATIO,
        max_reproj_err=s.MAX_REPROJ_ERR,
    )

    team_clf = None
    if enable_team:
        team_clf = TeamClassifierWrapper(device=s.DEVICE)

        # Fit from strided frames (like notebook)
        fit_gen = sv.get_video_frames_generator(source_video, stride=fit_team_stride)
        team_clf.fit_from_video_frames(
            frame_iter=fit_gen,
            player_model=player_model,
            det_conf=s.DET_CONF,
            player_class_id=PLAYER_ID,
            max_frames=fit_team_max_frames,
        )

    # Annotators
    ellipse_annotator = sv.EllipseAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_CENTER)

    csv_path = str(out_dir_p / "per_frame_tracks.csv")
    out_video_path = str(out_dir_p / "annotated.mp4")
    csvw = CSVWriter(csv_path)

    start_time = time.time()
    with VideoWriter(out_video_path, video_info) as vw:
        for frame_idx, frame in tqdm(enumerate(frames), total=video_info.total_frames):
            h, w = frame.shape[:2]

            # 1) Detect (players+ball)
            det = infer_players_and_ball(player_model, frame, conf=s.DET_CONF)

            # tiny-box filter
            det = tiny_box_filter(
                det, w, h,
                min_area_ratio_people=s.MIN_AREA_RATIO_PEOPLE,
                min_area_ratio_ball=s.MIN_AREA_RATIO_BALL,
            )

            # pad ball boxes
            ball_det = det[det.class_id == BALL_ID]
            if len(ball_det) > 0 and s.BALL_PAD_PX > 0:
                ball_det.xyxy = sv.pad_boxes(ball_det.xyxy, px=s.BALL_PAD_PX)

            non_ball = det[det.class_id != BALL_ID]

            # 2) Track non-ball
            tracks = tracker.update(non_ball)

            # 3) Team classification (players only)
            team_ids = None
            team_by_track = {}
            if team_clf is not None:
                players_tr = tracks[tracks.class_id == PLAYER_ID]
                team_ids = team_clf.predict_team_ids(frame, players_tr)

                # map team ids back to global tracks via tracker_id
                if team_ids is not None and players_tr.tracker_id is not None:
                    for tid, t in zip(players_tr.tracker_id, team_ids):
                        team_by_track[int(tid)] = int(t)

            # 4) Field keypoints -> Homography
            kp = infer_field_keypoints(field_model, frame, conf=s.FIELD_CONF)
            hres = h_est.estimate(kp)

            # ✅ FIX: no ViewTransformer / no make_transformer
            Hmat = hres.H

            # 5) Project to pitch coords (if H available)
            # Use NaN when not available (looks more professional than zeros)
            pitch_xy_tracks = np.full((len(tracks), 2), np.nan, dtype=np.float32)
            pitch_xy_ball = np.full((len(ball_det), 2), np.nan, dtype=np.float32)

            if Hmat is not None:
                pitch_xy_tracks = project_anchors_to_pitch(Hmat, tracks, anchor=sv.Position.BOTTOM_CENTER)
                pitch_xy_ball = project_anchors_to_pitch(Hmat, ball_det, anchor=sv.Position.BOTTOM_CENTER)

            # 6) Write CSV rows
            # tracks
            for i in range(len(tracks)):
                x1, y1, x2, y2 = tracks.xyxy[i].tolist()
                track_id = int(tracks.tracker_id[i]) if tracks.tracker_id is not None else -1
                class_id = int(tracks.class_id[i])

                row = {
                    "frame": frame_idx,
                    "track_id": track_id,
                    "class_id": class_id,
                    "conf": float(tracks.confidence[i]) if tracks.confidence is not None else 0.0,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "x_m": float(pitch_xy_tracks[i, 0]),
                    "y_m": float(pitch_xy_tracks[i, 1]),
                    "homography_ok": bool(hres.ok),
                    "kp_used": int(hres.n_points),
                    "inlier_ratio": float(hres.inlier_ratio),
                    "reproj_err": float(hres.reproj_err),
                }

                # Attach team_id only for players (if available)
                if class_id == PLAYER_ID:
                    row["team_id"] = team_by_track.get(track_id, -1)

                csvw.write_row(row)

            # ball as separate rows (track_id -1)
            for j in range(len(ball_det)):
                x1, y1, x2, y2 = ball_det.xyxy[j].tolist()
                row = {
                    "frame": frame_idx,
                    "track_id": -1,
                    "class_id": int(ball_det.class_id[j]),
                    "conf": float(ball_det.confidence[j]) if ball_det.confidence is not None else 0.0,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "x_m": float(pitch_xy_ball[j, 0]),
                    "y_m": float(pitch_xy_ball[j, 1]),
                    "homography_ok": bool(hres.ok),
                    "kp_used": int(hres.n_points),
                    "inlier_ratio": float(hres.inlier_ratio),
                    "reproj_err": float(hres.reproj_err),
                }
                csvw.write_row(row)

            
            annotated = frame.copy()
            annotated = ellipse_annotator.annotate(annotated, tracks)

            if tracks.tracker_id is not None and len(tracks) > 0:
                labels = [f"id:{int(tid)} cls:{int(cid)}" for tid, cid in zip(tracks.tracker_id, tracks.class_id)]
                annotated = label_annotator.annotate(annotated, tracks, labels=labels)

            vw.write(annotated)

    csvw.close()

    elapsed = time.time() - start_time
    fps = video_info.total_frames / max(elapsed, 1e-6)
    print(f"Done. Output video: {out_video_path}")
    print(f"CSV: {csv_path}")
    print(f"Elapsed: {elapsed:.2f}s | Effective FPS: {fps:.2f}")


if __name__ == "__main__":
    SOURCE_VIDEO = r"C:\Users\Chimdi\Downloads\analytics\HILAL-AHLI_match_B_up1.mp4"
    main(SOURCE_VIDEO, out_dir="outputs", enable_team=True)
