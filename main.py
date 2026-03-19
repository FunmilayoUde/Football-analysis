from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import supervision as sv
from tqdm import tqdm
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch

import logging
from core.config import load_settings
from core.utils import log_frame_diagnostics, detect_camera_motion
from vision.models import load_roboflow_models
from vision.detect import (
    infer_players_and_ball,
    infer_field_keypoints,
    tiny_box_filter,
    BALL_ID, PLAYER_ID, REFEREE_ID, GOALKEEPER_ID
)
from vision.track import EnhancedTracker
from vision.reid import ReIDExtractor
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
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
    reid_extractor = None
    if s.REID_ENABLED:
        reid_extractor = ReIDExtractor(device=s.DEVICE, fp16=s.REID_FP16)

    tracker = EnhancedTracker(
        track_activation_threshold=s.TRACK_ACTIVATION_THRESHOLD,
        lost_track_buffer=s.LOST_TRACK_BUFFER,
        minimum_matching_threshold=s.MINIMUM_MATCHING_THRESHOLD,
        frame_rate=s.TRACK_FRAME_RATE,
        reid_extractor=reid_extractor,
        reid_match_threshold=s.REID_MATCH_THRESHOLD,
        reid_gallery_alpha=s.REID_GALLERY_ALPHA,
        reid_lost_buffer=s.REID_LOST_BUFFER,
    )
    pitch_cfg = SoccerPitchConfiguration()

    h_est = HomographyEstimator(
        config=pitch_cfg,
        kp_conf=s.KP_CONF,
        ema_alpha=s.H_EMA_ALPHA,
        ransac_reproj_thresh=s.RANSAC_REPROJ_THRESH,
        min_kp=s.MIN_KP,
        min_inlier_ratio=s.MIN_INLIER_RATIO,
        max_reproj_err=s.MAX_REPROJ_ERR,
        jump_gate_px=s.H_JUMP_GATE_PX,
        fallback_expiry=s.H_FALLBACK_EXPIRY,
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

    prev_frame = None
    last_hres = None
    stage_times: dict[str, float] = {}

    start_time = time.time()
    with VideoWriter(out_video_path, video_info) as vw:
        for frame_idx, frame in tqdm(enumerate(frames), total=video_info.total_frames):
            h, w = frame.shape[:2]

            # 1) Detect
            t0 = time.perf_counter()
            det = infer_players_and_ball(player_model, frame, conf=s.DET_CONF)
            det = tiny_box_filter(
                det, w, h,
                min_area_ratio_people=s.MIN_AREA_RATIO_PEOPLE,
                min_area_ratio_ball=s.MIN_AREA_RATIO_BALL,
            )
            stage_times["detect"] = time.perf_counter() - t0

            ball_det = det[det.class_id == BALL_ID]
            if len(ball_det) > 0 and s.BALL_PAD_PX > 0:
                ball_det.xyxy = sv.pad_boxes(ball_det.xyxy, px=s.BALL_PAD_PX)
            non_ball = det[det.class_id != BALL_ID]

            # 2) Track (with ReID)
            t0 = time.perf_counter()
            tracks = tracker.update(non_ball, frame=frame)
            stage_times["track"] = time.perf_counter() - t0

            # 3) Team classification
            t0 = time.perf_counter()
            team_by_track = {}
            if team_clf is not None:
                players_tr = tracks[tracks.class_id == PLAYER_ID]
                team_ids = team_clf.predict_team_ids(frame, players_tr)
                if team_ids is not None and players_tr.tracker_id is not None:
                    for tid, t in zip(players_tr.tracker_id, team_ids):
                        team_by_track[int(tid)] = int(t)
            stage_times["team"] = time.perf_counter() - t0

            # 4) Keypoint skip + adaptive scheduling
            t0 = time.perf_counter()
            run_kp = (
                frame_idx == 0
                or frame_idx % s.KP_SKIP_FRAMES == 0
                or last_hres is None
                or not last_hres.ok
            )

            if not run_kp and prev_frame is not None:
                if detect_camera_motion(prev_frame, frame, threshold=s.KP_MOTION_THRESHOLD):
                    run_kp = True

            if run_kp:
                kp = infer_field_keypoints(field_model, frame, conf=s.FIELD_CONF)
                hres = h_est.estimate(kp)
                last_hres = hres
            else:
                hres = last_hres
            stage_times["homography"] = time.perf_counter() - t0

            log_frame_diagnostics(frame_idx, tracker.last_stats, hres)

            Hmat = hres.H

            # 5) Project to pitch coords
            pitch_xy_tracks = np.full((len(tracks), 2), np.nan, dtype=np.float32)
            pitch_xy_ball = np.full((len(ball_det), 2), np.nan, dtype=np.float32)
            if Hmat is not None:
                pitch_xy_tracks = project_anchors_to_pitch(Hmat, tracks, anchor=sv.Position.BOTTOM_CENTER)
                pitch_xy_ball = project_anchors_to_pitch(Hmat, ball_det, anchor=sv.Position.BOTTOM_CENTER)

            # 6) Write CSV rows
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
                    "homography_degraded": bool(hres.degraded),
                    "fallback_age": int(hres.fallback_age),
                    "kp_used": int(hres.n_points),
                    "inlier_ratio": float(hres.inlier_ratio),
                    "reproj_err": float(hres.reproj_err),
                }
                if class_id == PLAYER_ID:
                    row["team_id"] = team_by_track.get(track_id, -1)
                csvw.write_row(row)

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
                    "homography_degraded": bool(hres.degraded),
                    "fallback_age": int(hres.fallback_age),
                    "kp_used": int(hres.n_points),
                    "inlier_ratio": float(hres.inlier_ratio),
                    "reproj_err": float(hres.reproj_err),
                }
                csvw.write_row(row)

            # 7) Annotate
            annotated = frame.copy()
            annotated = ellipse_annotator.annotate(annotated, tracks)
            if tracks.tracker_id is not None and len(tracks) > 0:
                labels = [f"id:{int(tid)} cls:{int(cid)}" for tid, cid in zip(tracks.tracker_id, tracks.class_id)]
                annotated = label_annotator.annotate(annotated, tracks, labels=labels)
            vw.write(annotated)

            prev_frame = frame

    csvw.close()

    elapsed = time.time() - start_time
    n_frames = video_info.total_frames
    fps = n_frames / max(elapsed, 1e-6)

    print(f"\nDone. Output video: {out_video_path}")
    print(f"CSV: {csv_path}")
    print(f"Elapsed: {elapsed:.2f}s | Frames: {n_frames} | Effective FPS: {fps:.2f}")
    if stage_times:
        print("Last-frame stage latency (ms):")
        for stage, t in stage_times.items():
            print(f"  {stage:>12s}: {t*1000:.1f} ms")


if __name__ == "__main__":
    SOURCE_VIDEO = r"C:\Users\Chimdi\Downloads\analytics\HILAL-AHLI_match_B_up1.mp4"
    main(SOURCE_VIDEO, out_dir="outputs", enable_team=True)
