from __future__ import annotations
import time
import argparse
import queue
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from sports.configs.soccer import SoccerPitchConfiguration

from core.config import load_settings
from vision.models import load_roboflow_models
from vision.detect import (
    infer_players_and_ball_upscaled,
    infer_field_keypoints,
    tiny_box_filter,
    class_conf_filter,
    BALL_ID, PLAYER_ID, REFEREE_ID, GOALKEEPER_ID,
)
from vision.track import create_tracker, TrackManager, empty_detections, merge_detections
from vision.teams import TeamClassifierWrapper
from vision.team_color import ColorTeamClassifier
from vision.preprocess import enhance_frame
from vision.ball import BallSmoother
from vision.team_memory import TeamMemory
from vision.id_stabilizer import IDStabilizer
from geometry.homography import HomographyEstimator
from geometry.hstate import HomographyStateMachine
from geometry.projection import project_anchors_to_pitch
from io_utils.writers import CSVWriter, VideoWriter
from io_utils.minimap import render_side_panel, compose_side_by_side
from io_utils.kpi import write_kpi_summary


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a.tolist()]
    bx1, by1, bx2, by2 = [float(v) for v in b.tolist()]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-6:
        return 0.0
    return float(inter / union)


def main(
    source_video: str,
    out_dir: str = "outputs",
    fit_team_stride: int = 60,
    fit_team_max_frames: int = 30,
    enable_team: bool = True,
):
    s = load_settings()

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    print("[stage] Loading models...")
    # Models
    player_model, field_model = load_roboflow_models(
        api_key=s.ROBOFLOW_API_KEY,
        player_model_id=s.PLAYER_MODEL_ID,
        field_model_id=s.FIELD_MODEL_ID
    )
    print("[stage] Models loaded.")

    # Video
    print(f"[stage] Opening video: {source_video}")
    video_info = sv.VideoInfo.from_video_path(source_video)
    frame_limit = int(s.MAX_FRAMES) if int(s.MAX_FRAMES) > 0 else int(video_info.total_frames)
    progress_total = min(int(video_info.total_frames), frame_limit)
    print(f"[stage] Video ready. total_frames={video_info.total_frames}, processing={progress_total}")

    _prefetch_q: queue.Queue = queue.Queue(maxsize=8)

    def _prefetch_worker():
        for _i, _f in enumerate(sv.get_video_frames_generator(source_video)):
            if _i >= frame_limit:
                break
            _prefetch_q.put(_f)
        _prefetch_q.put(None)

    _prefetch_thread = threading.Thread(target=_prefetch_worker, daemon=True)
    _prefetch_thread.start()

    def _buffered_frames():
        while True:
            _f = _prefetch_q.get()
            if _f is None:
                return
            yield _f

    frames = _buffered_frames()

    # Components
    tracker_players = create_tracker(
        tracker_type=s.TRACKER_TYPE,
        track_activation_threshold=s.TRACK_ACTIVATION_THRESHOLD,
        track_low_threshold=s.TRACK_LOW_THRESHOLD,
        minimum_matching_threshold=s.TRACK_MIN_MATCHING_THRESHOLD,
        lost_track_buffer=s.TRACK_LOST_BUFFER,
        minimum_consecutive_frames=s.TRACK_MIN_CONSEC_FRAMES,
        frame_rate=video_info.fps,
        gmc_method=s.BOTSORT_GMC_METHOD,
        with_reid=s.BOTSORT_WITH_REID,
        reid_model=s.BOTSORT_REID_MODEL,
        proximity_thresh=s.BOTSORT_PROXIMITY_THRESH,
        appearance_thresh=s.BOTSORT_APPEARANCE_THRESH,
    )
    tracker_officials = create_tracker(
        tracker_type=s.TRACKER_TYPE,
        track_activation_threshold=s.TRACK_ACTIVATION_THRESHOLD,
        track_low_threshold=s.TRACK_LOW_THRESHOLD,
        minimum_matching_threshold=s.TRACK_MIN_MATCHING_THRESHOLD,
        lost_track_buffer=s.TRACK_LOST_BUFFER,
        minimum_consecutive_frames=s.TRACK_MIN_CONSEC_FRAMES,
        frame_rate=video_info.fps,
        gmc_method=s.BOTSORT_GMC_METHOD,
        with_reid=False,
        reid_model=s.BOTSORT_REID_MODEL,
        proximity_thresh=s.BOTSORT_PROXIMITY_THRESH,
        appearance_thresh=s.BOTSORT_APPEARANCE_THRESH,
    )
    print(f"[stage] Tracker mode: {str(s.TRACKER_TYPE).lower()}")
    detect_every_n = (s.DETECT_EVERY_N if s.FAST_MODE else 1)
    homography_every_n = (s.HOMOGRAPHY_EVERY_N if s.FAST_MODE else 1)
    team_every_n = (s.TEAM_UPDATE_EVERY_N if s.FAST_MODE else 1)
    track_mgr_players = TrackManager(
        tracker=tracker_players,
        detect_every_n=detect_every_n,
        max_stale_frames=s.TRACK_STALE_FRAMES,
    )
    track_mgr_officials = TrackManager(
        tracker=tracker_officials,
        detect_every_n=detect_every_n,
        max_stale_frames=s.TRACK_STALE_FRAMES,
    )
    pitch_cfg = SoccerPitchConfiguration()
    pitch_vertices_arr = np.asarray(pitch_cfg.vertices, dtype=np.float32)
    pitch_xmin = float(np.min(pitch_vertices_arr[:, 0]))
    pitch_xmax = float(np.max(pitch_vertices_arr[:, 0]))
    pitch_ymin = float(np.min(pitch_vertices_arr[:, 1]))
    pitch_ymax = float(np.max(pitch_vertices_arr[:, 1]))
    # Conservative margins to tolerate slight homography drift.
    on_pitch_margin_x_m = 8.0
    on_pitch_margin_y_m = 5.0
    ball_smoother = BallSmoother(
        ball_id=BALL_ID,
        max_missing=s.BALL_MAX_MISSING,
        max_interp_frames=s.BALL_MAX_INTERP_FRAMES,
        max_jump_px=s.BALL_MAX_JUMP_PX,
        min_conf=s.BALL_MIN_CONF,
    )
    team_memory = TeamMemory(history_size=35, min_votes=8)
    # Split stabilizers by role to prevent cross-class identity interference.
    id_stabilizer_players = IDStabilizer(
        max_relink_frames=s.ID_RELINK_FRAMES,
        memory_frames=s.ID_MEMORY_FRAMES,
        max_relink_px=s.ID_RELINK_PX,
        appearance_weight=s.ID_APP_WEIGHT,
        ambiguity_margin=s.ID_AMBIGUITY_MARGIN,
        max_relink_cost=s.ID_MAX_RELINK_COST,
        unknown_appearance_penalty=s.ID_UNKNOWN_APP_PENALTY,
        unknown_appearance_max_gap=s.ID_UNKNOWN_APP_MAX_GAP,
        unknown_appearance_max_space_frac=s.ID_UNKNOWN_APP_MAX_SPACE_FRAC,
        gallery_size=s.ID_GALLERY_SIZE,
        gallery_min_add_dist=s.ID_GALLERY_MIN_ADD_DIST,
        frame_continuity_iou=s.ID_FRAME_CONTINUITY_IOU,
        frame_continuity_margin=s.ID_FRAME_CONTINUITY_MARGIN,
        spatial_rescue_max_gap=s.ID_SPATIAL_RESCUE_MAX_GAP,
        spatial_rescue_margin=s.ID_SPATIAL_RESCUE_MARGIN,
        spatial_rescue_ratio=s.ID_SPATIAL_RESCUE_RATIO,
        max_player_ids=s.ID_MAX_PLAYER_IDS,
        max_goalkeeper_ids=0,
        max_referee_ids=0,
        hard_reuse_when_capped=s.ID_HARD_REUSE_WHEN_CAPPED,
        lock_frames=s.ID_LOCK_FRAMES,
        capped_reuse_min_gap=s.ID_CAPPED_REUSE_MIN_GAP,
    )
    id_stabilizer_goalkeepers = IDStabilizer(
        max_relink_frames=max(s.ID_RELINK_FRAMES, s.TRACK_LOST_BUFFER),
        memory_frames=max(s.ID_MEMORY_FRAMES, s.TRACK_LOST_BUFFER + 20),
        max_relink_px=s.ID_RELINK_PX * 0.85,
        appearance_weight=min(0.80, s.ID_APP_WEIGHT + 0.10),
        ambiguity_margin=max(s.ID_AMBIGUITY_MARGIN, 0.08),
        max_relink_cost=s.ID_MAX_RELINK_COST,
        unknown_appearance_penalty=min(0.75, s.ID_UNKNOWN_APP_PENALTY + 0.10),
        unknown_appearance_max_gap=s.ID_UNKNOWN_APP_MAX_GAP + 8,
        unknown_appearance_max_space_frac=s.ID_UNKNOWN_APP_MAX_SPACE_FRAC,
        gallery_size=s.ID_GALLERY_SIZE,
        gallery_min_add_dist=s.ID_GALLERY_MIN_ADD_DIST,
        frame_continuity_iou=max(0.50, s.ID_FRAME_CONTINUITY_IOU),
        frame_continuity_margin=s.ID_FRAME_CONTINUITY_MARGIN,
        spatial_rescue_max_gap=max(s.ID_SPATIAL_RESCUE_MAX_GAP, s.TRACK_LOST_BUFFER + 30),
        spatial_rescue_margin=s.ID_SPATIAL_RESCUE_MARGIN,
        spatial_rescue_ratio=s.ID_SPATIAL_RESCUE_RATIO,
        max_player_ids=0,
        max_goalkeeper_ids=s.ID_MAX_GOALKEEPER_IDS,
        max_referee_ids=0,
        hard_reuse_when_capped=s.ID_HARD_REUSE_WHEN_CAPPED,
        lock_frames=max(s.ID_LOCK_FRAMES, 16),
        capped_reuse_min_gap=s.ID_CAPPED_REUSE_MIN_GAP,
    )
    id_stabilizer_referees = IDStabilizer(
        max_relink_frames=max(s.ID_RELINK_FRAMES, s.TRACK_LOST_BUFFER),
        memory_frames=max(s.ID_MEMORY_FRAMES, s.TRACK_LOST_BUFFER + 10),
        max_relink_px=s.ID_RELINK_PX * 0.95,
        appearance_weight=min(0.75, s.ID_APP_WEIGHT + 0.05),
        ambiguity_margin=max(s.ID_AMBIGUITY_MARGIN, 0.08),
        max_relink_cost=s.ID_MAX_RELINK_COST,
        unknown_appearance_penalty=s.ID_UNKNOWN_APP_PENALTY,
        unknown_appearance_max_gap=s.ID_UNKNOWN_APP_MAX_GAP + 6,
        unknown_appearance_max_space_frac=s.ID_UNKNOWN_APP_MAX_SPACE_FRAC,
        gallery_size=s.ID_GALLERY_SIZE,
        gallery_min_add_dist=s.ID_GALLERY_MIN_ADD_DIST,
        frame_continuity_iou=s.ID_FRAME_CONTINUITY_IOU,
        frame_continuity_margin=s.ID_FRAME_CONTINUITY_MARGIN,
        spatial_rescue_max_gap=max(s.ID_SPATIAL_RESCUE_MAX_GAP, s.TRACK_LOST_BUFFER + 20),
        spatial_rescue_margin=s.ID_SPATIAL_RESCUE_MARGIN,
        spatial_rescue_ratio=s.ID_SPATIAL_RESCUE_RATIO,
        max_player_ids=0,
        max_goalkeeper_ids=0,
        max_referee_ids=s.ID_MAX_REFEREE_IDS,
        hard_reuse_when_capped=s.ID_HARD_REUSE_WHEN_CAPPED,
        lock_frames=max(s.ID_LOCK_FRAMES, 14),
        capped_reuse_min_gap=s.ID_CAPPED_REUSE_MIN_GAP,
    )
    display_caps = {
        PLAYER_ID: max(0, int(s.ID_MAX_PLAYER_IDS)),
        REFEREE_ID: max(0, int(s.ID_MAX_REFEREE_IDS)),
        GOALKEEPER_ID: max(0, int(s.ID_MAX_GOALKEEPER_IDS)),
    }
    display_sid_to_slot: dict[int, dict[int, int]] = {cid: {} for cid in display_caps}
    display_slot_to_sid: dict[int, dict[int, int]] = {cid: {} for cid in display_caps}
    stable_last_seen: dict[int, int] = {}
    official_role_memory: dict[int, tuple[int, np.ndarray, int, int]] = {}
    official_classes = {REFEREE_ID, GOALKEEPER_ID}
    official_lock_horizon = max(int(s.TRACK_LOST_BUFFER), 600)
    official_lock_min_iou = 0.20
    official_lock_confirm_frames = 1
    overlap_suppress_iou = 0.72
    official_quota = {
        REFEREE_ID: max(0, int(s.ID_MAX_REFEREE_IDS)),
        GOALKEEPER_ID: max(0, int(s.ID_MAX_GOALKEEPER_IDS)),
    }

    def to_display_id(stable_id: int, class_id: int, frame_i: int) -> int:
        if stable_id < 0:
            return -1
        sid = int(stable_id)
        cid = int(class_id)
        stable_last_seen[sid] = int(frame_i)
        cap = int(display_caps.get(cid, 0))
        if cap <= 0:
            return sid

        sid_map = display_sid_to_slot[cid]
        slot_map = display_slot_to_sid[cid]
        existing = sid_map.get(sid)
        if existing is not None:
            slot_map[int(existing)] = sid
            return int(existing)

        free_slot: int | None = None
        for slot in range(1, cap + 1):
            if slot not in slot_map:
                free_slot = slot
                break

        if free_slot is None:
            free_slot = min(
                slot_map.keys(),
                key=lambda slot: stable_last_seen.get(int(slot_map[slot]), -10**9),
            )
            old_sid = int(slot_map[free_slot])
            sid_map.pop(old_sid, None)

        sid_map[sid] = int(free_slot)
        slot_map[int(free_slot)] = sid
        return int(free_slot)

    h_est = HomographyEstimator(
        config=pitch_cfg,
        kp_conf=s.KP_CONF,
        ema_alpha=s.H_EMA_ALPHA,
        ransac_reproj_thresh=s.RANSAC_REPROJ_THRESH,
        min_kp=s.MIN_KP,
        min_inlier_ratio=s.MIN_INLIER_RATIO,
        max_reproj_err=s.MAX_REPROJ_ERR,
    )
    h_state = HomographyStateMachine(
        estimator=h_est,
        hold_max_frames=s.H_HOLD_MAX_FRAMES,
        reinit_frames=s.H_REINIT_FRAMES,
    )

    team_clf = None
    team_color_clf = None
    team_by_track: dict[int, int] = {}
    if enable_team:
        if str(s.TEAM_MODE).lower() == "embedding":
            try:
                print("[stage] Initializing embedding team classifier...")
                team_clf = TeamClassifierWrapper(device=s.DEVICE)
                effective_stride = max(60, fit_team_stride) if s.FAST_MODE else fit_team_stride
                effective_max = min(24, fit_team_max_frames) if s.FAST_MODE else fit_team_max_frames
                fit_gen = sv.get_video_frames_generator(source_video, stride=effective_stride)
                team_clf.fit_from_video_frames(
                    frame_iter=fit_gen,
                    player_model=player_model,
                    det_conf=s.DET_CONF,
                    player_class_id=PLAYER_ID,
                    max_frames=effective_max,
                )
                print(f"Team classifier fit complete (stride={effective_stride}, frames={effective_max}).")
            except Exception as e:
                print(f"[WARN] Embedding team classifier disabled: {e}")
                team_clf = None
        else:
            team_color_clf = ColorTeamClassifier(
                init_samples=s.TEAM_COLOR_INIT_SAMPLES,
                lr=s.TEAM_COLOR_LR,
                min_margin=s.TEAM_COLOR_MIN_MARGIN,
            )
            print(
                f"[stage] Color team classifier enabled "
                f"(init_samples={s.TEAM_COLOR_INIT_SAMPLES}, margin={s.TEAM_COLOR_MIN_MARGIN})."
            )
    else:
        print("[stage] Team classifier disabled (enable_team=False).")

    _pool = ThreadPoolExecutor(max_workers=2)

    # Annotators
    ellipse_annotator = sv.EllipseAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_CENTER)

    csv_path = str(out_dir_p / "per_frame_tracks.csv")
    out_video_path = str(out_dir_p / "annotated.mp4")
    csvw = CSVWriter(
        csv_path,
        fieldnames=[
            "frame", "track_id", "display_track_id", "class_id", "conf",
            "x1", "y1", "x2", "y2",
            "x_m", "y_m",
            "team_id",
            "homography_ok", "kp_used", "inlier_ratio", "reproj_err",
            "detector_ran", "homography_state", "ball_interpolated",
        ],
    )

    last_hmat = None
    homography_ok = False
    homography_state = "none"
    homography_ok_frames = 0
    homography_available_frames = 0
    player_total = 0
    ball_detect_frames = 0
    ball_interp_frames = 0
    valid_projection_rows = 0
    total_rows = 0
    kp_used = 0
    inlier_ratio = 0.0
    reproj_err = 1e9
    track_len = {}
    track_class: dict[int, int] = {}
    gk_demotions_pre_track = 0

    start_time = time.time()
    processed_frames = 0
    prev_player_boxes: dict[int, np.ndarray] = {}
    pitch_smooth: dict[int, tuple[float, float, int]] = {}
    PITCH_SMOOTH_ALPHA = 0.30
    PITCH_SMOOTH_RESET_GAP = 60
    GK_GOAL_ZONE_X_M = 1500.0  # cm; 15m from each goal line
    print("[stage] Starting frame loop...")
    with VideoWriter(out_video_path, video_info) as vw:
        for frame_idx, frame in tqdm(enumerate(frames), total=progress_total):
            if frame_idx >= frame_limit:
                break

            processed_frames += 1
            h, w = frame.shape[:2]
            frame_infer = enhance_frame(frame, enabled=bool(s.PREPROCESS_ENABLED))

            # 1) Submit both model inferences in parallel, then collect detection results
            should_detect = track_mgr_players.should_detect(frame_idx) or track_mgr_officials.should_detect(frame_idx)
            should_update_h = (frame_idx % max(1, homography_every_n) == 0) or (last_hmat is None) or (homography_state == "none")

            _fut_det = _pool.submit(infer_players_and_ball_upscaled, player_model, frame_infer, s.DET_CONF, s.DETECT_UPSCALE) if should_detect else None
            _fut_kp = _pool.submit(infer_field_keypoints, field_model, frame_infer, s.FIELD_CONF) if should_update_h else None

            players_det = None
            officials_det = None
            raw_ball_det = empty_detections()
            if _fut_det is not None:
                det = _fut_det.result()

                det = class_conf_filter(
                    det,
                    player_conf=s.DET_CONF_PLAYER,
                    referee_conf=s.DET_CONF_REFEREE,
                    goalkeeper_conf=s.DET_CONF_GOALKEEPER,
                    ball_conf=s.DET_CONF_BALL,
                )

                det = tiny_box_filter(
                    det, w, h,
                    min_area_ratio_people=s.MIN_AREA_RATIO_PEOPLE,
                    min_area_ratio_ball=s.MIN_AREA_RATIO_BALL,
                )

                # Pre-tracking GK demotion: if previous-frame H is available, project
                # GK detections to pitch coords and demote any that are far from either
                # goal zone BEFORE they enter the tracker. This prevents misclassified
                # GKs from acquiring a GK-pool stable ID, which the post-hoc demotion
                # at step 5c could not undo (causing cross-pool ID flicker).
                if last_hmat is not None and det.class_id is not None and len(det) > 0:
                    gk_det_mask = det.class_id == GOALKEEPER_ID
                    if np.any(gk_det_mask):
                        try:
                            gk_pitch_xy = project_anchors_to_pitch(
                                last_hmat, det[gk_det_mask], anchor=sv.Position.BOTTOM_CENTER
                            )
                        except Exception:
                            gk_pitch_xy = None
                        if gk_pitch_xy is not None:
                            gk_indices = np.where(gk_det_mask)[0]
                            new_cls = det.class_id.copy()
                            for j, idx in enumerate(gk_indices):
                                xm = float(gk_pitch_xy[j, 0])
                                if not np.isfinite(xm):
                                    continue
                                near_left = abs(xm - pitch_xmin) <= GK_GOAL_ZONE_X_M
                                near_right = abs(pitch_xmax - xm) <= GK_GOAL_ZONE_X_M
                                if not (near_left or near_right):
                                    new_cls[int(idx)] = PLAYER_ID
                                    gk_demotions_pre_track += 1
                            det.class_id = new_cls

                raw_ball_det = det[det.class_id == BALL_ID]
                if len(raw_ball_det) > 0 and s.BALL_PAD_PX > 0:
                    raw_ball_det.xyxy = sv.pad_boxes(raw_ball_det.xyxy, px=s.BALL_PAD_PX)

                players_det = det[det.class_id == PLAYER_ID]
                official_mask = np.isin(det.class_id, np.array([REFEREE_ID, GOALKEEPER_ID], dtype=np.int32))
                officials_det = det[official_mask]

            # 2) Track players and officials separately, then merge
            players_tracks, detector_ran_p = track_mgr_players.update(frame_idx, players_det, frame=frame_infer)
            officials_tracks, detector_ran_o = track_mgr_officials.update(frame_idx, officials_det, frame=frame_infer)
            tracks = merge_detections([players_tracks, officials_tracks])
            detector_ran = bool(detector_ran_p or detector_ran_o)

            ball_det, ball_imputed = ball_smoother.update(raw_ball_det)
            stable_track_ids = np.full((len(tracks),), -1, dtype=np.int32)
            if len(tracks) > 0 and tracks.class_id is not None:
                player_mask = tracks.class_id == PLAYER_ID
                goalkeeper_mask = tracks.class_id == GOALKEEPER_ID
                referee_mask = tracks.class_id == REFEREE_ID

                if np.any(player_mask):
                    player_tracks = tracks[player_mask]
                    det_team_ids_players = np.full((len(player_tracks),), -1, dtype=np.int32)
                    if team_color_clf is not None and team_color_clf.is_ready:
                        for ii in range(len(player_tracks)):
                            crop = team_color_clf._torso_crop(frame_infer, player_tracks.xyxy[ii])
                            if crop is None:
                                continue
                            feat = team_color_clf._feature_from_crop(crop)
                            if feat is None:
                                continue
                            tt, mm = team_color_clf._assign(feat)
                            if mm >= team_color_clf.min_margin:
                                det_team_ids_players[ii] = int(tt)
                    player_local_ids = id_stabilizer_players.update(
                        frame_idx, frame_infer, player_tracks, det_team_ids=det_team_ids_players
                    )
                    stable_track_ids[player_mask] = player_local_ids.astype(np.int32)
                if np.any(goalkeeper_mask):
                    gk_local_ids = id_stabilizer_goalkeepers.update(frame_idx, frame_infer, tracks[goalkeeper_mask])
                    # Keep global uniqueness across classes while preserving local pool for display.
                    stable_track_ids[goalkeeper_mask] = (1000 + gk_local_ids).astype(np.int32)
                if np.any(referee_mask):
                    ref_local_ids = id_stabilizer_referees.update(frame_idx, frame_infer, tracks[referee_mask])
                    stable_track_ids[referee_mask] = (2000 + ref_local_ids).astype(np.int32)
            if len(tracks) > 0 and tracks.class_id is not None:
                class_ids = tracks.class_id.astype(np.int32).copy()
                confs = (
                    tracks.confidence.astype(np.float32).copy()
                    if tracks.confidence is not None
                    else np.zeros((len(tracks),), dtype=np.float32)
                )

                # Keep currently-present official IDs reserved, then allow player->official rescue.
                claimed_official_sids: set[int] = {
                    int(stable_track_ids[i])
                    for i in range(len(tracks))
                    if int(class_ids[i]) in official_classes and int(stable_track_ids[i]) >= 0
                }
                for i in range(len(tracks)):
                    if int(class_ids[i]) != PLAYER_ID:
                        continue
                    best_sid = None
                    best_class = None
                    best_score = -1e9
                    for sid, (mem_class, mem_box, mem_frame, mem_streak) in official_role_memory.items():
                        gap = int(frame_idx - mem_frame)
                        if gap <= 0 or gap > official_lock_horizon:
                            continue
                        # Require stronger confirmation for long-gap rescue; allow short-gap continuity.
                        if int(mem_streak) < official_lock_confirm_frames and gap > 2:
                            continue
                        if sid in claimed_official_sids:
                            continue
                        iou = _bbox_iou(tracks.xyxy[i], mem_box)
                        if iou < official_lock_min_iou:
                            continue
                        # Small gap penalty keeps lock conservative over longer gaps.
                        score = float(iou) - 0.01 * float(gap)
                        if score > best_score:
                            best_score = score
                            best_sid = int(sid)
                            best_class = int(mem_class)
                    if best_sid is None or best_class is None:
                        continue
                    class_ids[i] = int(best_class)
                    stable_track_ids[i] = int(best_sid)
                    claimed_official_sids.add(int(best_sid))

                keep_mask = np.ones((len(tracks),), dtype=bool)
                gate_xy = np.full((len(tracks), 2), np.nan, dtype=np.float32)

                # Drop people projected far outside pitch bounds (sideline/staff leakage).
                if last_hmat is not None:
                    try:
                        gate_xy = project_anchors_to_pitch(last_hmat, tracks, anchor=sv.Position.BOTTOM_CENTER)
                    except Exception:
                        gate_xy = np.full((len(tracks), 2), np.nan, dtype=np.float32)
                    person_idx = np.where(
                        np.isin(class_ids, np.array([PLAYER_ID, GOALKEEPER_ID, REFEREE_ID], dtype=np.int32))
                    )[0].tolist()
                    for i in person_idx:
                        gx = float(gate_xy[i, 0])
                        gy = float(gate_xy[i, 1])
                        if not (np.isfinite(gx) and np.isfinite(gy)):
                            continue
                        if (
                            gx < (pitch_xmin - on_pitch_margin_x_m)
                            or gx > (pitch_xmax + on_pitch_margin_x_m)
                            or gy < (pitch_ymin - on_pitch_margin_y_m)
                            or gy > (pitch_ymax + on_pitch_margin_y_m)
                        ):
                            keep_mask[i] = False

                # Suppress overlapping player/official duplicates for the same person.
                player_idx = np.where(class_ids == PLAYER_ID)[0].tolist()
                official_idx = [i for i in np.where(np.isin(class_ids, np.array(list(official_classes), dtype=np.int32)))[0].tolist()]
                for oi in official_idx:
                    if not keep_mask[oi]:
                        continue
                    for pi in player_idx:
                        if not keep_mask[pi]:
                            continue
                        ov = _bbox_iou(tracks.xyxy[oi], tracks.xyxy[pi])
                        if ov < overlap_suppress_iou:
                            continue
                        sid_o = int(stable_track_ids[oi])
                        mem = official_role_memory.get(sid_o)
                        mem_bonus = 0.0
                        if (
                            mem is not None
                            and int(mem[0]) == int(class_ids[oi])
                            and (int(frame_idx) - int(mem[2])) <= official_lock_horizon
                            and int(mem[3]) >= official_lock_confirm_frames
                        ):
                            mem_bonus = 0.25
                        off_score = float(confs[oi]) + mem_bonus
                        ply_score = float(confs[pi])
                        if off_score >= (ply_score + 0.02):
                            keep_mask[pi] = False
                        else:
                            keep_mask[oi] = False
                            break

                # Hard quotas for official roles: at most 2 referees and 2 goalkeepers.
                for cid, cap in official_quota.items():
                    if cap <= 0:
                        continue
                    idx = [i for i in np.where((class_ids == int(cid)) & keep_mask)[0].tolist()]
                    if len(idx) <= cap:
                        continue
                    scored: list[tuple[float, int]] = []
                    for i in idx:
                        sid_i = int(stable_track_ids[i])
                        score = float(confs[i])
                        mem = official_role_memory.get(sid_i)
                        if (
                            mem is not None
                            and int(mem[0]) == int(cid)
                            and (int(frame_idx) - int(mem[2])) <= official_lock_horizon
                        ):
                            score += 0.35 + 0.07 * min(int(mem[3]), 5)
                        scored.append((score, i))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    for _, drop_i in scored[cap:]:
                        keep_mask[drop_i] = False

                # If multiple GKs are on the same side of the screen, keep only the strongest one.
                # Legitimate two-GK views should be far apart near opposite goals.
                gk_idx = [i for i in np.where((class_ids == GOALKEEPER_ID) & keep_mask)[0].tolist()]
                if len(gk_idx) > 1:
                    centers_x = [0.5 * float(tracks.xyxy[i][0] + tracks.xyxy[i][2]) for i in gk_idx]
                    spread = max(centers_x) - min(centers_x)
                    if spread < (0.45 * float(w)):
                        gk_scored: list[tuple[float, int]] = []
                        for i in gk_idx:
                            sid_i = int(stable_track_ids[i])
                            score = float(confs[i])
                            mem = official_role_memory.get(sid_i)
                            if (
                                mem is not None
                                and int(mem[0]) == GOALKEEPER_ID
                                and (int(frame_idx) - int(mem[2])) <= official_lock_horizon
                            ):
                                score += 0.35 + 0.07 * min(int(mem[3]), 5)
                            gk_scored.append((score, i))
                        gk_scored.sort(key=lambda x: x[0], reverse=True)
                        for _, drop_i in gk_scored[1:]:
                            keep_mask[drop_i] = False

                # Hard cap players to on-field count and prefer temporally consistent tracks.
                player_cap = max(0, int(s.ID_MAX_PLAYER_IDS))
                if player_cap > 0:
                    pidx = [i for i in np.where((class_ids == PLAYER_ID) & keep_mask)[0].tolist()]
                    if len(pidx) > player_cap:
                        p_scored: list[tuple[float, int]] = []
                        for i in pidx:
                            sid_i = int(stable_track_ids[i])
                            score = float(confs[i])
                            prev_box = prev_player_boxes.get(sid_i)
                            if prev_box is not None:
                                score += 0.25 * _bbox_iou(tracks.xyxy[i], prev_box)
                            gx = float(gate_xy[i, 0])
                            gy = float(gate_xy[i, 1])
                            if np.isfinite(gx) and np.isfinite(gy):
                                if pitch_xmin <= gx <= pitch_xmax and pitch_ymin <= gy <= pitch_ymax:
                                    score += 0.15
                            p_scored.append((score, i))
                        p_scored.sort(key=lambda x: x[0], reverse=True)
                        for _, drop_i in p_scored[player_cap:]:
                            keep_mask[drop_i] = False

                # Enforce one-to-one (class_id, stable_id) within a frame.
                sid_buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
                for i in np.where(keep_mask)[0].tolist():
                    sid_i = int(stable_track_ids[i])
                    if sid_i < 0:
                        continue
                    sid_buckets[(int(class_ids[i]), sid_i)].append(i)
                for _, idx in sid_buckets.items():
                    if len(idx) <= 1:
                        continue
                    idx_sorted = sorted(idx, key=lambda ii: float(confs[ii]), reverse=True)
                    for drop_i in idx_sorted[1:]:
                        keep_mask[drop_i] = False

                if not np.all(keep_mask):
                    tracks = tracks[keep_mask]
                    stable_track_ids = stable_track_ids[keep_mask]
                    class_ids = class_ids[keep_mask]
                    confs = confs[keep_mask]
                    gate_xy = gate_xy[keep_mask]

                tracks.class_id = class_ids

                prev_player_boxes = {
                    int(stable_track_ids[i]): tracks.xyxy[i].copy()
                    for i in np.where(tracks.class_id == PLAYER_ID)[0].tolist()
                    if int(stable_track_ids[i]) >= 0
                }

                refreshed_memory: dict[int, tuple[int, np.ndarray, int, int]] = {}
                for i in range(len(tracks)):
                    sid = int(stable_track_ids[i])
                    cid = int(tracks.class_id[i])
                    if sid < 0 or cid not in official_classes:
                        continue
                    prev = official_role_memory.get(sid)
                    if prev is not None and int(prev[0]) == cid and (int(frame_idx) - int(prev[2])) <= 2:
                        streak = min(12, int(prev[3]) + 1)
                    else:
                        streak = 1
                    refreshed_memory[sid] = (cid, tracks.xyxy[i].copy(), int(frame_idx), int(streak))

                for sid, mem in official_role_memory.items():
                    if sid in refreshed_memory:
                        continue
                    cid, box, seen_frame, streak = mem
                    if (int(frame_idx) - int(seen_frame)) <= official_lock_horizon:
                        refreshed_memory[sid] = (int(cid), box, int(seen_frame), max(1, int(streak) - 1))

                official_role_memory = refreshed_memory
            display_track_ids = np.full((len(stable_track_ids),), -1, dtype=np.int32)
            for i, sid in enumerate(stable_track_ids):
                if i >= len(tracks.class_id):
                    break
                display_track_ids[i] = to_display_id(
                    stable_id=int(sid),
                    class_id=int(tracks.class_id[i]),
                    frame_i=frame_idx,
                )

            # 3) Team classification with track-memory voting
            if team_clf is not None or team_color_clf is not None:
                players_tr = tracks[tracks.class_id == PLAYER_ID]
                player_mask = tracks.class_id == PLAYER_ID
                stable_player_ids = stable_track_ids[player_mask] if len(tracks) > 0 else np.array([], dtype=np.int32)
                team_update_frame = (frame_idx % max(1, team_every_n) == 0)
                if team_update_frame and detector_ran and len(players_tr) > 0:
                    if team_color_clf is not None:
                        pred_map = team_color_clf.update_players(frame_infer, players_tr, stable_player_ids)
                        for tid, t in pred_map.items():
                            team_memory.update(int(tid), int(t))
                    elif team_clf is not None:
                        team_ids = team_clf.predict_team_ids(frame_infer, players_tr)
                        if team_ids is not None and len(stable_player_ids) == len(team_ids):
                            for tid, t in zip(stable_player_ids, team_ids):
                                team_memory.update(int(tid), int(t))

                if len(stable_player_ids) > 0:
                    for tid in stable_player_ids:
                        team_by_track[int(tid)] = team_memory.get(int(tid))

            # 4) Field keypoints -> Homography state machine (result already computed in parallel)
            if _fut_kp is not None:
                kp = _fut_kp.result()
                Hmat, homography_ok, hres = h_state.update(kp)
                last_hmat = Hmat
                homography_state = "ok" if homography_ok else ("hold" if Hmat is not None else "none")
                homography_ok_frames += int(homography_ok)
                kp_used = int(hres.n_points)
                inlier_ratio = float(hres.inlier_ratio)
                reproj_err = float(hres.reproj_err)
            else:
                homography_ok = False
                if last_hmat is None:
                    homography_state = "none"
                elif homography_state != "ok":
                    homography_state = "hold"

            homography_available_frames += int(last_hmat is not None)

            # 5) Project to pitch coords (if H available)
            pitch_xy_tracks = np.full((len(tracks), 2), np.nan, dtype=np.float32)
            pitch_xy_ball = np.full((len(ball_det), 2), np.nan, dtype=np.float32)

            if last_hmat is not None:
                pitch_xy_tracks = project_anchors_to_pitch(last_hmat, tracks, anchor=sv.Position.BOTTOM_CENTER)
                pitch_xy_ball = project_anchors_to_pitch(last_hmat, ball_det, anchor=sv.Position.BOTTOM_CENTER)

            # 5b) Per-track pitch-coord EMA smoothing (reduces minimap dot jitter).
            for i in range(len(tracks)):
                sid = int(stable_track_ids[i]) if i < len(stable_track_ids) else -1
                if sid < 0:
                    continue
                if not (np.isfinite(pitch_xy_tracks[i, 0]) and np.isfinite(pitch_xy_tracks[i, 1])):
                    continue
                obs_x = float(pitch_xy_tracks[i, 0])
                obs_y = float(pitch_xy_tracks[i, 1])
                prev = pitch_smooth.get(sid)
                if prev is None or (frame_idx - prev[2]) > PITCH_SMOOTH_RESET_GAP:
                    pitch_smooth[sid] = (obs_x, obs_y, int(frame_idx))
                else:
                    sx = (1 - PITCH_SMOOTH_ALPHA) * prev[0] + PITCH_SMOOTH_ALPHA * obs_x
                    sy = (1 - PITCH_SMOOTH_ALPHA) * prev[1] + PITCH_SMOOTH_ALPHA * obs_y
                    pitch_smooth[sid] = (sx, sy, int(frame_idx))
                    pitch_xy_tracks[i, 0] = sx
                    pitch_xy_tracks[i, 1] = sy

            # 5c) Goalkeeper spatial constraint: demote GK detections that are far from any goal line.
            #     Conservative — never promotes random players, only demotes wrong GK labels.
            if last_hmat is not None and len(tracks) > 0 and tracks.class_id is not None:
                cls_arr = tracks.class_id.astype(np.int32)
                for i in range(len(tracks)):
                    if int(cls_arr[i]) != GOALKEEPER_ID:
                        continue
                    xm = float(pitch_xy_tracks[i, 0])
                    if not np.isfinite(xm):
                        continue
                    near_left = abs(xm - pitch_xmin) <= GK_GOAL_ZONE_X_M
                    near_right = abs(pitch_xmax - xm) <= GK_GOAL_ZONE_X_M
                    if not (near_left or near_right):
                        cls_arr[i] = PLAYER_ID
                tracks.class_id = cls_arr

            # 6) Write CSV rows
            player_total += int(np.sum(tracks.class_id == PLAYER_ID)) if len(tracks) > 0 else 0
            if len(ball_det) > 0:
                if ball_imputed:
                    ball_interp_frames += 1
                else:
                    ball_detect_frames += 1
            for i in range(len(tracks)):
                x1, y1, x2, y2 = tracks.xyxy[i].tolist()
                track_id = int(stable_track_ids[i]) if i < len(stable_track_ids) else -1
                class_id = int(tracks.class_id[i])
                row_team_id = team_by_track.get(track_id, -1) if class_id == PLAYER_ID else -1
                track_len[track_id] = track_len.get(track_id, 0) + 1
                if track_id >= 0:
                    track_class[track_id] = class_id

                row = {
                    "frame": frame_idx,
                    "track_id": track_id,
                    "display_track_id": int(display_track_ids[i]) if i < len(display_track_ids) else -1,
                    "class_id": class_id,
                    "conf": float(tracks.confidence[i]) if tracks.confidence is not None else 0.0,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "x_m": float(pitch_xy_tracks[i, 0]),
                    "y_m": float(pitch_xy_tracks[i, 1]),
                    "team_id": row_team_id,
                    "homography_ok": homography_ok,
                    "kp_used": kp_used,
                    "inlier_ratio": inlier_ratio,
                    "reproj_err": reproj_err,
                    "detector_ran": int(detector_ran),
                    "homography_state": homography_state,
                    "ball_interpolated": 0,
                }
                csvw.write_row(row)
                total_rows += 1
                if np.isfinite(pitch_xy_tracks[i, 0]) and np.isfinite(pitch_xy_tracks[i, 1]):
                    valid_projection_rows += 1

            for j in range(len(ball_det)):
                x1, y1, x2, y2 = ball_det.xyxy[j].tolist()
                row = {
                    "frame": frame_idx,
                    "track_id": -1,
                    "display_track_id": -1,
                    "class_id": int(ball_det.class_id[j]),
                    "conf": float(ball_det.confidence[j]) if ball_det.confidence is not None else 0.0,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "x_m": float(pitch_xy_ball[j, 0]),
                    "y_m": float(pitch_xy_ball[j, 1]),
                    "team_id": -1,
                    "homography_ok": homography_ok,
                    "kp_used": kp_used,
                    "inlier_ratio": inlier_ratio,
                    "reproj_err": reproj_err,
                    "detector_ran": int(detector_ran),
                    "homography_state": homography_state,
                    "ball_interpolated": int(ball_imputed),
                }
                csvw.write_row(row)
                total_rows += 1
                if np.isfinite(pitch_xy_ball[j, 0]) and np.isfinite(pitch_xy_ball[j, 1]):
                    valid_projection_rows += 1

            # 7) Draw main view
            annotated = frame.copy()
            annotated = ellipse_annotator.annotate(annotated, tracks)

            if tracks.tracker_id is not None and len(tracks) > 0:
                labels = []
                for sid, did, cid in zip(stable_track_ids, display_track_ids, tracks.class_id):
                    show_id = int(did) if int(did) >= 0 else int(sid)
                    if int(cid) == PLAYER_ID:
                        team_id = team_by_track.get(int(sid), -1)
                        if int(team_id) >= 0:
                            labels.append(f"P{show_id}-{int(team_id)}")
                        else:
                            labels.append(f"P{show_id}")
                    elif int(cid) == REFEREE_ID:
                        labels.append(f"R{show_id}")
                    elif int(cid) == GOALKEEPER_ID:
                        labels.append(f"GK{show_id}")
                    else:
                        labels.append(f"#{int(sid)}")
                annotated = label_annotator.annotate(annotated, tracks, labels=labels)

            for j in range(len(ball_det)):
                x1, y1, x2, y2 = [int(v) for v in ball_det.xyxy[j].tolist()]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 210, 255), 2)
                cv2.putText(annotated, "ball", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 255), 1, cv2.LINE_AA)

            # 8) Side panel rendering
            track_ids_arr = stable_track_ids if len(stable_track_ids) == len(tracks) else np.full((len(tracks),), -1, dtype=np.int32)
            panel = render_side_panel(
                frame=annotated,
                pitch_vertices=list(pitch_cfg.vertices),
                pitch_xy_tracks=pitch_xy_tracks,
                track_class_ids=tracks.class_id.astype(np.int32) if tracks.class_id is not None else np.zeros((len(tracks),), dtype=np.int32),
                track_ids=track_ids_arr.astype(np.int32),
                team_by_track=team_by_track,
                pitch_xy_ball=pitch_xy_ball,
                homography_ok=homography_ok,
            )
            if s.SIDE_BY_SIDE_VIEW:
                annotated = compose_side_by_side(annotated, panel, left_ratio=s.LEFT_VIEW_RATIO)
            else:
                panel_w = min(panel.shape[1], int(w * 0.38))
                panel_resized = cv2.resize(panel, (panel_w, h), interpolation=cv2.INTER_AREA)
                x0 = w - panel_w
                annotated[:, x0:w] = cv2.addWeighted(annotated[:, x0:w], 0.25, panel_resized, 0.75, 0.0)

            vw.write(annotated)

    csvw.close()
    _pool.shutdown(wait=False)

    elapsed = time.time() - start_time
    fps = processed_frames / max(elapsed, 1e-6)
    print(f"Done. Output video: {out_video_path}")
    print(f"CSV: {csv_path}")
    print(f"Processed frames: {processed_frames}")
    print(f"Avg tracked players/frame: {player_total / max(processed_frames, 1):.2f}")
    print(f"Ball detected frames: {ball_detect_frames}/{processed_frames}")
    print(f"Ball interpolated frames: {ball_interp_frames}/{processed_frames}")
    print(f"Homography OK frames: {homography_ok_frames}/{processed_frames}")
    print(f"Homography available frames: {homography_available_frames}/{processed_frames}")
    h_est.report_failure_summary()
    print(f"Valid projection rows: {valid_projection_rows}/{max(total_rows, 1)}")
    print(f"Elapsed: {elapsed:.2f}s | Effective FPS: {fps:.2f}")

    unique_player_ids = sum(1 for cid in track_class.values() if cid == PLAYER_ID)
    unique_gk_ids = sum(1 for cid in track_class.values() if cid == GOALKEEPER_ID)
    unique_ref_ids = sum(1 for cid in track_class.values() if cid == REFEREE_ID)
    team_a = sum(1 for t in team_by_track.values() if int(t) == 0)
    team_b = sum(1 for t in team_by_track.values() if int(t) == 1)
    team_unknown = sum(1 for t in team_by_track.values() if int(t) < 0)
    print(
        f"Unique IDs: players={unique_player_ids}, goalkeepers={unique_gk_ids}, "
        f"referees={unique_ref_ids}"
    )
    print(f"Team distribution: A={team_a}, B={team_b}, unknown={team_unknown}")
    print(f"Pre-tracking GK demotions: {gk_demotions_pre_track}")

    short_tracks = sum(1 for _, n in track_len.items() if n <= 5)
    metrics = {
        "source_video": source_video,
        "processed_frames": processed_frames,
        "effective_fps": fps,
        "avg_players_per_frame": player_total / max(processed_frames, 1),
        "ball_detect_coverage": ball_detect_frames / max(processed_frames, 1),
        "ball_interp_coverage": ball_interp_frames / max(processed_frames, 1),
        "homography_ok_rate": homography_ok_frames / max(processed_frames, 1),
        "homography_available_rate": homography_available_frames / max(processed_frames, 1),
        "valid_projection_ratio": valid_projection_rows / max(total_rows, 1),
        "unique_tracks": len(track_len),
        "short_track_ratio": short_tracks / max(len(track_len), 1),
        "team_unknown_ratio": (
            sum(1 for t in team_by_track.values() if int(t) < 0) / max(len(team_by_track), 1)
            if enable_team else 1.0
        ),
    }
    kpi_json, kpi_csv = write_kpi_summary(str(out_dir_p), metrics)
    print(f"KPI JSON: {kpi_json}")
    print(f"KPI CSV: {kpi_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Soccer analytics pipeline")
    parser.add_argument("--source-video", required=True, help="Path to input mp4")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    parser.add_argument("--enable-team", action="store_true", help="Enable team classifier")
    args = parser.parse_args()

    main(args.source_video, out_dir=args.out_dir, enable_team=args.enable_team)
