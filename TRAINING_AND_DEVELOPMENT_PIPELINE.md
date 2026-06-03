# Model Training and Development Pipeline

This project currently separates model training from runtime inference.

The detector/keypoint models were trained and hosted in Roboflow. This repository
contains the development pipeline that consumes those hosted models, runs the
football analytics workflow, and produces videos/CSVs/KPI summaries for
evaluation.

## Models Used

The model IDs are configured through `.env` and read by `core/config.py`.

Current Roboflow model slots:

```dotenv
PLAYER_MODEL_ID=footballs-player-detection-zkams-zia6c/2
FIELD_MODEL_ID=football-field-detection-f07vi/15
```

The player model is used for:

- players
- goalkeepers
- referees
- ball

The field model is used for:

- football pitch keypoints
- homography estimation
- minimap projection

Model loading happens in `vision/models.py` through Roboflow's `inference.get_model`.

### Tracking ReID model (local, not trained in Roboflow)

When `BOTSORT_WITH_REID=true` (the case in the verified-good config), the tracker
also uses a **local appearance model** for identity stabilization:

```dotenv
BOTSORT_WITH_REID=true
BOTSORT_REID_MODEL=yolo11n-cls.pt
```

`yolo11n-cls.pt` is a **pretrained** Ultralytics classifier bundled in the repo
(`vision/track.py` loads it). It is *not* trained in Roboflow and is not part of
the training pipeline below — it ships as-is.

### Configuration delivery (`.env` vs `baseline.env`)

Real secrets live in `.env`, which is **gitignored** and never shipped. The
verified-good settings are stored in a tracked, secrets-stripped file
**`baseline.env`** (tag `good-baseline-may9`). To run:

```bash
cp baseline.env .env          # then paste your own ROBOFLOW_API_KEY into .env
```

## Training Pipeline

Training is done in Roboflow, not directly in this repository.

Recommended workflow:

1. Collect more match frames from the target camera angle and video quality.
2. Upload frames/videos to Roboflow.
3. Annotate object classes:
   - `player`
   - `goalkeeper`
   - `referee`
   - `ball`
4. For homography, annotate pitch keypoints in the field/keypoint project.
5. Generate a dataset version.
6. Apply conservative preprocessing/augmentation:
   - resize to the selected training size
   - brightness/exposure variation
   - mild blur/noise for low-quality broadcast footage
   - small hue/saturation variation
7. Train the detector/keypoint model in Roboflow.
8. Copy the new Roboflow model version ID into `.env`.
9. Run this repository's pipeline on a held-out match clip.
10. Compare output video quality and KPI summaries before replacing the model ID.

### Model access (required for anyone running the pipeline)

`inference.get_model(model_id, api_key)` only works if the API key's Roboflow
account is allowed to use that model. Each `ROBOFLOW_API_KEY` must therefore
either:

- belong to the **workspace that owns** the project, or
- the project must be **public** on Roboflow Universe.

If someone else (e.g. a client) runs this with their own key, the project must be
public or they must be added to the workspace — otherwise loading fails
regardless of the code. This is the most common cause of a failed handoff.

## Development and Evaluation Pipeline

The local development pipeline is the code in this repository.

Main entry point:

```bash
python -u main.py \
  --source-video "short_sample/HILAL-HAZM_match_B_up7.mp4" \
  --out-dir "outputs_test/run_name" \
  --enable-team
```

Important files:

- `main.py` - end-to-end pipeline orchestration
- `core/config.py` - all `.env` configuration options
- `vision/models.py` - Roboflow model loading
- `vision/detect.py` - detector output parsing/filtering
- `vision/track.py` - tracker setup
- `vision/id_stabilizer.py` - identity stabilization logic
- `vision/team_color.py` - color-based team classification
- `geometry/homography.py` - field homography estimation
- `io_utils/kpi.py` - KPI summary output

Each run writes:

- `annotated.mp4`
- `per_frame_tracks.csv`
- `kpi_summary.json`
- `kpi_summary.csv`

These outputs are used to decide whether a newly trained model actually improves
the pipeline.

## What To Send As The Pipeline Summary

Use this wording when asked for the model training and development pipelines:

> The training pipeline is Roboflow-based. We collect representative football
> match frames, annotate player/goalkeeper/referee/ball object classes and field
> keypoints, generate a dataset version with conservative augmentations for
> low-quality broadcast footage, train the Roboflow model, then update the
> model IDs in `.env`.
>
> The development pipeline is local in this repository. `main.py` loads the
> Roboflow models, runs detection, tracking, team color classification,
> homography estimation, minimap projection, and writes an annotated video,
> per-frame CSV, and KPI summaries for evaluation.

## Current Limitation

There is no local YOLO/RF-DETR training script in this repository yet. If we
need fully reproducible local training, the next step is to add an export-based
training script that downloads a Roboflow dataset version and trains YOLO/RF-DETR
outside Roboflow.
