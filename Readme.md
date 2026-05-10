# Soccer Analytics Pipeline (Detection · Tracking · Teams · Homography)

A computer-vision pipeline that processes football match footage and produces:

- An **annotated video** with stable player IDs, team colors, and a top-down minimap.
- A **per-frame CSV** with track IDs, classes, image-space boxes, and pitch-space coordinates.
- Console diagnostics (homography quality, ID stability, team-classification status).

## Features

- **Detection** — players, goalkeepers, referees, and the ball (Roboflow-hosted YOLO models).
- **Tracking** — BoTSort with ReID, wrapped in a 4-pass `IDStabilizer` that handles re-entry, occlusions, and team-color veto.
- **Homography** — RANSAC fit on field keypoints with EMA smoothing, condition-number filtering, and full-frame reprojection check.
- **Team classification** — lightweight HSV-histogram color classifier (`ColorTeamClassifier`), vote-stabilized per track.
- **Goalkeeper sanity** — demotes false-positive GKs that aren't near a goal zone.

## Three ways to run it

1. **CLI script** — `main.py` (or the `run.sh` helper).
2. **REST API** — `api.py` (FastAPI, background jobs).
3. **Web UI** — `streamlit_app.py` (talks to the API).

---

## 1. Requirements

- Python **3.10+** (3.11 recommended)
- `ffmpeg` on PATH (recommended for reliable video I/O)
- A Roboflow API key with access to the player and field models

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Streamlit note:** `streamlit` is intentionally **not** in `requirements.txt` because it can conflict with the Roboflow `inference` stack. Install it in a separate venv if you need the UI:
>
> ```bash
> pip install streamlit==1.38.0 requests
> ```

---

## 2. Configuration (`.env`)

Create a `.env` file at the project root. **`.env` is gitignored** — never commit it.

Minimal template:

```dotenv
ROBOFLOW_API_KEY=your_roboflow_key_here
HF_TOKEN=optional_huggingface_token

PLAYER_MODEL_ID=footballs-player-detection-zkams-zia6c/2
FIELD_MODEL_ID=football-field-detection-f07vi/15

DEVICE=cpu                # or "cuda" if you have a GPU
MAX_FRAMES=0              # 0 = process the whole video
DETECT_EVERY_N=1
HOMOGRAPHY_EVERY_N=1
TEAM_MODE=color
```

Useful additional knobs (defaults live in `core/config.py`):

```dotenv
# Detection / preprocessing
DET_CONF=0.30
KP_CONF=0.30
PREPROCESS_ENABLED=true

# Homography
H_EMA_ALPHA=0.50
RANSAC_REPROJ_THRESH=25.0
MIN_KP=6
MIN_INLIER_RATIO=0.30
MAX_REPROJ_ERR=80.0

# Tracking (BoTSort + ReID)
TRACKER_TYPE=botsort
BOTSORT_WITH_REID=true
BOTSORT_REID_MODEL=yolo11n-cls.pt
TRACK_LOST_BUFFER=45

# ID stabilization
ID_RELINK_FRAMES=5400
ID_MEMORY_FRAMES=5400
ID_RELINK_PX=130
ID_APP_WEIGHT=0.80
ID_GALLERY_SIZE=64
ID_MAX_PLAYER_IDS=22
ID_MAX_GOALKEEPER_IDS=2
ID_MAX_REFEREE_IDS=2
```

The full set of supported keys is in `core/config.py`.

---

## 3. Run as a CLI script

`main.py` uses argparse — you don't need to edit any file.

```bash
python3 main.py \
  --source-video path/to/match.mp4 \
  --out-dir outputs \
  --enable-team
```

**Flags**

| Flag | Required | Description |
|------|----------|-------------|
| `--source-video` | yes | Path to input MP4 |
| `--out-dir` | no (default `outputs`) | Where to write artifacts |
| `--enable-team` | no | Turn on team-color classification |

**Or use the helper script** (edit `run.sh` to point at your video):

```bash
./run.sh
```

**Outputs** (in `--out-dir`):

- `annotated.mp4` — annotated video with side-panel minimap
- `per_frame_tracks.csv` — per-frame, per-track records

---

## 4. Run as a REST API

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

**`POST /analyze-video`** — upload an MP4, kick off a background job.

```bash
curl -X POST "http://127.0.0.1:8000/analyze-video?enable_team=true" \
  -F "file=@match.mp4"
```

Response:

```json
{ "job_id": "…", "status": "queued" }
```

**`GET /jobs/{job_id}`** — poll status (`queued`, `running`, `done`, `failed`).

```bash
curl "http://127.0.0.1:8000/jobs/<job_id>"
```

**`GET /jobs/{job_id}/artifacts/{filename}`** — download outputs once `done`.

```bash
curl -O "http://127.0.0.1:8000/jobs/<job_id>/artifacts/annotated.mp4"
curl -O "http://127.0.0.1:8000/jobs/<job_id>/artifacts/per_frame_tracks.csv"
```

Files on disk:
- Uploads: `uploads/{job_id}.mp4`
- Outputs: `outputs/{job_id}/`

---

## 5. Run with the Streamlit UI

Start the API in one terminal, the UI in another:

```bash
# Terminal 1
uvicorn api:app --reload --port 8000

# Terminal 2 (use a venv that has streamlit installed)
streamlit run streamlit_app.py
```

Open `http://localhost:8501`. The UI lets you upload a clip, toggle team classification, watch progress, and download the annotated video and CSV.

---

## 6. Project layout

```
.
├── main.py                # CLI entry point
├── api.py                 # FastAPI service
├── streamlit_app.py       # Streamlit frontend
├── run.sh                 # one-liner CLI helper
├── requirements.txt
├── .env                   # local config (gitignored)
│
├── core/
│   └── config.py          # Pydantic Settings (reads .env)
├── vision/
│   ├── detect.py          # Roboflow inference wrappers
│   ├── track.py           # BoTSort tracker
│   ├── id_stabilizer.py   # 4-pass ID stabilization
│   ├── team_color.py      # HSV team classifier
│   ├── teams.py           # team assignment glue
│   └── …
├── geometry/
│   └── homography.py      # RANSAC homography + EMA smoothing
└── io_utils/
    ├── writers.py         # video + CSV writers
    └── minimap.py         # side-panel renderer
```

---

## 7. Troubleshooting

**Pipeline runs but homography is mostly `HOLD`**
- Lower `MIN_INLIER_RATIO` (e.g. `0.25`) and raise `MAX_REPROJ_ERR` (e.g. `100`) in `.env`.
- Check the `[homography] reject reason=…` lines printed at startup — they tell you which threshold is failing.

**Player IDs keep changing after re-entry**
- Increase `ID_RELINK_FRAMES` and `ID_MEMORY_FRAMES`.
- Confirm `BOTSORT_WITH_REID=true` and that `yolo11n-cls.pt` is present in the project root.
- If team classification is on, IDs benefit from team-color veto — re-runs after the classifier is "ready" usually stabilize quickly.

**Goalkeepers labeled as players (or vice versa)**
- The pipeline only demotes a GK→player when the detection is far from either goal zone (controlled by `GK_GOAL_ZONE_X_M` in `main.py`). It never promotes a player to GK, by design.

**Slow on CPU**
- Set `DETECT_EVERY_N=2` (run detector every other frame) and `HOMOGRAPHY_EVERY_N=2`.
- Set `MAX_FRAMES` to a small value (e.g. `300`) for quick iteration.
- Run without `--enable-team` for the fastest path.

**Roboflow auth errors**
- Confirm `ROBOFLOW_API_KEY` in `.env` is set and that your account has access to both `PLAYER_MODEL_ID` and `FIELD_MODEL_ID`.

---

## 8. Branch / deliverable

This branch (`final-deliverable`) contains the full pipeline as described above.
