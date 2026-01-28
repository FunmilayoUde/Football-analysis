# Soccer Analytics Pipeline (Detection • Tracking • Teams • Homography)

This project runs a soccer analytics pipeline that:
- Detects players/ball/referees/goalkeepers
- Tracks identities across frames
- Classifies teams (optional)
- Estimates homography from field keypoints and projects entities to pitch coordinates
- Outputs:
  - Annotated video (`annotated.mp4`)
  - Per-frame CSV (`per_frame_tracks.csv`)
  - Basic run logs (via console + CSV metadata columns)

It can be run:
1) As a local script (`main.py`)
2) As an API (FastAPI) with background jobs (`api.py`)
3) With a Streamlit frontend (`streamlit_app.py`)

---

## 1) Project Structure

Typical layout:
your_project/
main.py
api.py
streamlit_app.py
requirements.txt
core/
vision/
geometry/
io_utils/
outputs/
uploads/


---

## 2) Requirements

- Python 3.10+ (Python 3.11 recommended)
- ffmpeg installed (recommended for reliable video IO)

### Install Python dependencies
From the project root:

```bash
pip install -r requirements.txt
If you are adding the API + Streamlit, ensure these are included in requirements.txt:

fastapi

uvicorn[standard]

python-multipart

streamlit

requests

3) Configuration (Roboflow / Model IDs)
This project loads models published on Roboflow. You must provide:

ROBOFLOW_API_KEY

PLAYER_MODEL_ID

FIELD_MODEL_ID

Option A: Environment Variables (recommended)
Set these in your shell:

PowerShell (Windows)

setx ROBOFLOW_API_KEY "YOUR_KEY"
setx PLAYER_MODEL_ID "YOUR_ROBOFLOW_PLAYER_MODEL_ID"
setx FIELD_MODEL_ID "YOUR_ROBOFLOW_FIELD_MODEL_ID"
Restart your terminal after setx.

Mac/Linux

export ROBOFLOW_API_KEY="YOUR_KEY"
export PLAYER_MODEL_ID="YOUR_ROBOFLOW_PLAYER_MODEL_ID"
export FIELD_MODEL_ID="YOUR_ROBOFLOW_FIELD_MODEL_ID"
Option B: .env file
If your load_settings() supports .env, create a .env in project root:

ROBOFLOW_API_KEY=YOUR_KEY
PLAYER_MODEL_ID=YOUR_ROBOFLOW_PLAYER_MODEL_ID
FIELD_MODEL_ID=YOUR_ROBOFLOW_FIELD_MODEL_ID
4) Run as a Script (main.py)
Edit the bottom of main.py:

if __name__ == "__main__":
    SOURCE_VIDEO = r"C:\path\to\match.mp4"
    main(SOURCE_VIDEO, out_dir="outputs", enable_team=True)
Run:

python main.py
Outputs:

outputs/annotated.mp4

outputs/per_frame_tracks.csv

Notes
If your machine is slow / hangs, run in “demo mode”:

enable_team=False (team classification is heavy)

or reduce team fitting:

fit_team_stride=60

fit_team_max_frames=20

Example:

main(SOURCE_VIDEO, out_dir="outputs", enable_team=False)
5) Run as an API (FastAPI)
The API accepts a video upload (mp4), creates a job, runs the pipeline in the background, and exposes artifacts.

Start the API server
From project root:

uvicorn api:app --reload --host 0.0.0.0 --port 8000
Endpoints
POST /analyze-video
Upload an MP4 and start processing

Returns job_id immediately

Example (curl):

curl -X POST "http://127.0.0.1:8000/analyze-video?enable_team=false" \
  -F "file=@HILAL-AHLI_match_B_up1.mp4"
Response:

{
  "job_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "status": "queued"
}
GET /jobs/{job_id}
Check status:

curl "http://127.0.0.1:8000/jobs/<job_id>"
Status values:

queued

running

done

failed

GET /jobs/{job_id}/artifacts/{filename}
Download artifacts once done:

annotated.mp4

per_frame_tracks.csv

Example:

curl -O "http://127.0.0.1:8000/jobs/<job_id>/artifacts/annotated.mp4"
curl -O "http://127.0.0.1:8000/jobs/<job_id>/artifacts/per_frame_tracks.csv"
Where files are stored
Uploaded videos: uploads/{job_id}.mp4

Outputs: outputs/{job_id}/

6) Run with Streamlit Frontend
Streamlit provides a UI to upload an MP4, trigger processing, poll status, and download outputs.

Start API first (Terminal 1)
uvicorn api:app --reload --port 8000
Start Streamlit (Terminal 2)
streamlit run streamlit_app.py
Open the Streamlit URL printed in the terminal (usually):

http://localhost:8501

What you can do in the UI
Upload MP4

Toggle “Enable team classification” (recommended OFF for slower laptops)

Click “Analyze video”

Watch status updates

View the annotated output

Download the CSV

7) Troubleshooting
A) “System hangs / very slow”
Common causes:

Team classification embeddings (SigLIP) are heavy

CPU-only inference on long videos

Fixes:

Run with enable_team=False

Reduce team fitting load:

increase fit_team_stride (e.g., 60)

reduce fit_team_max_frames (e.g., 20)

Test with a short clip first (30–60s)

