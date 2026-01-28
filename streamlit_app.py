import time
import requests
import streamlit as st

API_BASE = st.secrets.get("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Soccer Analytics Demo", layout="wide")
st.title("⚽ Soccer Analytics Pipeline")

st.sidebar.header("Settings")
enable_team = st.sidebar.checkbox("Enable team classification", value=False)
poll_every = st.sidebar.slider("Poll interval (seconds)", 1, 10, 2)

uploaded = st.file_uploader("Upload an MP4 video", type=["mp4"])

if uploaded is not None:
    st.info(f"Selected file: {uploaded.name} ({uploaded.size/1e6:.2f} MB)")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.video(uploaded)

    if st.button("Analyze video"):
        with st.spinner("Uploading to API and starting job..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), "video/mp4")}
            resp = requests.post(
                f"{API_BASE}/analyze-video",
                params={"enable_team": str(enable_team).lower()},
                files=files,
                timeout=120,
            )

        if resp.status_code != 200:
            st.error(f"API error: {resp.status_code} — {resp.text}")
            st.stop()

        data = resp.json()
        job_id = data["job_id"]
        st.success(f"Job created: {job_id}")

        status_box = st.empty()
        progress = st.progress(0)

        # Poll job status
        while True:
            r = requests.get(f"{API_BASE}/jobs/{job_id}", timeout=30)
            if r.status_code != 200:
                st.error(f"Failed to fetch job status: {r.status_code} — {r.text}")
                break

            job = r.json()
            status = job["status"]

            status_box.write(f"**Status:** `{status}`")

            if status == "queued":
                progress.progress(10)
            elif status == "running":
                progress.progress(60)
            elif status == "done":
                progress.progress(100)
                st.success("✅ Done!")

                artifacts = job.get("artifacts") or {}
                video_path = artifacts.get("annotated_video")
                csv_path = artifacts.get("csv")

                if video_path:
                    st.subheader("Annotated output")
                    st.video(f"{API_BASE}{video_path}")

                if csv_path:
                    st.subheader("Per-frame CSV")
                    csv_url = f"{API_BASE}{csv_path}"
                    csv_bytes = requests.get(csv_url, timeout=60).content
                    st.download_button(
                        label="Download CSV",
                        data=csv_bytes,
                        file_name="per_frame_tracks.csv",
                        mime="text/csv",
                    )

                break

            elif status == "failed":
                st.error(f"❌ Failed: {job.get('error')}")
                break

            time.sleep(poll_every)
