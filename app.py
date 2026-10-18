import os
import threading
import time

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

st.set_page_config(page_title="Camera Metrics", layout="wide")

# ---------------------------------------------------------------------------
# WebRTC / ICE configuration (STUN + optional TURN from env vars on Render)
# ---------------------------------------------------------------------------
def get_rtc_configuration():
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

    turn_url = os.environ.get("TURN_URL")  # e.g. turn:global.relay.metered.ca:443?transport=tcp
    turn_user = os.environ.get("TURN_USERNAME")
    turn_pass = os.environ.get("TURN_PASSWORD")
    if turn_url and turn_user and turn_pass:
        ice_servers.append(
            {"urls": [turn_url], "username": turn_user, "credential": turn_pass}
        )
    return {"iceServers": ice_servers}


# ---------------------------------------------------------------------------
# Video processor: runs MediaPipe FaceMesh on each frame and stores metrics
# ---------------------------------------------------------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(pts):
    a = np.linalg.norm(pts[1] - pts[5])
    b = np.linalg.norm(pts[2] - pts[4])
    c = np.linalg.norm(pts[0] - pts[3])
    return (a + b) / (2.0 * c + 1e-6)


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lock = threading.Lock()
        self._latest_metrics = None
        self._last_time = time.time()
        self._blinks = 0
        self._eye_closed = False

    @property
    def latest_metrics(self):
        with self._lock:
            return None if self._latest_metrics is None else dict(self._latest_metrics)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        now = time.time()
        fps = 1.0 / max(now - self._last_time, 1e-6)
        self._last_time = now

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        metrics = {"face_detected": False, "fps": round(fps, 1), "blinks": self._blinks}

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            pts = np.array([[p.x * w, p.y * h] for p in lm])

            ear = (
                eye_aspect_ratio(pts[LEFT_EYE]) + eye_aspect_ratio(pts[RIGHT_EYE])
            ) / 2.0

            if ear < 0.2 and not self._eye_closed:
                self._eye_closed = True
            elif ear >= 0.2 and self._eye_closed:
                self._eye_closed = False
                self._blinks += 1

            # Rough head tilt from the line between outer eye corners
            dx, dy = pts[263] - pts[33]
            tilt = float(np.degrees(np.arctan2(dy, dx)))

            metrics.update(
                face_detected=True,
                eye_aspect_ratio=round(float(ear), 3),
                head_tilt_deg=round(tilt, 1),
                blinks=self._blinks,
            )

            for idx in LEFT_EYE + RIGHT_EYE:
                cv2.circle(img, tuple(pts[idx].astype(int)), 2, (0, 255, 0), -1)

        cv2.putText(
            img, f"FPS: {metrics['fps']}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
        )

        with self._lock:
            self._latest_metrics = metrics

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------------------------------------------------------------
# Voice pipeline placeholder - replace with your own implementation
# ---------------------------------------------------------------------------
class VoicePipeline:
    def __init__(self):
        self.ready = True

    def respond(self, metrics):
        if not metrics or not metrics.get("face_detected"):
            return "I can't see your face right now."
        return (
            f"Blinks: {metrics['blinks']}, "
            f"eye ratio: {metrics['eye_aspect_ratio']}, "
            f"head tilt: {metrics['head_tilt_deg']} degrees."
        )


@st.cache_resource
def get_voice_pipeline():
    return VoicePipeline()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("Camera Metrics")

voice = get_voice_pipeline()

webrtc_ctx = webrtc_streamer(
    key="camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=get_rtc_configuration(),
    async_processing=True,
)

col1, col2 = st.columns(2)

with st.sidebar:
    st.subheader("Debug")
    st.write(f"Camera playing: `{webrtc_ctx.state.playing}`")
    st.write(f"Video processor created: `{webrtc_ctx.video_processor is not None}`")
    metrics = (
        webrtc_ctx.video_processor.latest_metrics
        if webrtc_ctx.state.playing and webrtc_ctx.video_processor
        else None
    )
    st.write(f"Latest metrics: `{metrics}`")
    st.write(f"Voice pipeline ready: `{voice.ready}`")

with col1:
    st.subheader("Live metrics")
    if metrics:
        st.json(metrics)
    else:
        st.info("Click START and allow camera access.")

with col2:
    st.subheader("Voice response")
    if st.button("Speak status"):
        st.write(voice.respond(metrics))

# Auto-refresh while the stream is running so metrics update
if webrtc_ctx.state.playing:
    time.sleep(0.5)
    st.rerun()

    
