import logging

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer


logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Camera Test")
st.title("Camera connection test")
st.write("Click START and allow camera access.")


def passthrough(frame):
    return frame


context = webrtc_streamer(
    key="camera-diagnostic",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    video_frame_callback=passthrough,
    async_processing=True,
)

st.write("WebRTC playing:", context.state.playing)
