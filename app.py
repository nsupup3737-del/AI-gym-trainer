import os
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from services.auth.login_wall import render_login_wall
from services.state.session_defaults import initial_session_defaults
from services.config.workout_config import EXERCISE_OPTIONS
from services.ui.style_loader import load_css, inject_local_font, inject_webrtc_styles
from services.persistence.exercise_repository import init_db, get_users_exercises
from services.vision.exercise_video_processor import VideoProcessorClass
from services.tracking.metrics import sync_metrics_update
from services.coaching.llm import LLMCoach
from services.coaching.tts import TextToSpeech
from services.coaching.voice_pipeline import VoicePipeline, play_audio




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))



def initialize_voice_pipeline():
    try:
        api_key = os.getenv("GROQ_API_KEY", "").strip()

        if not api_key:
            try:
                api_key = st.secrets["GROQ_API_KEY"]
            except Exception:
                api_key = ""

        if not api_key:
            st.warning("GROQ API key missing")
            return None

        client = Groq(api_key=api_key)
        llm = LLMCoach(client)
        tts = TextToSpeech(language="en", slow=False)

        print("Voice pipeline initialized successfully")
        return VoicePipeline(llm, tts)

    except Exception as e:
        print("Voice init error:", e)
        return None


def speak_event(event, exercise):
    """Run a voice event and store the result for playback on the next render."""
    pipeline = st.session_state.get("voice_pipeline")
    if not pipeline:
        return

    try:
        result = pipeline.process_event(event=event, exercise=exercise, metrics={})
        if result:
            st.session_state.coach_feedback = result["text"]
            st.session_state.audio_to_play = result["audio"]
    except Exception as e:
        print(f"{event} voice error:", e)




def render_exercise_metrics(exercise):
    ss = st.session_state

    if exercise == "Squats":
        st.subheader("Squat Metrics")
        st.metric("Knee Angle", f"{ss.get('knee_angle', 0)}°")
        st.metric("Back Angle", f"{ss.get('back_angle', 0)}°")
        st.metric("Depth", ss.get("depth_status", "N/A"))

    elif exercise == "Push-ups":
        st.subheader("Push-up Metrics")
        st.metric("Elbow Angle", f"{ss.get('elbow_angle', 0)}°")
        st.metric("Alignment", ss.get("body_alignment", "N/A"))
        st.metric("Hips", ss.get("hip_status", "N/A"))

    elif exercise == "Lunges":
        st.subheader("Lunge Metrics")
        st.metric("Knee Angle", f"{ss.get('front_knee_angle', 0)}°")
        st.metric("Torso Angle", f"{ss.get('torso_angle', 0)}°")
        st.metric("Balance", ss.get("balance_status", "N/A"))

    elif exercise == "Biceps Curls (Dumbbell)":
        st.subheader("Curl Metrics")
        st.metric("Elbow Angle", f"{ss.get('elbow_angle', 0)}°")
        st.metric("Shoulder", ss.get("shoulder_status", "N/A"))
        st.metric("Swing", ss.get("swing_status", "N/A"))

    elif exercise == "Shoulder Press":
        st.subheader("Shoulder Press Metrics")
        st.metric("Elbow Angle", f"{ss.get('elbow_angle', 0)}°")
        st.metric("Extension", ss.get("extension_status", "N/A"))
        st.metric("Back", ss.get("back_arch_status", "N/A"))


def start_workout(exercise, sets, reps):
    ss = st.session_state

    ss.exercise_type = exercise
    ss.target_sets = int(sets)
    ss.reps_per_set = int(reps)

    
    ss.reps = 0
    ss.sets_completed = 0
    ss.current_set_reps = 0
    ss.workout_complete = False
    ss.last_saved_sets_completed = 0
    ss.last_notified_workout_complete = False
    ss.set_cycle_started_at = time.time()

    ss.audio_to_play = None
    ss.current_audio = None
    ss.coach_feedback = None
    ss.styles_injected = False
    ss.workout_started = True

    speak_event("workout_started", exercise)


def end_workout(exercise):
    speak_event("workout_completed", exercise)
    st.session_state.workout_started = False



def main():
    st.set_page_config(
        page_title="AI Real-time GYM Coach",
        page_icon="🏋️",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # CSS / font
    css_file = os.path.join(BASE_DIR, "static", "style.css")
    font_file = os.path.join(BASE_DIR, "static", "AdobeClean.otf")

    if os.path.exists(css_file):
        load_css(css_file)

    if os.path.exists(font_file):
        inject_local_font(font_file, "AdobeClean")

    # Database
    init_db()

    # Login
    if not render_login_wall():
        return

    # Session
    initial_session_defaults()

    if "voice_pipeline" not in st.session_state:
        st.session_state.voice_pipeline = initialize_voice_pipeline()

    st.session_state.setdefault("audio_to_play", None)
    st.session_state.setdefault("coach_feedback", None)

    workout_started = st.session_state.get("workout_started", False)
    exercise = st.session_state.get("exercise_type")

    
    with st.sidebar:
        st.title("🏋️ Apna AI Coach")

        username = st.session_state.get("username")
        if username:
            st.caption(f"👤 Login as {username}")

        st.divider()
        st.subheader("Workout Plan")

        if not workout_started:
            plan_exercise = st.selectbox("Exercise", EXERCISE_OPTIONS, key="plan_exercise")

            plan_sets = st.number_input(
                "Sets", min_value=1, max_value=50, value=3, step=1, key="plan_sets"
            )

            plan_reps = st.number_input(
                "Reps per Set", min_value=1, max_value=100, value=10, step=1, key="plan_reps"
            )

            if st.button("Start Workout", width="stretch"):
                start_workout(plan_exercise, plan_sets, plan_reps)
                st.rerun()

        else:
            st.info(
                f"**{exercise}**\n\n"
                f"{st.session_state.target_sets} Sets × "
                f"{st.session_state.reps_per_set} Reps"
            )

            if st.button("End Workout", width="stretch"):
                end_workout(exercise)
                st.rerun()

            st.divider()
            st.subheader("Progress")

            st.metric("Total Reps", st.session_state.get("reps", 0))

            st.metric(
                "Current Set",
                f"{st.session_state.get('current_set_reps', 0)} / "
                f"{st.session_state.get('reps_per_set', 0)}",
            )

            st.metric(
                "Sets Completed",
                f"{st.session_state.get('sets_completed', 0)} / "
                f"{st.session_state.get('target_sets', 0)}",
            )

            render_exercise_metrics(exercise)



    st.title("AI Real-time GYM Coach")
    st.markdown("#### Real-time pose detection with AI voice coaching")

    
    new_audio = st.session_state.get("audio_to_play")
    if new_audio:
        st.session_state.current_audio = new_audio
        st.session_state.audio_to_play = None

    if st.session_state.get("current_audio"):
        play_audio(st.session_state.current_audio)

    text = st.session_state.get("coach_feedback")
    if text:
        st.success(f"🤖 Coach: {text}")

    
    context = None

    if not workout_started:
        st.markdown(
            """
            <div style="border:10px dashed #444; padding:50px;
                        text-align:center; color:#888">
                <h2>👈 Set your workout plan</h2>
                <p>Choose exercise, sets and reps then click Start Workout.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        context = webrtc_streamer(
            key="gym-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessorClass,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            desired_playing_state=True, 
        )

        proc = getattr(context, "video_processor", None)
        with st.expander("Debug (remove later)", expanded=True):
            st.write("Camera playing:", context.state.playing)
            st.write("Video processor created:", proc is not None)
            st.write("Latest metrics:", proc.get_latest_metrics() if proc else None)
            st.write("Voice pipeline ready:", st.session_state.get("voice_pipeline") is not None)

   

    st.divider()
    st.subheader("Workout History")

    user_id = st.session_state.get("user_id", 0)

    if isinstance(user_id, int):
        try:
            rows = get_users_exercises(user_id)

            data = [
                {
                    "Exercise": row["exercise_name"],
                    "Reps": row["reps"],
                    "Sets": row["sets"],
                    "Time (s)": row["time"],
                    "Date": row["created_at"],
                }
                for row in rows
            ]

            df = pd.DataFrame(data)

            if not df.empty:
                st.table(df)
            else:
                st.info("No workout history found.")

        except Exception as e:
            print("History error:", e)

    
    if workout_started and context:
        if not st.session_state.get("styles_injected"):
            inject_webrtc_styles()
            st.session_state.styles_injected = True
        sync_metrics_update(context)


if __name__ == "__main__":
    main()