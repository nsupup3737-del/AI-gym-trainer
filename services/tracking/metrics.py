import time
import streamlit as st
from services.config.workout_config import METRICS_FIELDS
from services.persistence.exercise_repository import add_exercise


def _speak(event, exercise, metrics):
    pipeline = st.session_state.get("voice_pipeline")
    if not pipeline:
        return

    result = pipeline.process_event(event=event, exercise=exercise, metrics=metrics)

    if result:
        st.session_state.audio_to_play = result["audio"]
        st.session_state.coach_feedback = result["text"]


def _handle_metrics(latest, exercise):
    pose_detected = latest.get("pose_detected", True)

    if "reps" in latest:
        reps = latest["reps"] or 0
        st.session_state.reps = reps

        for key, default in METRICS_FIELDS.get(exercise, {}).items():
            st.session_state[key] = latest.get(key, default)

        reps_per_set = st.session_state.get("reps_per_set", 0)
        target_sets = st.session_state.get("target_sets", 0)

        if reps_per_set > 0 and target_sets > 0:
            sets_completed = min(reps // reps_per_set, target_sets)
            workout_complete = sets_completed >= target_sets
            current_set_reps = 0 if workout_complete else reps % reps_per_set
        else:
            sets_completed = 0
            current_set_reps = 0
            workout_complete = False

        st.session_state.sets_completed = sets_completed
        st.session_state.current_set_reps = current_set_reps
        st.session_state.workout_complete = workout_complete

        last_saved = st.session_state.get("last_saved_sets_completed", 0)

        if sets_completed > last_saved:
            newly_completed = sets_completed - last_saved
            now_ts = time.time()
            started_at = st.session_state.get("set_cycle_started_at") or now_ts
            user_id = st.session_state.get("user_id")

            if isinstance(user_id, int):
                add_exercise(
                    user_id,
                    exercise,
                    newly_completed * reps_per_set,
                    newly_completed,
                    int(now_ts - started_at),
                )

            if not workout_complete:
                _speak("set_completed", exercise, latest)

            st.session_state.set_cycle_started_at = now_ts
            st.session_state.last_saved_sets_completed = sets_completed

        if workout_complete and not st.session_state.get("last_notified_workout_complete"):
            st.session_state.last_notified_workout_complete = True
            _speak("workout_completed", exercise, latest)
            return

    if not pose_detected:
        _speak("no_pose_detected", exercise, latest)
    elif "reps" in latest and not st.session_state.get("workout_complete"):
        _speak("ongoing_form_check", exercise, latest)


def sync_metrics_update(context):
    if not context or not hasattr(context, "state") or not context.state.playing:
        return

    processor = getattr(context, "video_processor", None)
    exercise = st.session_state.get("exercise_type")

    if processor and exercise:
        processor.set_exercise(exercise)
        latest = processor.get_latest_metrics()

        if latest:
            _handle_metrics(latest, exercise)

    time.sleep(0.5)
    st.rerun()