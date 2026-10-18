import time
import streamlit as st

ALWAYS_SPEAK = {"workout_started", "set_completed", "workout_completed"}

COOLDOWNS = {"no_pose_detected": 8, "ongoing_form_check": 15}
DEFAULT_COOLDOWN = 5

BAD_STATES = {
    "hip_status": {
        "SAGGING": "Hips are sagging, tighten your core",
        "PIKED UP": "Hips are too high, lower them into a straight line",
    },
    "body_alignment": {"Poor Form": "Body is not in a straight line"},
    "shoulder_status": {"ELBOW DRIFTING": "Elbow is drifting away from your body"},
    "swing_status": {"SWINGING": "You are swinging your torso, keep it still"},
    "back_arch_status": {"Excessive Arch": "Lower back is arching too much"},
    "balance_status": {"OFF BALANCE": "You are leaning sideways, stay balanced"},
    "depth_status": {"GO LOWER": "Squat depth is too shallow, go lower"},
}


def issue_from_metrics(metrics):
    for key, messages in BAD_STATES.items():
        message = messages.get(metrics.get(key))
        if message:
            return message
    return None


class VoicePipeline:
    def __init__(self, llm, tts):
        self.llm = llm
        self.tts = tts
        self.last_spoken_at = 0

    def process_event(self, event, exercise, metrics=None):
        metrics = metrics or {}
        now = time.time()

        if event not in ALWAYS_SPEAK:
            cooldown = COOLDOWNS.get(event, DEFAULT_COOLDOWN)
            if now - self.last_spoken_at < cooldown:
                return None

        issue = metrics.get("issue") or issue_from_metrics(metrics)

        try:
            coach_text = self.llm.give_feedback(event, issue)
            print("COACH TEXT:", coach_text)

            audio = self.tts.speak(coach_text)

            if audio:
                self.last_spoken_at = now
                return {"text": coach_text, "audio": audio}

        except Exception as e:
            print("VOICE PIPELINE ERROR:", e)

        return None


def play_audio(audio):
    if audio:
        st.audio(audio, format="audio/mp3", autoplay=True)