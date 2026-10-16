import os
import time
import threading
from pathlib import Path

import av
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import VideoProcessorBase

from detectors.squat import SquatDetector
from detectors.pushup import PushUpDetector
from detectors.biceps_curl import BicepsCurlDetector
from detectors.shoulder_press import ShoulderPressDetector
from detectors.lunge import LungesDetector
from services.config.workout_config import POSE_CONNECTIONS


MIN_VIS = 0.7
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)  # BGR

_ROOT = Path(__file__).resolve().parents[2]


def _find_model_path() -> str:
    candidates = [
        _ROOT / "ml_models" / "pose_landmarker_full.task",
        Path(os.getcwd()) / "ml_models" / "pose_landmarker_full.task",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "pose_landmarker_full.task not found. Looked in: "
        + ", ".join(str(p) for p in candidates)
    )


OVERLAY_TEXT = {
    "Squats": lambda m: f"DEPTH: {m['depth_status']}",
    "Push-ups": lambda m: f"BODY: {m['body_alignment']} | HIP: {m['hip_status']}",
    "Biceps Curls (Dumbbell)": lambda m: f"SWING: {m['swing_status']}",
    "Shoulder Press": lambda m: f"EXT: {m['extension_status']} | BACK: {m['back_arch_status']}",
    "Lunges": lambda m: f"BALANCE: {m['balance_status']}",
}


class VideoProcessorClass(VideoProcessorBase):
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_metrics = None
        self._exercise_type = None  

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=_find_model_path()),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=MIN_VIS,
            min_pose_presence_confidence=MIN_VIS,
            min_tracking_confidence=MIN_VIS,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

        self._detectors = {
            "Squats": SquatDetector(),
            "Push-ups": PushUpDetector(),
            "Biceps Curls (Dumbbell)": BicepsCurlDetector(),
            "Shoulder Press": ShoulderPressDetector(),
            "Lunges": LungesDetector(),
        }

        self._last_ts_ms = 0

   
    def set_latest_metrics(self, metrics):
        with self._lock:
            self._latest_metrics = metrics.copy()

    def get_latest_metrics(self):
        with self._lock:
            return None if self._latest_metrics is None else self._latest_metrics.copy()

    def set_exercise(self, exercise_type):
        with self._lock:
            if exercise_type == self._exercise_type:
                return
            self._exercise_type = exercise_type
            self._latest_metrics = None  
            detector = self._detectors.get(exercise_type)
            if detector:
                detector.reset() 

    def get_exercise(self):
        with self._lock:
            return self._exercise_type

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

 
    def _draw_skeleton(self, img, landmarks):
        h, w = img.shape[:2]

        for start_idx, end_idx in POSE_CONNECTIONS:
            p1, p2 = landmarks[start_idx], landmarks[end_idx]
            if p1.visibility > MIN_VIS and p2.visibility > MIN_VIS:
                cv2.line(
                    img,
                    (int(p1.x * w), int(p1.y * h)),
                    (int(p2.x * w), int(p2.y * h)),
                    GREEN,
                    8,
                )

        for lm in landmarks:
            if lm.visibility > MIN_VIS:
                cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 8, BLUE, -1)

    def _draw_no_pose_warnings(self, img):
        for text, y in (("NO POSE DETECTED", 50), ("PLEASE FACE THE CAMERA", 100)):
            cv2.putText(
                img, text, (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2, cv2.LINE_AA,
            )

    def _draw_overlays(self, img, metrics, ex_type):
        cv2.putText(
            img,
            f"REPS: {metrics.get('reps', 0)}  STAGE: {metrics.get('stage')}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2, cv2.LINE_AA,
        )

        build_text = OVERLAY_TEXT.get(ex_type)
        if not build_text:
            return

        h = img.shape[0]
        cv2.putText(
            img, build_text(metrics), (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2, cv2.LINE_AA,
        )

    def _next_timestamp_ms(self):
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    def recv(self, frame):
        image = np.ascontiguousarray(
            cv2.flip(frame.to_ndarray(format="bgr24"), 1), dtype=np.uint8
        )

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = self._landmarker.detect_for_video(
                mp_image, self._next_timestamp_ms()
            )

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                self._draw_skeleton(image, landmarks)

                ex_type = self.get_exercise()
                detector = self._detectors.get(ex_type)

                if detector:
                    metrics = detector.process(landmarks)
                    metrics["pose_detected"] = True
                    metrics["stage"] = getattr(detector, "stage", None)
                    self._draw_overlays(image, metrics, ex_type)
                    self.set_latest_metrics(metrics)
            else:
                self._draw_no_pose_warnings(image)

                with self._lock:
                    if self._latest_metrics is None:
                        self._latest_metrics = {"pose_detected": False}
                    else:
                        self._latest_metrics["pose_detected"] = False

        except Exception as e:
            print("Video processing error:", e)

        return av.VideoFrame.from_ndarray(image, format="bgr24")