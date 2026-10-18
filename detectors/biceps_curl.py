import math
from core.base_exercise import BaseExercise


class BicepsCurlDetector(BaseExercise):
    UP_THRESHOLD = 60          # elbow angle at the top of the curl (fully bent)
    DOWN_THRESHOLD = 150       # elbow angle at the bottom (arm nearly straight)
    MIN_VISIBILITY = 0.5
    ELBOW_DRIFT_TOLERANCE = 0.08
    SWING_THRESHOLD = 15

    LEFT_SHOULDER = 11
    LEFT_ELBOW = 13
    LEFT_WRIST = 15
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW = 14
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24

    def __init__(self):
        super().__init__()

    def reset(self) -> None:
        self.reps = 0
        self.stage = None

    def process(self, landmarks) -> dict:
        # Track the arm that is more visible to the camera
        if landmarks[self.LEFT_ELBOW].visibility >= landmarks[self.RIGHT_ELBOW].visibility:
            shoulder_idx = self.LEFT_SHOULDER
            elbow_idx = self.LEFT_ELBOW
            wrist_idx = self.LEFT_WRIST
        else:
            shoulder_idx = self.RIGHT_SHOULDER
            elbow_idx = self.RIGHT_ELBOW
            wrist_idx = self.RIGHT_WRIST

        elbow_angle = self.calculate_angle(
            self.get_point(landmarks, shoulder_idx),
            self.get_point(landmarks, elbow_idx),
            self.get_point(landmarks, wrist_idx),
        )

        key_landmarks_visible = all(
            landmarks[i].visibility > self.MIN_VISIBILITY
            for i in (shoulder_idx, elbow_idx, wrist_idx)
        )

        if key_landmarks_visible:
            if self.stage is None and elbow_angle > self.DOWN_THRESHOLD:
                self.stage = "down"

            if elbow_angle < self.UP_THRESHOLD and self.stage == "down":
                self.stage = "up"

            if elbow_angle > self.DOWN_THRESHOLD and self.stage == "up":
                self.stage = "down"
                self.reps += 1

        elbow_drift = abs(landmarks[elbow_idx].x - landmarks[shoulder_idx].x)
        if elbow_drift <= self.ELBOW_DRIFT_TOLERANCE:
            shoulder_status = "STABLE"
        else:
            shoulder_status = "ELBOW DRIFTING"

        hips_visible = (
            landmarks[self.LEFT_HIP].visibility > self.MIN_VISIBILITY
            and landmarks[self.RIGHT_HIP].visibility > self.MIN_VISIBILITY
        )

        if hips_visible:
            shoulder_mid_x = (landmarks[self.LEFT_SHOULDER].x + landmarks[self.RIGHT_SHOULDER].x) / 2
            shoulder_mid_y = (landmarks[self.LEFT_SHOULDER].y + landmarks[self.RIGHT_SHOULDER].y) / 2
            hip_mid_x = (landmarks[self.LEFT_HIP].x + landmarks[self.RIGHT_HIP].x) / 2
            hip_mid_y = (landmarks[self.LEFT_HIP].y + landmarks[self.RIGHT_HIP].y) / 2

            torso_angle_from_vertical = self._safe_angle(
                shoulder_mid_x - hip_mid_x,
                shoulder_mid_y - hip_mid_y,
            )

            if torso_angle_from_vertical <= self.SWING_THRESHOLD:
                swing_status = "NO SWING"
            else:
                swing_status = "SWINGING"
        else:
            swing_status = "N/A"

        return {
            "reps": self.reps,
            "elbow_angle": int(elbow_angle),
            "shoulder_status": shoulder_status,
            "swing_status": swing_status,
        }

    def _safe_angle(self, dx, dy):
        return math.degrees(math.atan2(abs(dx), abs(dy))) if dy != 0 else 0.0