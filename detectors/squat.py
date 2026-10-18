from core.base_exercise import BaseExercise


class SquatDetector(BaseExercise):
    DOWN_THRESHOLD = 100
    UP_THRESHOLD = 160
    MIN_VISIBILITY = 0.5

    LEFT_HIP = 23
    LEFT_KNEE = 25
    LEFT_ANKLE = 27
    RIGHT_HIP = 24
    RIGHT_KNEE = 26
    RIGHT_ANKLE = 28
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    def __init__(self):
        super().__init__()

    def reset(self):
        self.reps = 0
        self.stage = None

    def _knee_angle(self, landmarks, hip, knee, ankle):
        return self.calculate_angle(
            self.get_point(landmarks, hip),
            self.get_point(landmarks, knee),
            self.get_point(landmarks, ankle),
        )

    def process(self, landmarks):
        # Use the leg that is more visible to the camera
        if landmarks[self.LEFT_KNEE].visibility >= landmarks[self.RIGHT_KNEE].visibility:
            hip_idx, knee_idx, ankle_idx, shoulder_idx = (
                self.LEFT_HIP, self.LEFT_KNEE, self.LEFT_ANKLE, self.LEFT_SHOULDER
            )
        else:
            hip_idx, knee_idx, ankle_idx, shoulder_idx = (
                self.RIGHT_HIP, self.RIGHT_KNEE, self.RIGHT_ANKLE, self.RIGHT_SHOULDER
            )

        knee_angle = self._knee_angle(landmarks, hip_idx, knee_idx, ankle_idx)

        back_angle = self.calculate_angle(
            self.get_point(landmarks, shoulder_idx),
            self.get_point(landmarks, hip_idx),
            self.get_point(landmarks, knee_idx),
        )

        key_landmarks_visible = all(
            landmarks[i].visibility >= self.MIN_VISIBILITY
            for i in (hip_idx, knee_idx, ankle_idx)
        )

        if key_landmarks_visible:
            if knee_angle < self.DOWN_THRESHOLD:
                self.stage = "down"

            if knee_angle >= self.UP_THRESHOLD and self.stage == "down":
                self.stage = "up"
                self.reps += 1

        
        if self.stage == "down":
            depth_status = "GOOD DEPTH"
        elif knee_angle >= self.UP_THRESHOLD:
            depth_status = "STANDING"
        elif key_landmarks_visible:
            depth_status = "GO LOWER"
        else:
            depth_status = "N/A"

        return {
            "reps": self.reps,
            "knee_angle": int(knee_angle),
            "back_angle": int(back_angle),
            "depth_status": depth_status,
        }