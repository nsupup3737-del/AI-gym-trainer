from services.config.workout_config import PROMPT

FALLBACKS = {
    "workout_started": "Let's start your workout. Keep your back straight and maintain good form.",
    "set_completed": "Great job. Take a short rest and prepare for the next set.",
    "workout_completed": "Excellent workout. You completed today's session.",
    "no_pose_detected": "Step back and keep your whole body inside the camera frame.",
    "ongoing_form_check": "Keep going. Maintain proper form.",
}


class LLMCoach:
    def __init__(self, groq_client):
        self.client = groq_client
        self.system_prompt = PROMPT

    def give_feedback(self, event, issue=None):
        user_message = f"Event: {event}"
        if issue:
            user_message += f" Form Issue: {issue}"

        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.4,
            )
            text = response.choices[0].message.content
            if text:
                return text.strip()
        except Exception as e:
            print("Groq error:", e)

        return FALLBACKS.get(event, FALLBACKS["ongoing_form_check"])