from io import BytesIO
from gtts import gTTS


class TextToSpeech:
    def __init__(self, language="en", slow=False):
        self.language = language
        self.slow = slow

    def speak(self, text):
        if not text:
            return None

        try:
            audio = BytesIO()
            gTTS(text=text, lang=self.language, slow=self.slow).write_to_fp(audio)
            data = audio.getvalue()
            print("VOICE SIZE:", len(data))
            return data
        except Exception as e:
            print("TTS ERROR:", e)
            return None