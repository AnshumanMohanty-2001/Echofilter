# transcribe.py
from faster_whisper import WhisperModel

class Transcriber:
    """
    
    This class works on converting the input audio into text using the Small Faster-Whisper model. 
    
    Attributes:
        model (WhisperModel): An instance of the Faster-Whisper model to convert speech to text.
    
    Methods:
        transcribe_file(self, audio_path: str, output_path: str = None) -> str: performs speech to text translation and saves the initial transcript.
    """
    def __init__(self):
        model_size = "small"
        self.model = WhisperModel(model_size_or_path=model_size, device="auto", compute_type="int8")

    def speech_to_text(self, audio_path: str, output_path: str = None) -> str:
        segments, _ = self.model.transcribe(audio_path, beam_size=5)

        transcript = ""
        for part in segments:
            line = part.text.strip()
            transcript += line + "\n"

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            print(f"Initial Transcript saved to {output_path}")

        return transcript
