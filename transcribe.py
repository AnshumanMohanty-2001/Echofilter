from faster_whisper import WhisperModel
import nltk
from nltk.tokenize import sent_tokenize
import torch

nltk.download("punkt")
nltk.download('punkt_tab')

class Transcriber:
    """
    A transcription utility that uses the Faster-Whisper model to convert spoken audio into 
    punctuated, sentence-level text transcripts.

    Attributes:
        model (WhisperModel): A lightweight, efficient speech-to-text model configured for fast inference.

    Methods:
        speech_to_text(audio_path: str, output_path: str = None) -> str:
            Transcribes the input audio file to text using beam search for higher accuracy.
            Segments are joined and split into natural language sentences.
            Optionally saves the transcript to a file.
            Returns the cleaned, sentence-level transcript as a single string.
    """

    def __init__(self):
        model_size = "small"
        self.model = WhisperModel(model_size_or_path=model_size, device="auto", compute_type="int8")

    def speech_to_text(self, audio_path: str, output_path: str = None) -> str:
        segments, _ = self.model.transcribe(audio_path, beam_size=5)

        full_text = " ".join(" ".join(part.text.strip().split()) for part in segments)
        sentences = sent_tokenize(full_text)

        transcript = "\n".join(sentence.strip() for sentence in sentences)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            print(f"Initial Transcript saved to {output_path}")

        return transcript

