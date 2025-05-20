
import argparse
import os
from transcribe import Transcriber

def main():
    """

    This function acts as an entry for the EchoFilter AI Powered Personal Audio Firewall Project. This system listens to the input audio and intelligently filters 
    and flags content that matches a user’s predefined list of sensitive topics, trigger words, or personal data.

    Usage:
        python main.py --audio_path 'Path for input audio file (e.g.:- .wav or .mp3)'

    Arguments:
        --audio_path: Path to the input audio file (.wav, .mp3, etc.).
    """
    
    parser = argparse.ArgumentParser(description="EchoFilter")
    parser.add_argument("--audio_path", type=str, required=True, help="Input audio file")
    args = parser.parse_args()

    initial_transcript_path = "outputs/translated_transcript.txt"
    os.makedirs(os.path.dirname(initial_transcript_path), exist_ok=True)


    transcriber = Transcriber()
    initial_transcript = transcriber.speech_to_text(args.audio_path, initial_transcript_path)

    print("\n--- Initial Transcript ---")
    print(initial_transcript)

if __name__ == "__main__":
    main()
