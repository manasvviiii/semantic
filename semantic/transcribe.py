import argparse
from pathlib import Path

from faster_whisper import WhisperModel


def transcribe_audio(audio_path, model_size="base"):
    path = Path(audio_path)

    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
    )

    segments, info = model.transcribe(
        str(path),
        beam_size=3,
        vad_filter=True,
    )

    transcript_segments = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            transcript_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": text,
            })

    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "transcript": " ".join(
            segment["text"] for segment in transcript_segments
        ),
        "segments": transcript_segments,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to a WAV, MP3, or supported audio file")
    parser.add_argument("--model", default="base")
    args = parser.parse_args()

    result = transcribe_audio(args.audio, args.model)

    print(f"Detected language: {result['language']}")
    print("\nTranscript:")
    print(result["transcript"])
