
import argparse
import json

from semantic.transcribe import transcribe_audio
from semantic.slm_inference import predict


def analyze_call(audio_path, whisper_model="base"):
    # Step 1: Convert audio into text
    transcription = transcribe_audio(
        audio_path,
        model_size=whisper_model,
    )

    if not transcription["transcript"].strip():
        raise ValueError(
            "No speech was transcribed. Try a clearer audio recording."
        )

    # Step 2: Classify the transcript
    classification = predict(transcription["transcript"])

    return {
        "audio_file": audio_path,
        "language": transcription["language"],
        "transcript": transcription["transcript"],
        "prediction": classification["label"],
        "model_probability": classification["model_probability"],
        "class_probabilities": classification["class_probabilities"],
        "warning": (
            "Experimental model. This result is not proof of fraud "
            "and must not be treated as a definitive accusation."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vak-Siddhi semantic call analysis"
    )
    parser.add_argument("audio", help="Path to the audio file")
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small"],
    )
    args = parser.parse_args()

    result = analyze_call(
        args.audio,
        whisper_model=args.whisper_model,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
