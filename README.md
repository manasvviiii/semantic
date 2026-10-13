# Semantic Scam Detection

An experimental semantic pipeline for transcribing audio and classifying potentially coercive or scam-related language.

## Components

- `semantic/transcribe.py` — audio transcription
- `semantic/slm_inference.py` — V5 semantic inference
- `semantic/pipeline.py` — pipeline entry point
- `semantic/classifier.py` — baseline classifier
- `tests/` — automated tests
- `docs/semantic_model_evaluation.md` — evaluation notes

## Validation

The current test suite passes 8 tests.

## Limitations

This project is experimental. Its current evaluation is limited and does not establish real-world scam-detection accuracy. Predictions should not be treated as definitive judgments.
