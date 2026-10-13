
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "semantic_slm_security_v5.joblib"

_tokenizer = None
_encoder = None
_classifier = None
_model_metadata = None


def load_model():
    """Load V5 and its matching pretrained encoder once."""
    global _tokenizer, _encoder, _classifier, _model_metadata

    if _classifier is not None:
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"V5 model not found: {MODEL_PATH}")

    saved = joblib.load(MODEL_PATH)

    if not isinstance(saved, dict):
        raise TypeError("Expected V5 model to be a dictionary.")

    model_name = saved.get("model_name")
    if not model_name:
        raise ValueError("Saved model does not specify model_name.")

    classifier = saved.get("classifier")
    if classifier is None or not hasattr(classifier, "predict_proba"):
        raise ValueError("Saved model has no compatible classifier.")

    if 1 not in classifier.classes_:
        raise ValueError("Classifier has no coercive class (label 1).")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    encoder.to("cpu")
    encoder.eval()

    _tokenizer = tokenizer
    _encoder = encoder
    _classifier = classifier
    _model_metadata = saved


def encode_text(text):
    """Match the mean-pooling and normalization used during V5 training."""
    tokens = _tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=int(_model_metadata.get("max_length", 256)),
        return_tensors="pt",
    )

    with torch.inference_mode():
        output = _encoder(**tokens)
        hidden = output.last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)

        pooled = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

    vector = pooled.cpu().numpy()

    expected_dimensions = int(
        _model_metadata.get("embedding_dimensions", vector.shape[1])
    )
    if vector.shape[1] != expected_dimensions:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dimensions}, "
            f"received {vector.shape[1]}"
        )

    return vector


def predict(text):
    """Return a V5 prediction and both class probabilities."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string.")

    load_model()
    vector = encode_text(text.strip())

    probabilities = _classifier.predict_proba(vector)[0]
    classes = list(_classifier.classes_)

    scores = {
        str(int(cls)): float(prob)
        for cls, prob in zip(classes, probabilities)
    }

    label = int(_classifier.predict(vector)[0])

    return {
        "label": "scam_coercion" if label == 1 else "legitimate",
        "class": label,
        "model_probability": scores[str(label)],
        "class_probabilities": scores,
        "model_version": "V5",
        "warning": (
            "Experimental model. A prediction is not proof of fraud. "
            "Verify suspicious requests through official channels."
        ),
    }
