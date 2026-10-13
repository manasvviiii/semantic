import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from semantic.dataset import load_dataset

MODEL_PATH = Path("models/semantic_classifier.joblib")
METRICS_PATH = Path("eval/semantic_metrics.json")


def build_classifier():
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=20000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def train():
    df = load_dataset()

    if len(df) < 10 or df["label"].value_counts().min() < 2:
        raise ValueError(
            "Add more labeled transcripts before training. "
            "Each class needs at least two examples, and 10 total "
            "is the minimum for this baseline."
        )

    train_texts, test_texts, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    model = build_classifier()
    model.fit(train_texts, y_train)

    predictions = model.predict(test_texts)

    metrics = {
        "test_samples": len(test_texts),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["legitimate", "scam_coercion"],
            output_dict=True,
            zero_division=0,
        ),
        "warning": (
            "Metrics are only as representative as the dataset. "
            "Use source-separated test data for stronger evaluation."
        ),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


def predict(text, model_path=MODEL_PATH):
    if not Path(model_path).exists():
        raise FileNotFoundError(
            "Model not found. Train it first using: "
            "python -m semantic.classifier train"
        )

    model = joblib.load(model_path)
    label = int(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]
    classes = list(model.named_steps["classifier"].classes_)
    scores = {
        str(int(cls)): float(prob)
        for cls, prob in zip(classes, probabilities)
    }

    return {
        "label": "scam_coercion" if label == 1 else "legitimate",
        "class": label,
        "model_probability": scores[str(label)],
        "class_probabilities": scores,
        "warning": (
            "Experimental transcript classifier; not proof of a scam "
            "or a calibrated risk probability."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "predict"])
    parser.add_argument("--text", help="Transcript text to classify")
    args = parser.parse_args()

    if args.action == "train":
        train()
    else:
        if not args.text:
            parser.error("predict requires --text")
        print(json.dumps(predict(args.text), indent=2))