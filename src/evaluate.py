"""Evaluate the persisted Iris classifier with cross-validation."""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.model_selection import cross_validate

from utils import load_iris_df

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "iris_pipeline.joblib"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Train the model via python src/train.py first."
        )

    pipeline = joblib.load(MODEL_PATH)
    X, y, _ = load_iris_df()

    scoring = {"accuracy": "accuracy", "f1": "f1_macro"}
    cv_results = cross_validate(pipeline, X, y, cv=5, scoring=scoring, n_jobs=None)

    print("5-fold cross-validation on full dataset using saved pipeline:")
    for metric, label in [("test_accuracy", "Accuracy"), ("test_f1", "F1-macro")]:
        scores = cv_results[metric]
        print(f"  {label}: {scores.mean():.3f} ± {scores.std():.3f}")


if __name__ == "__main__":
    main()
