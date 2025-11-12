"""Train a Logistic Regression pipeline on the Iris dataset."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import load_iris_df

RANDOM_STATE = 42
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "iris_pipeline.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, multi_class="auto", random_state=RANDOM_STATE),
            ),
        ]
    )


def main() -> None:
    X, y, target_names = load_iris_df()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    scoring = {"accuracy": "accuracy", "f1": "f1_macro"}
    cv_results = cross_validate(pipeline, X_train, y_train, cv=5, scoring=scoring, n_jobs=None)

    print("5-fold CV metrics on training split:")
    for metric, label in [("test_accuracy", "Accuracy"), ("test_f1", "F1-macro")]:
        scores = cv_results[metric]
        print(f"  {label}: {scores.mean():.3f} ± {scores.std():.3f}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    print("\nHold-out test metrics:")
    print(f"  Accuracy: {test_acc:.3f}")
    print(f"  F1-macro: {test_f1:.3f}")

    pipeline.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel trained on full data and saved to {MODEL_PATH}")
    print(f"Classes: {', '.join(target_names)}")


if __name__ == "__main__":
    main()
