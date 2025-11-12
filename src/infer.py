"""Command-line inference helper for the Iris classifier."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import joblib
import numpy as np

from utils import load_iris_df

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "iris_pipeline.joblib"


FEATURE_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict the Iris species for a single flower measurement sample."
    )
    for name in FEATURE_NAMES:
        parser.add_argument(name, type=float, help=f"Value for {name.replace('_', ' ')} (cm)")
    return parser.parse_args()


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Train the model via python src/train.py first."
        )

    args = parse_args()
    values: List[float] = [getattr(args, name) for name in FEATURE_NAMES]
    pipeline = joblib.load(MODEL_PATH)
    _, _, target_names = load_iris_df()

    sample = np.array(values).reshape(1, -1)
    pred_idx = pipeline.predict(sample)[0]
    probabilities = pipeline.predict_proba(sample)[0]
    pred_name = target_names[pred_idx]

    print(f"Predicted species: {pred_name}")
    for idx, prob in enumerate(probabilities):
        label = target_names[idx]
        print(f"  P({label}) = {prob:.3f}")


if __name__ == "__main__":
    main()
