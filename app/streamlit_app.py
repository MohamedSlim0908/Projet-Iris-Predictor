"""Streamlit interface for the Iris prediction pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils import load_iris_df  # noqa: E402

MODEL_PATH = ROOT / "models" / "iris_pipeline.joblib"


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        st.error("Le modèle n'a pas encore été entraîné. Lancez python src/train.py.")
        st.stop()
    return joblib.load(MODEL_PATH)


FEATURE_BOUNDS = {
    "sepal_length": (0.0, 10.0, 5.1),
    "sepal_width": (0.0, 10.0, 3.5),
    "petal_length": (0.0, 10.0, 1.4),
    "petal_width": (0.0, 10.0, 0.2),
}


_, _, TARGET_NAMES = load_iris_df()

st.set_page_config(page_title="Iris Predictor", page_icon="🌸")
st.title("Iris Predictor")
st.write(
    "Ce projet entraîne une régression logistique avec normalisation et propose une interface "
    "pour tester les prédictions sur le dataset Iris."
)

cols = st.columns(2)
user_inputs = []
for idx, (feature, (min_val, max_val, default)) in enumerate(FEATURE_BOUNDS.items()):
    col = cols[idx % 2]
    label = feature.replace("_", " ").title() + " (cm)"
    user_inputs.append(
        col.number_input(label, min_value=min_val, max_value=max_val, value=default, step=0.1)
    )

if st.button("Prédire"):
    model = load_model()
    sample = np.array(user_inputs).reshape(1, -1)
    pred_idx = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]

    st.success(f"Espèce prédite : **{TARGET_NAMES[pred_idx]}**")
    st.write("Probabilités par classe :")
    st.dataframe(
        {
            "Espèce": TARGET_NAMES,
            "Probabilité": [round(prob, 3) for prob in probabilities],
        }
    )
else:
    st.info("Saisissez des mesures réalistes (0-10 cm) puis cliquez sur Prédire.")
