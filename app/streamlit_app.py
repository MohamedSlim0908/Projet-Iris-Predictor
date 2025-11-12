"""Streamlit interface for the Iris prediction pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from landing_component import iris_landing_component  # noqa: E402
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

st.set_page_config(page_title="Iris Predictor", page_icon="🌸", layout="wide")

if "show_predictor" not in st.session_state:
    st.session_state.show_predictor = False

cta_clicked = iris_landing_component(
    title="Iris Predictor Studio",
    subtitle="Train, évalue et teste un pipeline Logistic Regression entièrement automatisé sur le dataset Iris.",
    highlight="Expérience ML interactive",
    bullets=[
        "StandardScaler + LogisticRegression(max_iter=1000)",
        "Validation croisée 5-fold (accuracy & F1-macro)",
        "Scripts train/evaluate/infer + app Streamlit",
    ],
    metrics=[
        {"label": "Accuracy CV", "value": "0.97 ± 0.02"},
        {"label": "F1-macro CV", "value": "0.97 ± 0.02"},
        {"label": "Hold-out", "value": "≈ 0.95 - 1.00"},
    ],
    cta_label="Lancer l'atelier de prédiction",
)

if cta_clicked:
    st.session_state.show_predictor = True

if not st.session_state.show_predictor:
    st.stop()

st.divider()
st.header("Atelier de prédiction temps réel")
st.caption("Saisis des mesures réalistes (0 à 10 cm) et observe la classe prédite en direct.")

cols = st.columns(2)
user_inputs = []
for idx, (feature, (min_val, max_val, default)) in enumerate(FEATURE_BOUNDS.items()):
    col = cols[idx % 2]
    label = feature.replace("_", " ").title() + " (cm)"
    user_inputs.append(
        col.number_input(label, min_value=min_val, max_value=max_val, value=default, step=0.1)
    )

predict_btn = st.button("Prédire", type="primary")
if predict_btn:
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
    st.info("Ajuste les curseurs puis clique sur « Prédire » pour lancer le pipeline.")
