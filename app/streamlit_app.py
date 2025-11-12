"""Streamlit interface for the Iris prediction pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

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
        st.error("The model has not been trained yet. Run `python src/train.py` first.")
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
if "scroll_predictor" not in st.session_state:
    st.session_state.scroll_predictor = False

cta_clicked = iris_landing_component(
    title="Iris Predictor Studio",
    subtitle="Train, evaluate, and test a logistic-regression pipeline on the classic Iris dataset — no ML background needed.",
    highlight="Guided playground",
    bullets=[
        "See the dataset, the pipeline, and the metrics in one place",
        "Understand how the model learns before running predictions",
        "Use the friendly form to try a flower instantly",
    ],
    metrics=[
        {"label": "Accuracy CV", "value": "0.97 ± 0.02"},
        {"label": "F1-macro CV", "value": "0.97 ± 0.02"},
        {"label": "Hold-out", "value": "≈ 0.95 - 1.00"},
    ],
    cta_label="Launch the prediction workshop",
)

if cta_clicked:
    st.session_state.show_predictor = True
    st.session_state.scroll_predictor = True

if not st.session_state.show_predictor:
    st.stop()

st.markdown('<div id="predictor-anchor"></div>', unsafe_allow_html=True)
st.divider()
st.header("Simple prediction workshop")
st.caption(
    "Enter petal and sepal lengths/widths (0 to 10 cm). The model figures out which Iris species is the best match."
)
st.write(
    "Each field expects a measurement in centimeters. Just pick the values you observe, then hit "
    "“Predict” to see the result and the confidence per species."
)

cols = st.columns(2)
user_inputs = []
for idx, (feature, (min_val, max_val, default)) in enumerate(FEATURE_BOUNDS.items()):
    col = cols[idx % 2]
    label = feature.replace("_", " ").title() + " (cm)"
    user_inputs.append(
        col.number_input(label, min_value=min_val, max_value=max_val, value=default, step=0.1)
    )

predict_btn = st.button("Predict", type="primary")
if predict_btn:
    model = load_model()
    sample = np.array(user_inputs).reshape(1, -1)
    pred_idx = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]

    st.success(
        f"The model thinks this flower is most likely **{TARGET_NAMES[pred_idx]}**."
    )
    st.write("Confidence per species:")
    st.dataframe(
        {
            "Species": TARGET_NAMES,
            "Probability": [round(prob, 3) for prob in probabilities],
        }
    )
else:
    st.info(
        "Adjust the sliders above and click “Predict” to run the trained model."
    )

if st.session_state.get("scroll_predictor"):
    st.session_state.scroll_predictor = False
    components.html(
        """
        <script>
        const scrollToPredictor = () => {
            const anchor = window.parent.document.getElementById('predictor-anchor');
            if (anchor) {
                anchor.scrollIntoView({behavior: 'smooth', block: 'start'});
            }
        };
        setTimeout(scrollToPredictor, 100);
        </script>
        """,
        height=0,
    )
