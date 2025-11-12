# Iris Predictor

End-to-end Iris classification pipeline featuring automated training, evaluation/inference scripts, an exploratory notebook, and a Streamlit web experience powered by a custom React hero component.

## Overview
- **Dataset**: Iris (scikit-learn) with 150 balanced samples over 3 species.
- **Model**: `StandardScaler` + `LogisticRegression(max_iter=1000)` packed in a scikit-learn pipeline.
- **Validation**: stratified 80/20 split plus 5-fold cross-validation.
- **Target accuracy**: 95–98% with transparent metrics.

## Getting started
```bash
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -r requirements.txt

python src/train.py        # trains and saves models/iris_pipeline.joblib
python src/evaluate.py     # reruns a 5-fold CV on the persisted pipeline
python src/infer.py 5.1 3.5 1.4 0.2  # quick CLI prediction
streamlit run app/streamlit_app.py   # launches the web app
```

## Project structure
```
iris-predictor/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ notebook/
│  └─ 01_exploration.ipynb
├─ src/
│  ├─ utils.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ infer.py
├─ models/
│  └─ iris_pipeline.joblib (created after training)
├─ landing_component/
│  ├─ __init__.py
│  └─ frontend/ (React + Vite animated hero)
└─ app/
   └─ streamlit_app.py
```

## Key scripts
- `src/utils.py`: loads and cleans the Iris dataset via `load_iris_df`.
- `src/train.py`: handles the 80/20 split, 5-fold CV (accuracy + macro F1), full training, and model export.
- `src/evaluate.py`: reloads `iris_pipeline.joblib` and reports CV mean ± std on accuracy/F1.
- `src/infer.py`: CLI helper (`python src/infer.py 5.1 3.5 1.4 0.2`) that prints the predicted species and probabilities.
- `app/streamlit_app.py`: Streamlit UI with an animated landing (custom component) and a friendly prediction form.
- `landing_component/`: React/Vite project compiled into a Streamlit component. Build via `cd landing_component/frontend && npm install && npm run build`.

## Exploration notebook
`notebook/01_exploration.ipynb` contains 7 cells (loading, descriptive stats, two Matplotlib charts, and a short conclusion) to verify the linear separability of the species.

## Expected results
Typical `python src/train.py` output:
- CV (train split): accuracy ≈ 0.97 ± 0.02, F1-macro ≈ 0.97 ± 0.02.
- Hold-out 20%: accuracy/F1 close to 1.00 thanks to the linear boundaries.

`python src/evaluate.py` reruns the 5-fold CV on the full dataset to confirm the saved pipeline.

### Sample console run
```text
$ python src/train.py
5-fold CV metrics on training split:
  Accuracy: 0.978 ± 0.015
  F1-macro: 0.977 ± 0.018

Hold-out test metrics:
  Accuracy: 1.000
  F1-macro: 1.000

Model trained on full data and saved to models/iris_pipeline.joblib
Classes: setosa, versicolor, virginica

$ python src/evaluate.py
5-fold cross-validation on full dataset using saved pipeline:
  Accuracy: 0.980 ± 0.014
  F1-macro: 0.979 ± 0.015
```

## Next steps
1. **GridSearchCV**: tune `C`, `penalty`, and compare `l1`/`l2` regularization.
2. **Confusion matrix**: visualize the rare versicolor vs. virginica mix-ups.
3. **Streamlit Cloud**: deploy `app/streamlit_app.py` (model artifact included) for self-serve demos.

---
Full-stack classification demo (scikit-learn + Streamlit): pipeline, 5-fold validation, persisted model, and interactive prediction UI.

## Animated landing (React)
The Streamlit landing view relies on a custom component (`landing_component/`) built with React, Vite, `streamlit-component-lib`, and Framer Motion.

To edit or rebuild it:
```bash
cd landing_component/frontend
npm install          # first time only
npm run build        # regenerates landing_component/frontend/dist
```
The `dist/` folder is versioned so Streamlit Cloud can serve the compiled assets without running any Node.js build step. Modify the React sources, run `npm run build`, and commit the updated assets.
