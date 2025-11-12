# Iris Predictor

Pipeline complet de classification du dataset Iris alliant entraînement automatisé, scripts d'évaluation/inférence et application Streamlit prête à l'emploi.

## Aperçu
- **Dataset** : Iris (scikit-learn), 150 échantillons et 3 classes équilibrées.
- **Modèle** : `StandardScaler` + `LogisticRegression(max_iter=1000)` empaquetés dans un pipeline scikit-learn.
- **Validation** : séparation 80/20 + validation croisée stratifiée à 5 plis.
- **Objectif** : précision comprise entre 95 % et 98 % avec explicabilité totale.

## Installation & exécution
```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt

python src/train.py        # entraîne et sauvegarde models/iris_pipeline.joblib
python src/evaluate.py     # relance une CV 5-fold sur le pipeline sauvegardé
python src/infer.py 5.1 3.5 1.4 0.2  # inférence en CLI
streamlit run app/streamlit_app.py   # interface web
```

## Arborescence
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
│  └─ iris_pipeline.joblib (créé après l'entraînement)
└─ app/
   └─ streamlit_app.py
```

## Scripts principaux
- `src/utils.py` : charge et nettoie les données (`load_iris_df`).
- `src/train.py` : split 80/20, CV 5-fold (accuracy & F1 macro), entraînement final + sauvegarde.
- `src/evaluate.py` : recharge `iris_pipeline.joblib` et calcule accuracy/F1 moyenne ± écart-type.
- `src/infer.py` : script CLI (`python src/infer.py 5.1 3.5 1.4 0.2`) qui affiche l'espèce prédite et les probabilités.
- `app/streamlit_app.py` : 4 champs numériques (0–10 cm) + bouton « Prédire » montrant la classe et les probabilités.

## Notebook d'exploration
`notebook/01_exploration.ipynb` contient 7 cellules (chargement, statistiques descriptives, 2 visualisations Matplotlib et conclusion) pour valider la séparabilité des espèces.

## Résultats attendus
Une exécution type de `python src/train.py` fournit :
- CV 5-fold sur l'entraînement : accuracy ≈ 0.97 ± 0.02, F1-macro ≈ 0.97 ± 0.02.
- Hold-out 20 % : accuracy/F1 proches de 1.00 grâce à la séparation linéaire des classes.

`python src/evaluate.py` reproduit la CV 5-fold complète sur le dataset entier afin de vérifier la stabilité du pipeline sauvegardé.

## Next steps
1. **GridSearchCV** : affiner `C`, `penalty` et choisir une régularisation `l1`/`l2` hybride.
2. **Matrice de confusion** : visualiser les rares confusions (versicolor vs virginica).
3. **Streamlit Cloud** : déployer `app/streamlit_app.py` avec le modèle empaqueté pour des démos publiques.

---
Création d’un modèle de classification complet (scikit-learn + Streamlit) : pipeline, validation 5-fold, sauvegarde du modèle et interface web de prédiction.
