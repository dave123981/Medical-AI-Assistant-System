# Training Notebooks

matching versioning:

```
train_v1_decision_tree.ipynb
train_v2_random_forest.ipynb
train_v3_xgboost.ipynb
train_v4_neural_network.ipynb
```

## Dataset
[Disease Symptom Prediction Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)
on Kaggle. It ships four CSVs:

- `dataset.csv` — Disease, Symptom_1..Symptom_17 (one-hot-able symptom columns)
- `symptom_Description.csv` — Disease, Description
- `symptom_precaution.csv` — Disease, Precaution_1..4
- `Symptom-severity.csv` — Symptom, weight (optional, useful as a feature for v2+)

## What every notebook must export

Whatever the model, the notebook's last cell should always produce these
exact filenames so `model_loader.py` can pick them up without code changes:

```python
import json, joblib

joblib.dump(clf, "v1_decision_tree.joblib")          # bump filename per version
json.dump(symptom_columns, open("symptom_vocab.json", "w"))
json.dump(list(label_encoder.classes_), open("label_classes.json", "w"))
# symptom_Description.csv / symptom_precaution.csv are copied over unmodified
# from the raw Kaggle download.
```

Download the four files from Colab and place them in
`services/disease-diagnosis/models/`, then set `MODEL_FILENAME` and
`MODEL_VERSION` in the service's environment (or Dockerfile) to match.

## Version progression (per the project spec)

| Version | Model            | Notes |
|---------|------------------|-------|
| v1      | Decision Tree    | Baseline, fastest to explain/debug |
| v2      | Random Forest    | Should beat v1 on accuracy; add feature importances to the response if useful |
| v3      | XGBoost          | Watch for overfitting on a small/imbalanced label set |
| v4      | Neural Network   | Only worth it if v1-v3 plateau; needs more careful preprocessing (scaling) |

Each version should keep the same input/output contract
(`schemas.py` / `gateway/internal/models/diagnosis.go`) so swapping
versions in production is just a config change.
