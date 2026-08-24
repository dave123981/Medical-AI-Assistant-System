"""
Loads whichever model version is active from disk, once, on first use.

Expected layout under MODEL_DIR (default: ./models), all produced by the
Colab training notebook and downloaded here — nothing is trained inside
this service:

    models/
      v1_decision_tree.joblib   <- sklearn DecisionTreeClassifier (or v2/v3/v4 equivalent)
      symptom_vocab.json        <- ordered list of symptom names used at train time
      label_classes.json        <- ordered list of disease names, index-aligned to model output
      symptom_description.csv   <- from the Kaggle dataset: Disease, Description
      symptom_precaution.csv    <- from the Kaggle dataset: Disease, Precaution_1..4

Swapping model versions is just: drop in v2_random_forest.joblib, set
MODEL_FILENAME=v2_random_forest.joblib, restart. No code change needed
unless the feature vector shape itself changes.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "v1_decision_tree.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1-decision-tree")


class ModelArtifacts:
    def __init__(self, model, vocab, label_classes, descriptions, precautions):
        self.model = model
        self.vocab = vocab
        self.label_classes = label_classes
        self.descriptions = descriptions  # dict: disease -> description
        self.precautions = precautions    # dict: disease -> list[str]


def _load_json_list(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing '{path.name}' at {path}. Train v1 in the Colab notebook "
            f"(see services/disease-diagnosis/notebooks/) and export it here."
        )
    with open(path) as f:
        return json.load(f)


def _load_lookup_csv(path: Path, key_col: str, value_cols: list) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df[key_col] = df[key_col].str.strip()
    if len(value_cols) == 1:
        return dict(zip(df[key_col], df[value_cols[0]].fillna("")))
    return {
        row[key_col]: [str(row[c]).strip() for c in value_cols if pd.notna(row.get(c))]
        for _, row in df.iterrows()
    }


@lru_cache(maxsize=1)
def get_artifacts() -> ModelArtifacts:
    model_path = MODEL_DIR / MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}. Train v1 in Colab, download the "
            f".joblib artifact, and place it at services/disease-diagnosis/models/{MODEL_FILENAME}."
        )

    model = joblib.load(model_path)
    vocab = _load_json_list(MODEL_DIR / "symptom_vocab.json")
    label_classes = _load_json_list(MODEL_DIR / "label_classes.json")
    descriptions = _load_lookup_csv(MODEL_DIR / "symptom_description.csv", "Disease", ["Description"])
    precautions = _load_lookup_csv(
        MODEL_DIR / "symptom_precaution.csv", "Disease",
        ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"],
    )

    return ModelArtifacts(model, vocab, label_classes, descriptions, precautions)
