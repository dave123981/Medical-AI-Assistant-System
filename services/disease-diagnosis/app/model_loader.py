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
import numpy as np
import pandas as pd

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
MODEL_EXTENSIONS = {".joblib", ".pkl", ".keras"}


class ArtifactMismatchError(Exception):
    """
    Raised when the loaded model, symptom_vocab.json, and label_classes.json
    don't agree on input/output size — almost always because they came from
    different training runs and got mixed together in models/.
    """
    pass


class ModelLoadError(Exception):
    """
    Raised when the model file itself fails to load — wrong/missing
    package for the model type, corrupted file, incompatible version,
    etc. Wraps the original exception's message so the real cause is
    still visible, instead of letting an unrelated exception type
    (ModuleNotFoundError, pickle errors, etc.) escape as a raw 500.
    """
    pass


class ModelDiscoveryError(Exception):
    """
    Raised when auto-detecting which model file to load fails — either
    no candidate files exist in MODEL_DIR, or more than one does and
    it's ambiguous which one should be served.
    """
    pass


def resolve_model_filename() -> str:
    """
    Decides which model file to load, in priority order:

      1. MODEL_FILENAME env var, if set — always wins, use this to force
         a specific model when multiple sit in the folder at once.
      2. Auto-detected from MODEL_DIR — if exactly one .joblib/.pkl/.keras
         file exists there, use it. This is the common case: you only
         ever have one model version's file in models/ at a time.

    Deliberately NOT cached at import time — models/ can change between
    server restarts (or even mid-session if you're swapping files by
    hand), and this is cheap enough to just re-scan.
    """
    explicit = os.getenv("MODEL_FILENAME")
    if explicit:
        return explicit

    if not MODEL_DIR.exists():
        raise ModelDiscoveryError(
            f"MODEL_DIR does not exist: {MODEL_DIR}. Create it and place a "
            f"trained model artifact inside, or set MODEL_DIR to point "
            f"somewhere that exists."
        )

    candidates = sorted(
        f.name for f in MODEL_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in MODEL_EXTENSIONS
    )

    if len(candidates) == 0:
        raise ModelDiscoveryError(
            f"No model file (.joblib, .pkl, or .keras) found in {MODEL_DIR}. "
            f"Train a model in Colab and place the exported artifact here."
        )

    if len(candidates) > 1:
        raise ModelDiscoveryError(
            f"Found {len(candidates)} model files in {MODEL_DIR}, so it's "
            f"ambiguous which one to serve: {', '.join(candidates)}. "
            f"Either remove all but one, or set the MODEL_FILENAME env var "
            f"to name the one you want (e.g. MODEL_FILENAME={candidates[0]})."
        )

    return candidates[0]


def _infer_version_from_filename(filename: str) -> str:
    """
    Turns "v4_neural_network.keras" into "v4-neural-network" so the API
    response always reflects the model actually loaded, even if nobody
    remembered to set MODEL_VERSION explicitly.
    """
    stem = Path(filename).stem
    return stem.replace("_", "-")


def resolve_model_version(model_filename: str) -> str:
    """
    MODEL_VERSION env var wins if set (e.g. to label an experimental
    build "v4-neural-network-tuned"); otherwise derived from whichever
    filename was actually resolved, so it can't silently go stale.
    """
    return os.getenv("MODEL_VERSION") or _infer_version_from_filename(model_filename)


class ModelArtifacts:
    def __init__(self, model, model_type, vocab, label_classes, descriptions, precautions, version):
        self.model = model
        self.model_type = model_type  # "sklearn" or "keras"
        self.vocab = vocab
        self.label_classes = label_classes
        self.descriptions = descriptions  # dict: disease -> description
        self.precautions = precautions    # dict: disease -> list[str]
        self.version = version

    def predict_proba(self, vector: np.ndarray) -> np.ndarray:
        """
        Returns a (1, num_classes) probability array regardless of the
        underlying model type, so callers never branch on model_type.
        """
        if self.model_type == "keras":
            return self.model.predict(vector, verbose=0)
        return self.model.predict_proba(vector)


def _load_json_list(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing '{path.name}' at {path}. Train a model in Colab "
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


def _validate_artifacts(model, model_type: str, vocab: list, label_classes: list) -> None:
    """
    Confirms the model's expected input width matches len(vocab) and its
    output width matches len(label_classes), BEFORE any request is served.
    Catching this at load time turns a confusing mid-request IndexError
    into a clear, actionable startup error.
    """
    if model_type == "keras":
        input_width = model.input_shape[-1]
        output_width = model.output_shape[-1]
    else:
        input_width = getattr(model, "n_features_in_", None)
        output_width = len(getattr(model, "classes_", [])) or None

    mismatch_msg = (
        "\n\nThis almost always means the model file, symptom_vocab.json, and "
        "label_classes.json in models/ came from DIFFERENT training runs and "
        "got mixed together. Fix: re-extract all three files from the SAME "
        "export zip for the version you're deploying, overwriting whatever "
        "was in models/ before — don't keep one file from an older version."
    )

    if input_width is not None and input_width != len(vocab):
        raise ArtifactMismatchError(
            f"Model expects {input_width} input features but symptom_vocab.json "
            f"has {len(vocab)} entries.{mismatch_msg}"
        )

    if output_width is not None and output_width != len(label_classes):
        raise ArtifactMismatchError(
            f"Model outputs {output_width} classes but label_classes.json has "
            f"{len(label_classes)} entries.{mismatch_msg}"
        )


@lru_cache(maxsize=1)
def get_artifacts() -> ModelArtifacts:
    model_filename = resolve_model_filename()
    model_version = resolve_model_version(model_filename)
    model_path = MODEL_DIR / model_filename

    if not model_path.exists():
        # Shouldn't normally happen since resolve_model_filename() only
        # returns names it found on disk, but MODEL_FILENAME could still
        # point at something nonexistent if set explicitly and wrong.
        raise FileNotFoundError(
            f"No model found at {model_path}. Train a model in Colab, download "
            f"the artifact, and place it at services/disease-diagnosis/models/{model_filename}."
        )

    suffix = model_path.suffix.lower()
    try:
        if suffix == ".keras":
            # Imported lazily so services running v1-v3 don't need TensorFlow
            # installed at all — it's only pulled in if a .keras file is
            # actually being loaded.
            from tensorflow import keras
            model = keras.models.load_model(model_path)
            model_type = "keras"
        elif suffix in (".joblib", ".pkl"):
            model = joblib.load(model_path)
            model_type = "sklearn"
        else:
            raise ValueError(
                f"Unsupported model file extension '{suffix}' for {model_path}. "
                f"Expected .joblib, .pkl, or .keras."
            )
    except ModuleNotFoundError as e:
        raise ModelLoadError(
            f"Failed to load {model_path.name}: missing package '{e.name}'. "
            f".joblib files store a reference to the library used to train "
            f"them (e.g. xgboost for v3, tensorflow for v4) — that library "
            f"must be installed even though your code never imports it "
            f"directly. Fix: pip install {e.name}, then add it to requirements.txt."
        ) from None
    except Exception as e:
        raise ModelLoadError(f"Failed to load {model_path.name}: {e}") from None

    vocab = _load_json_list(MODEL_DIR / "symptom_vocab.json")
    label_classes = _load_json_list(MODEL_DIR / "label_classes.json")
    descriptions = _load_lookup_csv(MODEL_DIR / "symptom_description.csv", "Disease", ["Description"])
    precautions = _load_lookup_csv(
        MODEL_DIR / "symptom_precaution.csv", "Disease",
        ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"],
    )

    _validate_artifacts(model, model_type, vocab, label_classes)

    return ModelArtifacts(model, model_type, vocab, label_classes, descriptions, precautions, model_version)
