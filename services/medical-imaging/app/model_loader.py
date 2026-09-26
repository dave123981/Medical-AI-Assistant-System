import json
import os
from functools import lru_cache
from pathlib import Path

from app.gradcam import find_last_conv_layer_name

MODEL_DIR = Path(os.getenv("IMAGING_MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
MODEL_EXTENSIONS = {".keras", ".h5"}

SUPPORTED_IMAGE_TYPES = {"chest_xray", "skin_lesion", "retinal"}
DEFAULT_THRESHOLD = 0.5


class UnsupportedImageTypeError(Exception):
    """Raised when image_type isn't one of the three the service knows about at all."""
    pass


class ModelDiscoveryError(Exception):
    """
    Raised when auto-detecting which model file to load for an image type
    fails — no candidate file in that image type's folder (often because
    that image type just hasn't been built yet), or more than one exists.
    """
    pass


class ArtifactMismatchError(Exception):
    """
    Raised when the model's output width doesn't match condition_names.json's
    length — same class of bug as Service 1: files from different training
    runs got mixed together.
    """
    pass


class ModelLoadError(Exception):
    """Raised when the model file itself fails to load for any reason."""
    pass


class ImagingModelArtifacts:
    def __init__(self, model, image_type: str, condition_names: list, version: str,
                 input_size: tuple, per_class_thresholds: dict,
                 gradcam_capable: bool, base_model=None, gap_layer=None,
                 dense_layer=None, last_conv_layer_name=None):
        self.model = model
        self.image_type = image_type
        self.condition_names = condition_names
        self.version = version
        self.input_size = input_size  # (height, width) — read from the model itself
        self.per_class_thresholds = per_class_thresholds

        self.gradcam_capable = gradcam_capable
        self.base_model = base_model
        self.gap_layer = gap_layer
        self.dense_layer = dense_layer
        self.last_conv_layer_name = last_conv_layer_name

    def predict_proba(self, batch) -> "list[float]":
        """Returns a flat list of per-condition probabilities for one image."""
        return self.model.predict(batch, verbose=0)[0].tolist()

    def get_threshold(self, condition: str) -> float:
        return self.per_class_thresholds.get(condition, DEFAULT_THRESHOLD)


def _resolve_model_filename(image_type: str) -> str:
    env_var = f"{image_type.upper()}_MODEL_FILENAME"
    explicit = os.getenv(env_var)
    if explicit:
        return explicit

    type_dir = MODEL_DIR / image_type
    if not type_dir.exists():
        raise ModelDiscoveryError(
            f"No models/{image_type}/ folder found. This image type hasn't "
            f"been built yet, or the folder needs to be created."
        )

    candidates = sorted(
        f.name for f in type_dir.iterdir()
        if f.is_file() and f.suffix.lower() in MODEL_EXTENSIONS
    )

    if len(candidates) == 0:
        raise ModelDiscoveryError(
            f"No model file (.keras or .h5) found in models/{image_type}/. "
            f"Train a model in Colab and place the exported artifact here."
        )

    if len(candidates) > 1:
        raise ModelDiscoveryError(
            f"Found {len(candidates)} model files in models/{image_type}/, so "
            f"it's ambiguous which to serve: {', '.join(candidates)}. Remove "
            f"all but one, or set {env_var} to name the one you want."
        )

    return candidates[0]


def _infer_version_from_filename(filename: str) -> str:
    return Path(filename).stem.replace("_", "-")


def _resolve_model_version(image_type: str, filename: str) -> str:
    env_var = f"{image_type.upper()}_MODEL_VERSION"
    return os.getenv(env_var) or _infer_version_from_filename(filename)


def _load_per_class_thresholds(image_type: str) -> dict:
    path = MODEL_DIR / image_type / "per_class_thresholds.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _detect_gradcam_capability(model):
    """
    Looks for the shape a transfer-learning model built like v2+ has: a
    nested pretrained backbone (Functional sub-layer), a
    GlobalAveragePooling2D, and a Dense output layer. Returns
    (capable, base_model, gap_layer, dense_layer, last_conv_layer_name) —
    capable=False (with the rest None) for any model that doesn't match,
    e.g. v1's plain from-scratch CNN.
    """
    try:
        base_model = next(l for l in model.layers if l.__class__.__name__ == "Functional")
        gap_layer = next(l for l in model.layers if l.__class__.__name__ == "GlobalAveragePooling2D")
        dense_layer = next(l for l in model.layers if l.__class__.__name__ == "Dense")
        last_conv_layer_name = find_last_conv_layer_name(base_model)
        if last_conv_layer_name is None:
            return False, None, None, None, None
        return True, base_model, gap_layer, dense_layer, last_conv_layer_name
    except StopIteration:
        return False, None, None, None, None


@lru_cache(maxsize=None)
def get_artifacts(image_type: str) -> ImagingModelArtifacts:
    if image_type not in SUPPORTED_IMAGE_TYPES:
        raise UnsupportedImageTypeError(
            f"'{image_type}' is not a recognized image type. "
            f"Supported: {', '.join(sorted(SUPPORTED_IMAGE_TYPES))}."
        )

    model_filename = _resolve_model_filename(image_type)
    model_version = _resolve_model_version(image_type, model_filename)
    model_path = MODEL_DIR / image_type / model_filename

    try:
        from tensorflow import keras
        model = keras.models.load_model(model_path)
    except ModuleNotFoundError as e:
        raise ModelLoadError(
            f"Failed to load {model_path.name}: missing package '{e.name}'. "
            f"Image models require TensorFlow — pip install tensorflow."
        ) from None
    except Exception as e:
        raise ModelLoadError(f"Failed to load {model_path.name}: {e}") from None

    condition_names_path = MODEL_DIR / image_type / "condition_names.json"
    if not condition_names_path.exists():
        raise ModelDiscoveryError(
            f"Missing condition_names.json in models/{image_type}/. Export it "
            f"alongside the model from the training notebook."
        )
    with open(condition_names_path) as f:
        condition_names = json.load(f)

    output_width = model.output_shape[-1]
    if output_width != len(condition_names):
        raise ArtifactMismatchError(
            f"Model outputs {output_width} conditions but condition_names.json "
            f"has {len(condition_names)} entries for image type '{image_type}'. "
            f"These likely came from different training runs — re-export both "
            f"together from the same notebook run."
        )

    per_class_thresholds = _load_per_class_thresholds(image_type)
    if per_class_thresholds:
        missing = [c for c in condition_names if c not in per_class_thresholds]
        if missing:
            raise ArtifactMismatchError(
                f"per_class_thresholds.json is missing entries for: {', '.join(missing)}. "
                f"It must cover every condition in condition_names.json, or be absent "
                f"entirely (in which case all conditions fall back to {DEFAULT_THRESHOLD})."
            )

    gradcam_capable, base_model, gap_layer, dense_layer, last_conv_layer_name = \
        _detect_gradcam_capability(model)

    input_shape = model.input_shape
    input_size = (input_shape[1], input_shape[2])

    return ImagingModelArtifacts(
        model, image_type, condition_names, model_version, input_size,
        per_class_thresholds, gradcam_capable, base_model, gap_layer,
        dense_layer, last_conv_layer_name,
    )
