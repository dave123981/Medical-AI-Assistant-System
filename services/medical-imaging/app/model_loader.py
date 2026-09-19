"""
Loads whichever image-classification model is active for a given image
type (chest_xray, skin_lesion, retinal), each living in its own subfolder
under MODEL_DIR so adding a new image type later is just "add a folder,"
not a service restructure.

"""
import json
import os
from functools import lru_cache
from pathlib import Path

MODEL_DIR = Path(os.getenv("IMAGING_MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
MODEL_EXTENSIONS = {".keras", ".h5"}

SUPPORTED_IMAGE_TYPES = {"chest_xray", "skin_lesion", "retinal"}


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
    def __init__(self, model, image_type: str, condition_names: list, version: str, input_size: tuple):
        self.model = model
        self.image_type = image_type
        self.condition_names = condition_names
        self.version = version
        self.input_size = input_size  # (height, width) — read from the model itself

    def predict_proba(self, batch) -> "list[float]":
        """Returns a flat list of per-condition probabilities for one image."""
        return self.model.predict(batch, verbose=0)[0].tolist()


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

    # (height, width) from the model's own input shape — e.g. (None, 224, 224, 3)
    # -> (224, 224). Preprocessing resizes to this dynamically rather than
    # hardcoding a size, so swapping in a model trained at a different
    # resolution just works.
    input_shape = model.input_shape
    input_size = (input_shape[1], input_shape[2])

    return ImagingModelArtifacts(model, image_type, condition_names, model_version, input_size)
