"""
Turns an uploaded image file into the batch the model expects.

Resizes dynamically to whatever input size the loaded model actually
wants (read from model.input_shape in model_loader.py) rather than a
hardcoded constant, so swapping in a model trained at a different
resolution doesn't require touching this file.
"""
import io
from typing import Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError


class InvalidImageError(Exception):
    """Raised when the uploaded file isn't a readable image."""
    pass


def preprocess_image(file_bytes: bytes, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Returns a (1, height, width, 3) float32 array scaled to [0, 1],
    ready to pass straight into model.predict().
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image = image.convert("RGB")  # handles grayscale X-rays and RGBA uploads alike
    except UnidentifiedImageError:
        raise InvalidImageError(
            "Could not read the uploaded file as an image. "
            "Supported formats: JPEG, PNG."
        ) from None

    image = image.resize((target_size[1], target_size[0]))  # PIL wants (width, height)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)
