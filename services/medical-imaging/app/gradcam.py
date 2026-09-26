"""
Grad-CAM heatmap generation, verified working through direct testing before
being ported here (including the specific "not connected to inputs" Keras
nested-sub-model gotcha this implementation works around).

This module has no dependency on any specific model version — it operates
on whatever base_model/gap_layer/dense_layer a model exposes, via
model_loader.detect_gradcam_capability(). Models that don't match this
transfer-learning-with-a-pretrained-backbone shape (e.g. v1's from-scratch
CNN) simply don't get Grad-CAM support, and the service degrades to
heatmap_base64=None for them rather than erroring.
"""
import base64
import io

import numpy as np
import tensorflow as tf
from PIL import Image


def find_last_conv_layer_name(base_model) -> "str | None":
    """
    DenseNet121's final feature layer is named 'relu' (verified). Falls
    back to searching for the last 4D-output layer for other backbones,
    rather than assuming every architecture uses the same name.
    """
    layer_names = [l.name for l in base_model.layers]
    if "relu" in layer_names:
        return "relu"

    for layer in reversed(base_model.layers):
        try:
            shape = layer.output.shape
        except Exception:
            continue
        if len(shape) == 4:
            return layer.name

    return None


def make_gradcam_heatmap(img_array, base_model, gap_layer, dense_layer, class_index, last_conv_layer_name):
    """
    Built from base_model's OWN standalone input/output graph, then
    manually re-applies the head layers inside the same GradientTape.
    Building this from the outer wrapping model's layer.output directly
    raises 'not connected to inputs' — confirmed during development, not
    a hypothetical concern.
    """
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, base_features = grad_model(img_array)
        pooled = gap_layer(base_features)
        predictions = dense_layer(pooled)
        class_output = predictions[:, class_index]

    grads = tape.gradient(class_output, conv_output)
    if grads is None:
        return None
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_image_array: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> str:
    """
    original_image_array: (H, W, 3) float32 in [0, 1] — the same array
    preprocessing.py produced for the model's own input.
    Returns a base64-encoded PNG string. Verified to round-trip to a
    valid, decodable image.
    """
    h, w = original_image_array.shape[:2]

    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize((w, h), resample=Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img) / 255.0

    heatmap_colored = np.zeros((h, w, 3), dtype=np.float32)
    heatmap_colored[..., 0] = heatmap_resized  # simple red-intensity overlay

    original_uint8 = (original_image_array * 255).astype(np.float32)
    overlay = original_uint8 * (1 - alpha) + heatmap_colored * 255 * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def compute_gradcam_overlay(artifacts, img_array_batch: np.ndarray, class_index: int) -> "str | None":
    """
    Single entry point analyze.py calls. Returns None (never raises) if
    the loaded model doesn't support Grad-CAM — a missing heatmap should
    never break a prediction request.
    """
    if not artifacts.gradcam_capable:
        return None

    try:
        heatmap = make_gradcam_heatmap(
            img_array_batch, artifacts.base_model, artifacts.gap_layer,
            artifacts.dense_layer, class_index, artifacts.last_conv_layer_name,
        )
        if heatmap is None:
            return None
        return overlay_heatmap(img_array_batch[0], heatmap)
    except Exception:
        # Grad-CAM is a nice-to-have visualization, not a critical path —
        # a failure here should degrade to no heatmap, not fail the request.
        return None
