"""Conservative low-light enhancement for detector input only."""

from __future__ import annotations

import cv2
import numpy as np


def lime_enhance(image: np.ndarray) -> np.ndarray:
    """Improve dark detector input while preserving the source image.

    This is a bounded LIME-style illumination correction. It is intentionally
    used only before face detection; SFace alignment and embedding extraction
    continue to use the original pixels.
    """

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("LIME enhancement requires a BGR image.")
    if image.size == 0:
        raise ValueError("LIME enhancement requires a non-empty image.")
    source = image.astype(np.float32) / 255.0
    illumination = np.max(source, axis=2)
    illumination = cv2.GaussianBlur(illumination, (0, 0), sigmaX=15)
    illumination = np.clip(illumination, 0.08, 1.0)
    # Gentle gamma avoids lifting noise as aggressively as histogram equalize.
    gain = np.power(illumination, -0.35)[..., None]
    enhanced = np.clip(source * gain, 0.0, 1.0)
    return np.ascontiguousarray(np.round(enhanced * 255.0).astype(np.uint8))


def enhance_ndi_frame(image: np.ndarray) -> np.ndarray:
    """Conservative LIME plus CLAHE enhancement for live NDI frames."""
    enhanced = lime_enhance(image)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    light, a_channel, b_channel = cv2.split(lab)
    light = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(light)
    return np.ascontiguousarray(cv2.cvtColor(
        cv2.merge((light, a_channel, b_channel)), cv2.COLOR_LAB2BGR
    ))
