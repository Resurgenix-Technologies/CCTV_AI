"""Lightweight enhancement for small/noisy face-region crops, applied
before running detection and recognition on them."""

from __future__ import annotations

import cv2
import numpy as np


# ==================== Optional Super-Resolution (FSRCNN) ====================
# If configured (via configure_superres), face crops get AI-based upscaling
# instead of plain cubic interpolation - noticeably sharper detail recovery
# for small/distant CCTV faces. If not configured, everything still works,
# just falls back to cubic resize.

_sr = None


def configure_superres(model_path: str, model_name: str = "fsrcnn", scale: int = 3) -> bool:
    """
    Load an OpenCV dnn_superres model (e.g. FSRCNN_x3.pb). Returns True on
    success. Safe to call once at startup; if it fails (missing file,
    missing opencv-contrib module), the pipeline silently keeps using
    cubic interpolation instead.
    """
    global _sr
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel(model_name, scale)
        _sr = sr
        print(f"Super-resolution enabled: {model_name} x{scale} ({model_path})")
        return True
    except Exception as exc:
        print(f"Super-resolution NOT enabled ({exc}). Falling back to cubic resize.")
        _sr = None
        return False


def upscale_face_crop(bgr_image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    Upscale a face-region crop to (target_w, target_h). Uses the
    configured super-resolution model if available (then resizes to the
    exact target size, since SR models only support fixed integer
    scales), otherwise falls back to high-quality cubic interpolation.
    """
    if _sr is not None:
        try:
            upscaled = _sr.upsample(bgr_image)
            return cv2.resize(upscaled, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        except Exception:
            pass  # fall through to plain cubic resize below

    return cv2.resize(bgr_image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


# ==================== Auto Gamma Correction ====================
# CCTV footage is very often under-lit (corners, night, poor fixtures).
# Dark faces hurt both detection confidence and embedding quality. This
# brightens only frames that are actually dark - well-lit frames pass
# through basically unchanged.

def _auto_gamma_correct(bgr_image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray)) / 255.0
    mean_brightness = max(mean_brightness, 1e-3)

    target = 0.45  # desired mid-brightness
    gamma = np.log(target) / np.log(mean_brightness)
    # Clip: only ever brighten (gamma < 1), never darken an already-bright frame.
    gamma = float(np.clip(gamma, 0.4, 1.0))

    if abs(gamma - 1.0) < 0.03:
        return bgr_image  # already bright enough, skip the LUT op

    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype(np.uint8)
    return cv2.LUT(bgr_image, table)


def enhance_face_region(bgr_image: np.ndarray) -> np.ndarray:
    """
    Gamma-correct, denoise, contrast-normalize, and sharpen a face-region
    crop. Returns an image of the SAME dimensions as the input.
    """

    if bgr_image is None or bgr_image.size == 0:
        return bgr_image

    gamma_corrected = _auto_gamma_correct(bgr_image)

    # Use fast bilateral filter for noise reduction while keeping facial edges sharp
    denoised = cv2.bilateralFilter(gamma_corrected, d=5, sigmaColor=35, sigmaSpace=35)

    # CLAHE in YCrCb luminance space
    ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    y_equalized = clahe.apply(y_channel)
    ycrcb_equalized = cv2.merge((y_equalized, cr_channel, cb_channel))
    contrast_enhanced = cv2.cvtColor(ycrcb_equalized, cv2.COLOR_YCrCb2BGR)

    # Unsharp mask for crisp eye/nose/mouth edges
    blurred = cv2.GaussianBlur(contrast_enhanced, (0, 0), sigmaX=2.5)
    sharpened = cv2.addWeighted(contrast_enhanced, 1.4, blurred, -0.4, 0)

    return sharpened


# ==================== Test-Time Augmentation (TTA) ====================
# Generates a handful of slightly perturbed copies of an already-enhanced
# face crop (brightness up/down, small rotations, horizontal flip). Each
# variant is independently detected + embedded elsewhere in the pipeline,
# and the resulting embeddings are averaged - this smooths out the
# per-frame sensitivity a single embedding has to exact lighting/angle,
# which matters a lot on small/noisy CCTV faces.

def generate_tta_variants(bgr_image: np.ndarray) -> list[np.ndarray]:
    if bgr_image is None or bgr_image.size == 0:
        return []

    variants: list[np.ndarray] = []

    # Brighter / darker versions cover borderline exposure.
    variants.append(cv2.convertScaleAbs(bgr_image, alpha=1.15, beta=12))
    variants.append(cv2.convertScaleAbs(bgr_image, alpha=0.85, beta=-12))

    # Horizontal flip - faces are roughly bilaterally symmetric, and
    # flip-and-average is a standard accuracy trick for face embedding
    # models (many are trained with random horizontal-flip augmentation).
    variants.append(cv2.flip(bgr_image, 1))

    # Small rotations cover slight head tilt / camera mounting angle.
    h, w = bgr_image.shape[:2]
    center = (w / 2, h / 2)
    for angle in (-5, 5):
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            bgr_image, rot_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE
        )
        variants.append(rotated)

    return variants


# ==================== Enrollment Domain Augmentation ====================
# Clean enrollment photos (phone/webcam quality) live in a different
# visual "domain" than live CCTV footage (noisy, blurry, dim, distant,
# compressed). Rather than just hoping the enhancement pipeline bridges
# that gap at inference time, we ALSO create synthetic CCTV-like copies
# of each enrollment photo, and enroll THEM too (same person, additional
# embeddings). This directly narrows the domain gap using nothing but
# the photos you already have - no extra real-world data collection
# needed.

def generate_cctv_domain_variants(bgr_image: np.ndarray) -> list[np.ndarray]:
    """
    Returns a few synthetically degraded copies of a clean photo, each
    simulating a different way real CCTV footage commonly looks worse
    than a clean enrollment photo:
      - motion blur + sensor noise (handheld/compressed footage)
      - a downsample/upsample round-trip (a distant/low-res camera)
      - underexposed + slightly desaturated (poor CCTV lighting)
    """
    if bgr_image is None or bgr_image.size == 0:
        return []

    h, w = bgr_image.shape[:2]
    variants = []

    blurred = cv2.GaussianBlur(bgr_image, (5, 5), sigmaX=1.5)
    noise = np.random.normal(0, 8, blurred.shape).astype(np.float32)
    noisy = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    variants.append(noisy)

    small_w, small_h = max(1, w // 3), max(1, h // 3)
    small = cv2.resize(bgr_image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    variants.append(restored)

    darker = cv2.convertScaleAbs(bgr_image, alpha=0.6, beta=-15)
    variants.append(darker)

    return variants