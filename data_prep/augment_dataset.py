"""
augment_dataset.py

Generates a SMALL number of mild, CCTV-realistic augmented variants of
each face/body reference image, so we can later benchmark whether
"original only" registration beats "original + augmented" registration.

Does NOT touch test_images/ — everything is written to a separate
augmented_test_images/ tree. Does NOT run InsightFace or OSNet: this
script only produces images. Embedding generation stays the
responsibility of build_face_registry.py / build_body_registry.py.

Face images are augmented as-is (no crop, no forced resize, no
grayscale conversion — InsightFace still needs to do its own detection
and alignment on the result).

Body images are first passed through the same YOLO person-detection
logic used in build_body_registry.py so the crop doesn't include an
unrealistic amount of background, then augmented. The face/head is
NOT removed or blacked out from the body crop.

Usage:
    python augment_dataset.py
"""

import os
import json
import random
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# CONFIG
# ============================================================

SOURCE_ROOT = "test_images"
OUTPUT_ROOT = "augmented_test_images"
MANIFEST_PATH = os.path.join(OUTPUT_ROOT, "augmentation_manifest.json")

MAX_AUGMENTATIONS_PER_IMAGE = 3
RANDOM_SEED = 42

YOLO_WEIGHTS = "yolov8n.pt"
PERSON_CONFIDENCE = 0.4
BODY_CROP_PADDING = 0.05   # fraction of box width/height added on each side
YOLO_DEVICE = "cpu"        # current dev environment is CPU-only

VALID_EXT = (".jpg", ".jpeg", ".png")

ENABLE_BRIGHTNESS = True
ENABLE_CONTRAST = True
ENABLE_BLUR = True
ENABLE_JPEG = True
ENABLE_NOISE = True
ENABLE_ROTATION = True
ENABLE_HORIZONTAL_FLIP = True

BRIGHTNESS_RANGE = (-0.20, 0.20)   # +/- 20%
CONTRAST_RANGE = (-0.20, 0.20)     # +/- 20%
BLUR_KERNELS = (3, 5)              # small kernels only
JPEG_QUALITY_RANGE = (60, 85)
NOISE_STD_RANGE = (3, 10)          # gaussian noise std, 0-255 pixel scale
ROTATION_RANGE_DEG = (-5, 5)       # small rotation only

TYPE_LABELS = {
    "brightness": "brightness",
    "contrast": "contrast",
    "blur": "gaussian_blur",
    "jpeg": "jpeg_compression",
    "noise": "gaussian_noise",
    "rotation": "rotation",
    "hflip": "horizontal_flip",
}


# ============================================================
# AUGMENTATION FUNCTIONS
# Each takes (img_bgr, rng) and returns (augmented_img, params_dict)
# ============================================================

def aug_brightness(img, rng):
    factor = 1.0 + rng.uniform(*BRIGHTNESS_RANGE)
    out = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return out, {"factor": round(factor, 3)}


def aug_contrast(img, rng):
    factor = 1.0 + rng.uniform(*CONTRAST_RANGE)
    mean = float(img.astype(np.float32).mean())
    out = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
    return out, {"factor": round(factor, 3)}


def aug_blur(img, rng):
    k = rng.choice(BLUR_KERNELS)
    out = cv2.GaussianBlur(img, (k, k), 0)
    return out, {"kernel_size": k}


def aug_jpeg(img, rng):
    quality = rng.randint(*JPEG_QUALITY_RANGE)
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img, {"quality": quality, "note": "encode_failed_used_original"}
    out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return out, {"quality": quality}


def aug_noise(img, rng):
    std = rng.uniform(*NOISE_STD_RANGE)
    noise = np.random.RandomState(rng.randint(0, 2**31 - 1)).normal(0, std, img.shape)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out, {"std": round(std, 2)}


def aug_rotation(img, rng):
    angle = rng.uniform(*ROTATION_RANGE_DEG)
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out, {"angle_deg": round(angle, 2)}


def aug_horizontal_flip(img, rng):
    return cv2.flip(img, 1), {}


AUGMENTATIONS = {}
if ENABLE_BRIGHTNESS:
    AUGMENTATIONS["brightness"] = aug_brightness
if ENABLE_CONTRAST:
    AUGMENTATIONS["contrast"] = aug_contrast
if ENABLE_BLUR:
    AUGMENTATIONS["blur"] = aug_blur
if ENABLE_JPEG:
    AUGMENTATIONS["jpeg"] = aug_jpeg
if ENABLE_NOISE:
    AUGMENTATIONS["noise"] = aug_noise
if ENABLE_ROTATION:
    AUGMENTATIONS["rotation"] = aug_rotation
if ENABLE_HORIZONTAL_FLIP:
    AUGMENTATIONS["hflip"] = aug_horizontal_flip


# ============================================================
# PERSON DETECTION (body crop only) — loaded once, lazily
# ============================================================

_detector = None


def get_detector():
    global _detector
    if _detector is None:
        from ultralytics import YOLO
        print("Loading person detector (YOLOv8, CPU)...")
        _detector = YOLO(YOLO_WEIGHTS)
    return _detector


def largest_person_crop(img, padding=BODY_CROP_PADDING):
    detector = get_detector()
    results = detector.predict(
        img, classes=[0], conf=PERSON_CONFIDENCE, device=YOLO_DEVICE, verbose=False
    )

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    if len(xyxy) > 1:
        print(f"  [WARN] {len(xyxy)} people detected, using the largest box")

    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    x1, y1, x2, y2 = xyxy[int(np.argmax(areas))]

    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * padding
    x2 += bw * padding
    y1 -= bh * padding
    y2 += bh * padding

    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]


# ============================================================
# DATASET WALK
# ============================================================

def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(VALID_EXT))


def unique_output_path(output_folder, stem, aug_key):
    out_name = f"{stem}_aug_{aug_key}.jpg"
    out_path = os.path.join(output_folder, out_name)
    counter = 2
    while os.path.exists(out_path):
        out_name = f"{stem}_aug_{aug_key}_{counter}.jpg"
        out_path = os.path.join(output_folder, out_name)
        counter += 1
    return out_path


def process_modality(identity, modality, source_folder, output_folder, rng, manifest, is_body):
    images = list_images(source_folder)
    if not images:
        return

    os.makedirs(output_folder, exist_ok=True)

    for fname in images:
        src_path = os.path.join(source_folder, fname)
        img = cv2.imread(src_path)

        if img is None:
            print(f"  [WARN] could not read {src_path}")
            continue

        base_img = img
        if is_body:
            crop = largest_person_crop(img)
            if crop is None:
                print(f"  [WARN] no person detected in {src_path}, skipping")
                continue
            base_img = crop

        stem = Path(fname).stem

        available_types = list(AUGMENTATIONS.keys())
        k = min(MAX_AUGMENTATIONS_PER_IMAGE, len(available_types))
        chosen = rng.sample(available_types, k=k)

        for aug_key in chosen:
            out_img, params = AUGMENTATIONS[aug_key](base_img, rng)
            out_path = unique_output_path(output_folder, stem, aug_key)

            cv2.imwrite(out_path, out_img)

            manifest.append({
                "identity": identity,
                "modality": modality,
                "source_image": src_path.replace("\\", "/"),
                "augmented_image": out_path.replace("\\", "/"),
                "augmentation_type": TYPE_LABELS[aug_key],
                "parameters": params,
            })

        print(f"  [OK] {fname}: generated {len(chosen)} augmentation(s)")


def main():
    rng = random.Random(RANDOM_SEED)
    manifest = []

    identities = sorted(
        d for d in os.listdir(SOURCE_ROOT)
        if os.path.isdir(os.path.join(SOURCE_ROOT, d))
    )
    print(f"Found {len(identities)} identities under {SOURCE_ROOT}\n")

    for identity in identities:
        print(f"Processing {identity}...")

        face_src = os.path.join(SOURCE_ROOT, identity, "face")
        face_out = os.path.join(OUTPUT_ROOT, identity, "face")
        process_modality(identity, "face", face_src, face_out, rng, manifest, is_body=False)

        body_src = os.path.join(SOURCE_ROOT, identity, "body")
        body_out = os.path.join(OUTPUT_ROOT, identity, "body")
        process_modality(identity, "body", body_src, body_out, rng, manifest, is_body=True)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGenerated {len(manifest)} augmented images.")
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()