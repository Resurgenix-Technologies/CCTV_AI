"""
build_body_registry.py

Builds the BODY / person-ReID registry for a given experiment mode
(baseline or augmented). Reads the body_folders list produced by
build_persons.py and iterates over every folder in that list — one
embedding per usable image. Embeddings are NEVER averaged together,
and this registry is never combined with the face embedding space.

Usage:
    python build_body_registry.py --mode baseline
    python build_body_registry.py --mode augmented

    # or explicit paths:
    python build_body_registry.py --persons data/persons_baseline.json --output registry/baseline/body
"""

import os
import json
import argparse
import numpy as np
import cv2
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor

YOLO_WEIGHTS = "yolov8n.pt"
PERSON_DETECT_CONF = 0.4
REID_MODEL_NAME = "osnet_x1_0"
REID_MODEL_PATH = ""     # "" -> torchreid uses its own pretrained weights
DEVICE = "cpu"            # change to "cuda" if a GPU is available


def resolve_args():
    parser = argparse.ArgumentParser(description="Build the body/ReID registry.")
    parser.add_argument("--mode", choices=["baseline", "augmented"], default=None,
                         help="Shortcut that sets --persons/--output defaults.")
    parser.add_argument("--persons", default=None, help="Path to persons.json")
    parser.add_argument("--output", default=None, help="Output directory for <ID>.npy files")
    args = parser.parse_args()

    mode = args.mode or "baseline"
    persons_path = args.persons or os.path.join("data", f"persons_{mode}.json")
    output_dir = args.output or os.path.join("registry", mode, "body")

    return persons_path, output_dir


def largest_person_crop(detector, img):
    results = detector.predict(
        img, classes=[0], conf=PERSON_DETECT_CONF, device=DEVICE, verbose=False
    )

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    if len(xyxy) > 1:
        print(f"  [WARN] {len(xyxy)} people detected in registration image, using the largest box")

    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    best = xyxy[int(np.argmax(areas))]

    x1, y1, x2, y2 = map(int, best)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]


def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def main():
    persons_path, output_dir = resolve_args()
    os.makedirs(output_dir, exist_ok=True)

    with open(persons_path, "r") as f:
        persons = json.load(f)

    print("Loading person detector (YOLOv8)...")
    detector = YOLO(YOLO_WEIGHTS)

    print("Loading OSNet feature extractor...")
    extractor = FeatureExtractor(
        model_name=REID_MODEL_NAME, model_path=REID_MODEL_PATH, device=DEVICE
    )
    print(f"Ready. Building registry from {persons_path} -> {output_dir}\n")

    for global_id, info in persons.items():
        name = info["name"]
        folders = info.get("body_folders", [])

        if not folders:
            print(f"[SKIP] {global_id} ({name}): no body folders listed")
            continue

        embeddings = []

        for folder in folders:
            if not os.path.isdir(folder):
                print(f"  [WARN] folder not found -> {folder}")
                continue

            for fname in sorted(os.listdir(folder)):
                img_path = os.path.join(folder, fname)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"  [WARN] could not read {img_path}")
                    continue

                crop = largest_person_crop(detector, img)
                if crop is None:
                    print(f"  [WARN] no person detected in {img_path}")
                    continue

                feats = extractor([crop])
                embedding = feats[0]
                embedding = embedding.cpu().numpy() if hasattr(embedding, "cpu") else np.array(embedding)
                embedding = l2_normalize(embedding)

                embeddings.append(embedding)

        if len(embeddings) == 0:
            print(f"[FAIL] {global_id} ({name}): no usable body embeddings, skipping")
            continue

        # Stack -> shape (N, D). Do NOT average these rows together.
        embeddings = np.stack(embeddings, axis=0)

        out_path = os.path.join(output_dir, f"{global_id}.npy")
        np.save(out_path, embeddings)

        print(f"[OK] {global_id} ({name}): stored {embeddings.shape[0]} body embeddings")

    print("\nBody registry build complete.")


if __name__ == "__main__":
    main()