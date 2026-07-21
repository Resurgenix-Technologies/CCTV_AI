"""
build_face_registry.py

Builds the FACE registry for a given experiment mode (baseline or
augmented). Reads the face_folders list produced by build_persons.py
and iterates over every folder in that list — one embedding per usable
image, no matter which folder it came from. Embeddings are NEVER
averaged together.

Usage:
    python build_face_registry.py --mode baseline
    python build_face_registry.py --mode augmented

    # or explicit paths:
    python build_face_registry.py --persons data/persons_baseline.json --output registry/baseline/face
"""

import os
import json
import argparse
import numpy as np
import cv2
from insightface.app import FaceAnalysis


def resolve_args():
    parser = argparse.ArgumentParser(description="Build the face registry.")
    parser.add_argument("--mode", choices=["baseline", "augmented"], default=None,
                         help="Shortcut that sets --persons/--output defaults.")
    parser.add_argument("--persons", default=None, help="Path to persons.json")
    parser.add_argument("--output", default=None, help="Output directory for <ID>.npy files")
    args = parser.parse_args()

    mode = args.mode or "baseline"
    persons_path = args.persons or os.path.join("data", f"persons_{mode}.json")
    output_dir = args.output or os.path.join("registry", mode, "face")

    return persons_path, output_dir


def main():
    persons_path, output_dir = resolve_args()
    os.makedirs(output_dir, exist_ok=True)

    with open(persons_path, "r") as f:
        persons = json.load(f)

    print("Loading InsightFace...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.4)
    print(f"InsightFace ready. Building registry from {persons_path} -> {output_dir}\n")

    for global_id, info in persons.items():
        name = info["name"]
        folders = info.get("face_folders", [])

        if not folders:
            print(f"[SKIP] {global_id} ({name}): no face folders listed")
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

                faces = app.get(img)

                if len(faces) == 0:
                    print(f"  [WARN] no face found in {img_path}")
                    continue

                # If multiple faces appear, assume the largest is the subject.
                faces.sort(
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    reverse=True,
                )
                embeddings.append(faces[0].embedding)

        if len(embeddings) == 0:
            print(f"[FAIL] {global_id} ({name}): no usable face embeddings, skipping")
            continue

        # Stack -> shape (N, 512). Do NOT average these rows together.
        embeddings = np.stack(embeddings, axis=0)

        out_path = os.path.join(output_dir, f"{global_id}.npy")
        np.save(out_path, embeddings)

        print(f"[OK] {global_id} ({name}): stored {embeddings.shape[0]} face embeddings")

    print("\nFace registry build complete.")


if __name__ == "__main__":
    main()