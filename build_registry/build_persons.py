"""
build_persons.py

Discovers registered identities and writes a persons.json describing
where each identity's face/body reference images live.

Supports two modes so the "original-only" and "original+augmented"
experiments (see augment_dataset.py) can coexist without overwriting
each other:

  --mode baseline    -> face_folders/body_folders point ONLY at
                         test_images/<person>/{face,body}

  --mode augmented   -> face_folders/body_folders point at BOTH
                         test_images/<person>/{face,body}  AND
                         augmented_test_images/<person>/{face,body}

Each entry stores a LIST of folders. Registry builders iterate over
every folder in the list and store one embedding per image regardless
of which folder it came from — nothing is averaged here or downstream.

Usage:
    python build_persons.py --mode baseline
    python build_persons.py --mode augmented
"""

import os
import json
import argparse

VALID_EXT = (".jpg", ".jpeg", ".png")


def count_images(folder):
    if not os.path.isdir(folder):
        return 0
    return sum(1 for f in os.listdir(folder) if f.lower().endswith(VALID_EXT))


def existing_folders(*folders):
    """Only include folders that exist and contain at least one image."""
    return [f for f in folders if count_images(f) > 0]


def main():
    parser = argparse.ArgumentParser(description="Build persons.json for a given experiment mode.")
    parser.add_argument("--mode", choices=["baseline", "augmented"], default="baseline")
    parser.add_argument("--source-root", default="test_images",
                         help="Original dataset root (always included).")
    parser.add_argument("--augmented-root", default="augmented_test_images",
                         help="Augmented dataset root (only used when --mode augmented).")
    parser.add_argument("--output", default=None,
                         help="Output path. Defaults to data/persons_<mode>.json")
    args = parser.parse_args()

    output_path = args.output or os.path.join("data", f"persons_{args.mode}.json")

    persons = {}
    counter = 1

    for person_name in sorted(os.listdir(args.source_root)):
        person_path = os.path.join(args.source_root, person_name)
        if not os.path.isdir(person_path):
            continue

        base_face = os.path.join(args.source_root, person_name, "face")
        base_body = os.path.join(args.source_root, person_name, "body")

        if args.mode == "baseline":
            face_folders = existing_folders(base_face)
            body_folders = existing_folders(base_body)
        else:  # augmented
            aug_face = os.path.join(args.augmented_root, person_name, "face")
            aug_body = os.path.join(args.augmented_root, person_name, "body")
            face_folders = existing_folders(base_face, aug_face)
            body_folders = existing_folders(base_body, aug_body)

        if not face_folders and not body_folders:
            print(f"[SKIP] {person_name}: no usable face/body images found")
            continue

        global_id = f"P{counter:04d}"
        face_count = sum(count_images(f) for f in face_folders)
        body_count = sum(count_images(f) for f in body_folders)

        persons[global_id] = {
            "name": person_name,
            "face_folders": face_folders,
            "body_folders": body_folders,
            "face_image_count": face_count,
            "body_image_count": body_count,
        }

        print(
            f"[OK] {global_id} ({person_name}): "
            f"{face_count} face image(s) from {len(face_folders)} folder(s), "
            f"{body_count} body image(s) from {len(body_folders)} folder(s)"
        )
        counter += 1

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(persons, f, indent=4)

    print(f"\n[{args.mode}] Wrote {len(persons)} identities to {output_path}")


if __name__ == "__main__":
    main()