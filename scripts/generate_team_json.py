"""Utility script to generate or update AI team JSON configurations for enrollment.

Supports:
  1. Adding/updating team members in JSON config.
  2. Auto-discovering photo folders from test_images/.
  3. Pre-computing embedding vectors into JSON so enrollment scripts can run as pure DB writers.

Examples:
  # Add a new team member to config/ai_team.json:
  python scripts/generate_team_json.py --add --name "John Doe" --folder "johndoe" --phone "+919999999999" --age 25 --designation "AI Engineer"

  # Precompute embeddings into JSON file for pure DB enrollment:
  python scripts/generate_team_json.py --precompute --output config/ai_team_standalone.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings

LOGGER = logging.getLogger("generate_team_json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and manage AI team JSON enrollment files.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "ai_team.json",
        help="Path to target AI team JSON config file.",
    )
    parser.add_argument(
        "--add",
        action="store_true",
        help="Add or update a team member in the JSON config.",
    )
    parser.add_argument("--name", help="Full display name of the member.")
    parser.add_argument("--folder", help="Image folder name inside test_images/.")
    parser.add_argument("--phone", help="Phone number (required for Postgres primary key).")
    parser.add_argument("--age", type=int, default=25, help="Age of the team member.")
    parser.add_argument("--designation", default="AI Engineer", help="Role / designation.")
    parser.add_argument(
        "--person-type",
        choices=("AI_TEAM", "VISITOR"),
        default="AI_TEAM",
        help="Person type classification.",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan test_images/ directory and auto-generate missing team JSON entries.",
    )
    parser.add_argument(
        "--precompute",
        action="store_true",
        help="Extract face embeddings from images and embed vectors directly into the JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path when saving precomputed JSON (defaults to overwriting --config).",
    )
    return parser


def load_config(config_path: Path) -> list[dict]:
    if config_path.is_file():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.error("Error reading JSON file %s: %s", config_path, exc)
    return []


def save_config(config_path: Path, data: list[dict]):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    LOGGER.info("Saved team JSON config to: %s (%d members)", config_path, len(data))


def add_member(data: list[dict], name: str, folder: str, phone: str, age: int, designation: str, person_type: str) -> list[dict]:
    updated = False
    new_entry = {
        "folder": folder,
        "full_name": name,
        "person_type": person_type,
        "phone": phone,
        "age": age,
        "designation": designation,
    }

    for idx, item in enumerate(data):
        if item.get("folder") == folder or item.get("full_name") == name:
            data[idx].update(new_entry)
            updated = True
            LOGGER.info("Updated existing entry for: %s", name)
            break

    if not updated:
        data.append(new_entry)
        LOGGER.info("Added new team member entry for: %s", name)

    return data


def scan_test_images(settings: Settings, data: list[dict]) -> list[dict]:
    test_images_dir = settings.paths.test_images
    if not test_images_dir.is_dir():
        LOGGER.error("test_images directory not found: %s", test_images_dir)
        return data

    existing_folders = {item.get("folder") for item in data if "folder" in item}
    counter = len(data) + 1

    for child in sorted(test_images_dir.iterdir()):
        if child.is_dir() and child.name not in existing_folders:
            default_name = child.name.capitalize()
            default_phone = f"+91000000000{counter}"
            data.append({
                "folder": child.name,
                "full_name": default_name,
                "person_type": "AI_TEAM",
                "phone": default_phone,
                "age": 25,
                "designation": "AI Engineer",
            })
            LOGGER.info("Discovered folder '%s' -> Added entry for %s (%s)", child.name, default_name, default_phone)
            counter += 1

    return data


def precompute_embeddings(settings: Settings, data: list[dict]) -> list[dict]:
    """Extracts embeddings using FaceEngine and includes them inside the JSON payload."""
    from src.face_engine import FaceEngine, FaceEngineError, SUPPORTED_IMAGE_EXTENSIONS

    LOGGER.info("Initializing FaceEngine to pre-compute embeddings into JSON...")
    engine = FaceEngine(
        settings.paths.face_detector_model,
        settings.paths.face_recognizer_model,
        detector_backend=settings.face_detector_backend,
        low_light_enhancement=settings.face_low_light_enhancement,
        detection_score_threshold=settings.face_detection_score_threshold,
    )

    images_root = settings.paths.test_images

    for item in data:
        folder_name = item.get("folder")
        if not folder_name:
            continue

        folder = images_root / folder_name
        if not folder.is_dir():
            continue

        image_paths = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )

        embeddings_list = []
        for img_path in image_paths:
            try:
                img = engine.read_image(img_path)
                result = engine.extract_single_face(img, minimum_face_size=settings.min_face_size)
                variants = engine.embedding_variants(result.aligned_face, settings.enrollment_variants)
                for _var_name, emb_vector in variants:
                    embeddings_list.append(emb_vector.astype(float).tolist())
            except FaceEngineError as exc:
                LOGGER.warning("Skipping %s for %s: %s", img_path.name, item.get("full_name"), exc)

        item["embeddings"] = embeddings_list
        LOGGER.info("Precomputed %d embedding vectors for %s", len(embeddings_list), item.get("full_name"))

    return data


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    settings = Settings.from_env()

    config_path = args.config.resolve()
    data = load_config(config_path)

    if args.add:
        if not args.name or not args.folder or not args.phone:
            LOGGER.error("--add requires --name, --folder, and --phone parameters.")
            return 1
        data = add_member(
            data,
            name=args.name,
            folder=args.folder,
            phone=args.phone,
            age=args.age,
            designation=args.designation,
            person_type=args.person_type,
        )

    if args.scan:
        data = scan_test_images(settings, data)

    if args.precompute:
        data = precompute_embeddings(settings, data)

    output_path = (args.output or config_path).resolve()
    save_config(output_path, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
