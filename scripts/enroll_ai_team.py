"""Enrol the AI team directly from JSON configuration or test_images folder.

Database writing script:
  - Can read pre-computed JSON payloads containing team member embeddings directly into SQLite & Postgres.
  - Can fallback to image folder processing if raw photographs are provided.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psycopg2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402
from src.repository import EmbeddingInput, FaceRepository  # noqa: E402

LOGGER = logging.getLogger("enroll_ai_team")
MODEL_NAME = "face_recognition_sface_2021dec.onnx"

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


@dataclass(frozen=True)
class Candidate:
    source_image: str
    augmentation_name: str
    detection_score: float
    embedding: object
    is_original: bool


def select_candidates(
    originals: list[Candidate],
    augmented: list[Candidate],
    max_embeddings: int,
) -> list[Candidate]:
    """Keep independent photos first, then balance variants across sources."""
    if max_embeddings < 1:
        raise ValueError("max_embeddings must be positive.")

    selected = list(originals[:max_embeddings])
    remaining = max_embeddings - len(selected)
    if remaining <= 0:
        return selected

    source_order = list(dict.fromkeys(item.source_image for item in originals))
    variant_order = list(dict.fromkeys(item.augmentation_name for item in augmented))
    indexed = {(item.source_image, item.augmentation_name): item for item in augmented}

    for variant_name in variant_order:
        for source_image in source_order:
            candidate = indexed.get((source_image, variant_name))
            if candidate is None:
                continue
            selected.append(candidate)
            remaining -= 1
            if remaining == 0:
                return selected

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enroll AI team from JSON or image folders into DB.")
    parser.add_argument("--json", type=Path, help="Path to team JSON enrollment config.")
    parser.add_argument("--config", type=Path, help="Alias for --json config file.")
    parser.add_argument("--images-root", type=Path)
    parser.add_argument("--variants", type=int)
    parser.add_argument("--max-embeddings", type=int)
    parser.add_argument("--min-embeddings", type=int, default=5)
    parser.add_argument(
        "--skip-postgres-sync",
        action="store_true",
        help="Only write to local SQLite; skip pushing to Neon Postgres.",
    )
    return parser.parse_args()


def relative_text(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def get_postgres_connection():
    if not DATABASE_URL:
        LOGGER.warning("DATABASE_URL not set in .env; skipping Postgres sync.")
        return None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        return conn
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Could not connect to Postgres: %s", exc)
        return None


def push_to_postgres(pg_conn, person: dict, saved, selected: list[Candidate | EmbeddingInput]) -> None:
    """Upsert this person into the ai_team table on Neon Postgres."""
    phone = person.get("phone")
    if not phone:
        LOGGER.warning(
            "SKIPPED POSTGRES SYNC | %s has no phone in config; "
            "phone is required as the ai_team primary key.",
            person.get("full_name") or person.get("name"),
        )
        return

    embeddings = []
    for item in selected:
        emb = item.embedding if hasattr(item, "embedding") else item
        if hasattr(emb, "astype"):
            embeddings.append(emb.astype(float).tolist())
        else:
            embeddings.append(list(emb))

    name = person.get("full_name") or person.get("name")
    cur = pg_conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO ai_team
                (phone, face_id, name, age, designation, created_at, track_id, embeddings)
            VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s)
            ON CONFLICT (phone) DO UPDATE
            SET face_id = EXCLUDED.face_id,
                name = EXCLUDED.name,
                age = EXCLUDED.age,
                designation = EXCLUDED.designation,
                embeddings = EXCLUDED.embeddings;
            """,
            (
                phone,
                str(saved.person_id),
                name,
                person.get("age", 25),
                person.get("designation", "UNKNOWN"),
                [],
                embeddings,
            ),
        )
        LOGGER.info(
            "SYNCED TO POSTGRES | %s | phone=%s | face_id=%s | embeddings=%d",
            name,
            phone,
            saved.person_id,
            len(embeddings),
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("POSTGRES SYNC FAILED | %s | %s", name, exc)
    finally:
        cur.close()


def main() -> int:
    args = parse_args()
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config_path = (args.json or args.config or settings.paths.team_config).resolve()
    images_root = (args.images_root or settings.paths.test_images).resolve()
    variants = args.variants or settings.enrollment_variants
    max_embeddings = args.max_embeddings or settings.max_embeddings_per_person

    if not config_path.is_file():
        LOGGER.error("Team config JSON not found: %s", config_path)
        return 1

    people = json.loads(config_path.read_text(encoding="utf-8"))
    repository = FaceRepository(settings.database_path)
    pg_conn = None if args.skip_postgres_sync else get_postgres_connection()

    face_engine_instance = None
    failed_people = 0

    for person in people:
        full_name = person.get("full_name") or person.get("name")
        if not full_name:
            LOGGER.error("JSON person entry missing 'full_name' or 'name'")
            continue

        raw_embeddings = person.get("embeddings") or []

        # Mode A: Pre-computed embeddings provided in JSON -> Pure DB writing
        if raw_embeddings:
            prepared_inputs: list[EmbeddingInput] = []
            for idx, vec_raw in enumerate(raw_embeddings):
                vec = np.asarray(vec_raw, dtype=np.float32)
                prepared_inputs.append(
                    EmbeddingInput(
                        embedding=vec,
                        model_name=person.get("model_name", MODEL_NAME),
                        source_image=person.get("source_image", f"json_{full_name}_{idx}"),
                        augmentation_name=person.get("augmentation_name", "original"),
                        detection_score=float(person.get("detection_score", 1.0)),
                    )
                )

            saved = repository.replace_enrollment(
                full_name=full_name,
                person_type=person.get("person_type", "AI_TEAM"),
                phone=person.get("phone"),
                embeddings=prepared_inputs,
            )

            LOGGER.info(
                "ENROLLED FROM JSON | %s | person_id=%s | stored_embeddings=%d",
                saved.full_name,
                saved.person_id,
                saved.stored_embeddings,
            )

            if pg_conn is not None:
                push_to_postgres(pg_conn, person, saved, prepared_inputs)

            continue

        # Mode B: Image folder fallback (requires FaceEngine)
        if face_engine_instance is None:
            from src.face_engine import FaceEngine, SUPPORTED_IMAGE_EXTENSIONS
            face_engine_instance = FaceEngine(
                settings.paths.face_detector_model,
                settings.paths.face_recognizer_model,
                detector_backend=settings.face_detector_backend,
                low_light_enhancement=settings.face_low_light_enhancement,
                detection_score_threshold=settings.face_detection_score_threshold,
            )

        from src.face_engine import FaceEngineError, SUPPORTED_IMAGE_EXTENSIONS

        folder_name = person.get("folder") or full_name
        folder = images_root / folder_name
        image_paths = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ) if folder.is_dir() else []

        LOGGER.info("Processing %s from %s (%d files)", full_name, folder, len(image_paths))

        originals: list[Candidate] = []
        augmented: list[Candidate] = []
        rejected = 0

        for image_path in image_paths:
            try:
                image = face_engine_instance.read_image(image_path)
                result = face_engine_instance.extract_single_face(
                    image, minimum_face_size=settings.min_face_size
                )
                vector_variants = face_engine_instance.embedding_variants(
                    result.aligned_face, variants
                )
            except FaceEngineError as exc:
                rejected += 1
                LOGGER.warning("REJECTED | %s | %s", image_path.name, exc)
                continue

            for variant_name, embedding in vector_variants:
                candidate = Candidate(
                    source_image=relative_text(image_path),
                    augmentation_name=variant_name,
                    detection_score=result.detection.score,
                    embedding=embedding,
                    is_original=variant_name == "original",
                )
                (originals if candidate.is_original else augmented).append(candidate)

        selected = select_candidates(originals, augmented, max_embeddings)

        if len(selected) < args.min_embeddings:
            failed_people += 1
            LOGGER.error(
                "SKIPPED | %s has only %d usable embeddings; minimum is %d.",
                full_name,
                len(selected),
                args.min_embeddings,
            )
            continue

        inputs = [
            EmbeddingInput(
                embedding=item.embedding,
                model_name=MODEL_NAME,
                source_image=item.source_image,
                augmentation_name=item.augmentation_name,
                detection_score=item.detection_score,
            )
            for item in selected
        ]

        saved = repository.replace_enrollment(
            full_name=full_name,
            person_type=person.get("person_type", "AI_TEAM"),
            phone=person.get("phone"),
            embeddings=inputs,
        )

        LOGGER.info(
            "ENROLLED | %s | person_id=%s | stored_embeddings=%d",
            saved.full_name,
            saved.person_id,
            saved.stored_embeddings,
        )

        if pg_conn is not None:
            push_to_postgres(pg_conn, person, saved, selected)

    if pg_conn is not None:
        pg_conn.close()

    for full_name, person_id, count in repository.embedding_counts():
        LOGGER.info("DATABASE | %s | %s | embeddings=%d", full_name, person_id, count)

    return 1 if failed_people else 0


if __name__ == "__main__":
    raise SystemExit(main())