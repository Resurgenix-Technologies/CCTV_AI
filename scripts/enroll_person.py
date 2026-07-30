"""Validate enrolment photographs or JSON embeddings, and store SFace embeddings in SQLite and Postgres.

Database writing script:
  - Can read pre-computed JSON payloads containing embeddings and metadata directly into SQLite & Postgres.
  - Can fallback to FaceEngine image processing if an image folder is provided.

Examples:
    python scripts/enroll_person.py --json person.json
    python scripts/enroll_person.py --name "Test" --folder path/to/test/photos --type VISITOR --age 30 --designation Guest
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings  # noqa: E402
from src.repository import EmbeddingInput, FaceRepository, RepositoryError  # noqa: E402

LOGGER = logging.getLogger("face_identity.enroll_person")
MODEL_NAME = "face_recognition_sface_2021dec.onnx"

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


@dataclass(frozen=True)
class RejectedImage:
    path: Path
    reason: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Store SFace embeddings and metadata directly in SQLite & Postgres "
            "from JSON or image photographs."
        )
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="JSON file containing person metadata and pre-computed embeddings array.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help='Full display name, for example "Shrayan Sarkar". Required if not in --json.',
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=None,
        help="Directory containing photographs of this person.",
    )
    parser.add_argument(
        "--type",
        choices=("AI_TEAM", "VISITOR"),
        dest="person_type",
        default=None,
        help="Person classification stored in the database.",
    )
    parser.add_argument(
        "--phone",
        default=None,
        help="Optional phone number. Required for AI_TEAM Postgres sync.",
    )
    parser.add_argument(
        "--age",
        type=int,
        default=None,
        help="Optional age, stored in Postgres (visitors/ai_team).",
    )
    parser.add_argument(
        "--designation",
        default=None,
        help="Optional designation/role, stored in Postgres.",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=None,
        help="Number of augmentation variants per accepted photo.",
    )
    parser.add_argument(
        "--max-embeddings",
        type=int,
        default=None,
        help="Cap on total embeddings stored for this person.",
    )
    parser.add_argument(
        "--minimum-valid",
        type=int,
        default=5,
        help="Minimum accepted (original) photos required to save anything.",
    )
    parser.add_argument(
        "--skip-postgres-sync",
        action="store_true",
        help="Only write to local SQLite; skip pushing to Neon Postgres.",
    )
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def discover_images(folder: Path) -> list[Path]:
    from src.face_engine import SUPPORTED_IMAGE_EXTENSIONS
    if not folder.exists():
        raise FileNotFoundError(f"Enrolment folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Enrolment path is not a directory: {folder}")
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def private_source_label(image_path: Path, settings: Settings) -> str:
    try:
        return image_path.resolve().relative_to(
            settings.paths.root.resolve()
        ).as_posix()
    except ValueError:
        return f"{image_path.parent.name}/{image_path.name}"


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


def push_to_postgres(
    pg_conn,
    name: str,
    person_type: str,
    phone: str | None,
    age: int | None,
    designation: str | None,
    person_id: str,
    all_embeddings: list,
) -> None:
    """Upsert into visitors (PK face_id) or ai_team (PK phone) on Neon Postgres."""
    embeddings = [
        item.embedding.astype(float).tolist()
        if hasattr(item.embedding, "astype")
        else list(item.embedding)
        for item in all_embeddings
    ]

    cur = pg_conn.cursor()
    try:
        if person_type == "VISITOR":
            cur.execute(
                """
                INSERT INTO visitors
                    (face_id, name, phone, age, designation, created_at, track_id, embeddings)
                VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s)
                ON CONFLICT (face_id) DO UPDATE
                SET name = EXCLUDED.name,
                    phone = EXCLUDED.phone,
                    age = EXCLUDED.age,
                    designation = EXCLUDED.designation,
                    embeddings = EXCLUDED.embeddings;
                """,
                (person_id, name, phone, age, designation, [], embeddings),
            )
        else:  # AI_TEAM
            if not phone:
                LOGGER.warning(
                    "SKIPPED POSTGRES SYNC | %s has no --phone; "
                    "phone is required as the ai_team primary key.",
                    name,
                )
                return
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
                (phone, person_id, name, age, designation, [], embeddings),
            )

        LOGGER.info(
            "SYNCED TO POSTGRES | %s | table=%s | face_id=%s | embeddings=%d",
            name,
            "visitors" if person_type == "VISITOR" else "ai_team",
            person_id,
            len(embeddings),
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("POSTGRES SYNC FAILED | %s | %s", name, exc)
    finally:
        cur.close()


def process_json_enrollment(
    json_path: Path,
    repository: FaceRepository,
    args: argparse.Namespace,
) -> int:
    """Processes a JSON payload directly into SQLite and Postgres without image model inference."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        items = data
    else:
        items = [data]

    for entry in items:
        name = entry.get("name") or entry.get("full_name") or args.name
        person_type = entry.get("person_type") or entry.get("type") or args.person_type or "VISITOR"
        phone = entry.get("phone") or args.phone
        age = entry.get("age") if entry.get("age") is not None else args.age
        designation = entry.get("designation") or args.designation
        raw_embeddings = entry.get("embeddings") or []

        if not name:
            LOGGER.error("JSON entry missing required 'name' or 'full_name' field.")
            continue

        if not raw_embeddings:
            LOGGER.warning("JSON entry for %s contains no pre-computed embeddings vector array.", name)
            continue

        prepared_inputs: list[EmbeddingInput] = []
        for idx, emb_vec in enumerate(raw_embeddings):
            vec = np.asarray(emb_vec, dtype=np.float32)
            prepared_inputs.append(
                EmbeddingInput(
                    embedding=vec,
                    model_name=entry.get("model_name", MODEL_NAME),
                    source_image=entry.get("source_image", f"json_input_{idx}"),
                    augmentation_name=entry.get("augmentation_name", "original"),
                    detection_score=float(entry.get("detection_score", 1.0)),
                )
            )

        save_result = repository.replace_enrollment(
            full_name=name,
            person_type=person_type,
            phone=phone,
            embeddings=prepared_inputs,
        )

        LOGGER.info(
            "ENROLLED FROM JSON | %s | person_id=%s | stored_embeddings=%d",
            save_result.full_name,
            save_result.person_id,
            save_result.stored_embeddings,
        )

        if not args.skip_postgres_sync:
            pg_conn = get_postgres_connection()
            if pg_conn is not None:
                push_to_postgres(
                    pg_conn=pg_conn,
                    name=name,
                    person_type=person_type,
                    phone=phone,
                    age=age,
                    designation=designation,
                    person_id=save_result.person_id,
                    all_embeddings=prepared_inputs,
                )
                pg_conn.close()

    return 0


def main() -> int:
    args = build_parser().parse_args()

    try:
        settings = Settings.from_env()
        configure_logging(settings.log_level)
        repository = FaceRepository(settings.database_path)

        # Mode 1: Pure JSON DB enrollment (independent of ONNX model / OpenCV)
        if args.json:
            json_path = args.json.expanduser().resolve()
            if not json_path.is_file():
                raise FileNotFoundError(f"JSON file not found: {json_path}")
            return process_json_enrollment(json_path, repository, args)

        # Mode 2: Image folder enrollment (uses FaceEngine)
        if not args.name or not args.folder:
            LOGGER.error("Either --json or both --name and --folder must be provided.")
            return 1

        if not args.person_type:
            LOGGER.error("--type (AI_TEAM or VISITOR) is required when enrolling via --folder.")
            return 1

        from src.face_engine import FaceEngine, FaceEngineError

        variants = args.variants or settings.enrollment_variants
        max_embeddings = args.max_embeddings or settings.max_embeddings_per_person

        folder = args.folder.expanduser().resolve()
        image_paths = discover_images(folder)

        if not image_paths:
            raise FileNotFoundError(f"No supported images found in {folder}.")

        LOGGER.info("Found %d enrolment image(s) in %s", len(image_paths), folder)

        engine = FaceEngine(
            settings.paths.face_detector_model,
            settings.paths.face_recognizer_model,
            detector_backend=settings.face_detector_backend,
            low_light_enhancement=settings.face_low_light_enhancement,
            detection_score_threshold=settings.face_detection_score_threshold,
        )

        originals: list[EmbeddingInput] = []
        augmented: list[EmbeddingInput] = []
        rejected: list[RejectedImage] = []

        for image_path in image_paths:
            try:
                image = engine.read_image(image_path)
                result = engine.extract_single_face(
                    image, minimum_face_size=settings.min_face_size
                )
                vector_variants = engine.embedding_variants(
                    result.aligned_face, variants
                )
            except FaceEngineError as exc:
                rejected.append(RejectedImage(path=image_path, reason=str(exc)))
                LOGGER.warning("REJECTED | %s | %s", image_path.name, exc)
                continue

            source_label = private_source_label(image_path, settings)
            for variant_name, embedding in vector_variants:
                item = EmbeddingInput(
                    embedding=embedding,
                    model_name=MODEL_NAME,
                    source_image=source_label,
                    augmentation_name=variant_name,
                    detection_score=result.detection.score,
                )
                (originals if variant_name == "original" else augmented).append(item)

        if len(originals) < args.minimum_valid:
            LOGGER.error(
                "Only %d valid image(s) were found; at least %d are required.",
                len(originals),
                args.minimum_valid,
            )
            return 1

        selected = originals[:max_embeddings] + augmented[: max(0, max_embeddings - len(originals))]

        save_result = repository.replace_enrollment(
            full_name=args.name,
            person_type=args.person_type,
            phone=args.phone,
            embeddings=selected,
        )

        LOGGER.info(
            "Person: %s | person_id=%s | stored_embeddings=%d",
            save_result.full_name,
            save_result.person_id,
            save_result.stored_embeddings,
        )

        if not args.skip_postgres_sync:
            pg_conn = get_postgres_connection()
            if pg_conn is not None:
                push_to_postgres(
                    pg_conn=pg_conn,
                    name=args.name,
                    person_type=args.person_type,
                    phone=args.phone,
                    age=args.age,
                    designation=args.designation,
                    person_id=save_result.person_id,
                    all_embeddings=selected,
                )
                pg_conn.close()

        return 0

    except Exception as exc:  # noqa: BLE001
        logging.basicConfig(
            level=logging.ERROR, format="%(asctime)s | %(levelname)s | %(message)s"
        )
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())