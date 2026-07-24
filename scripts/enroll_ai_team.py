"""Enrol the AI team directly from repository test_images."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402
from src.face_engine import (  # noqa: E402
    FaceEngine,
    FaceEngineError,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from src.repository import EmbeddingInput, FaceRepository  # noqa: E402


LOGGER = logging.getLogger("enroll_ai_team")


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
    """
    Keep independent photos first, then balance variants across sources.

    Source-major selection can spend the entire augmentation budget on one
    photograph. Variant-major selection gives the CCTV-domain representation
    of each independent source a chance before adding a second transform.
    """

    if max_embeddings < 1:
        raise ValueError("max_embeddings must be positive.")

    selected = list(originals[:max_embeddings])
    remaining = max_embeddings - len(selected)
    if remaining <= 0:
        return selected

    source_order = list(dict.fromkeys(
        item.source_image for item in originals
    ))
    variant_order = list(dict.fromkeys(
        item.augmentation_name for item in augmented
    ))
    indexed = {
        (item.source_image, item.augmentation_name): item
        for item in augmented
    }

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--images-root", type=Path)
    parser.add_argument("--variants", type=int)
    parser.add_argument("--max-embeddings", type=int)
    parser.add_argument("--min-embeddings", type=int, default=5)
    return parser.parse_args()


def relative_text(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def main() -> int:
    args = parse_args()
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config_path = (args.config or settings.paths.team_config).resolve()
    images_root = (args.images_root or settings.paths.test_images).resolve()
    variants = args.variants or settings.enrollment_variants
    max_embeddings = (
        args.max_embeddings or settings.max_embeddings_per_person
    )

    if not config_path.is_file():
        LOGGER.error("Team config not found: %s", config_path)
        return 1
    if not images_root.is_dir():
        LOGGER.error("Image root not found: %s", images_root)
        return 1

    people = json.loads(config_path.read_text(encoding="utf-8"))

    try:
        engine = FaceEngine(
            settings.paths.face_detector_model,
            settings.paths.face_recognizer_model,
            detector_backend=settings.face_detector_backend,
            low_light_enhancement=settings.face_low_light_enhancement,
            detection_score_threshold=(
                settings.face_detection_score_threshold
            ),
        )
    except FaceEngineError as exc:
        LOGGER.error("%s", exc)
        return 1

    repository = FaceRepository(settings.database_path)
    failed_people = 0

    for person in people:
        folder = images_root / person["folder"]
        image_paths = sorted(
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ) if folder.is_dir() else []

        LOGGER.info(
            "Processing %s from %s (%d files)",
            person["full_name"],
            folder,
            len(image_paths),
        )

        originals: list[Candidate] = []
        augmented: list[Candidate] = []
        rejected = 0

        for image_path in image_paths:
            try:
                image = engine.read_image(image_path)
                result = engine.extract_single_face(
                    image,
                    minimum_face_size=settings.min_face_size,
                )
                vector_variants = engine.embedding_variants(
                    result.aligned_face,
                    variants,
                )
            except FaceEngineError as exc:
                rejected += 1
                LOGGER.warning(
                    "REJECTED | %s | %s", image_path.name, exc
                )
                continue

            LOGGER.info(
                "ACCEPTED | %s | face=%dx%d | score=%.3f",
                image_path.name,
                result.detection.width,
                result.detection.height,
                result.detection.score,
            )

            for variant_name, embedding in vector_variants:
                candidate = Candidate(
                    source_image=relative_text(image_path),
                    augmentation_name=variant_name,
                    detection_score=result.detection.score,
                    embedding=embedding,
                    is_original=variant_name == "original",
                )
                (originals if candidate.is_original else augmented).append(
                    candidate
                )

        selected = select_candidates(
            originals,
            augmented,
            max_embeddings,
        )

        if len(selected) < args.min_embeddings:
            failed_people += 1
            LOGGER.error(
                "SKIPPED | %s has only %d usable embeddings; "
                "minimum is %d.",
                person["full_name"],
                len(selected),
                args.min_embeddings,
            )
            continue

        inputs = [
            EmbeddingInput(
                embedding=item.embedding,
                model_name="face_recognition_sface_2021dec.onnx",
                source_image=item.source_image,
                augmentation_name=item.augmentation_name,
                detection_score=item.detection_score,
            )
            for item in selected
        ]

        saved = repository.replace_enrollment(
            full_name=person["full_name"],
            person_type=person.get("person_type", "AI_TEAM"),
            phone=person.get("phone"),
            embeddings=inputs,
        )

        LOGGER.info(
            "ENROLLED | %s | person_id=%s | original_photos=%d | "
            "stored_embeddings=%d | rejected=%d",
            saved.full_name,
            saved.person_id,
            len(originals),
            saved.stored_embeddings,
            rejected,
        )

        if len(originals) < 5:
            LOGGER.warning(
                "%s uses augmentation because fewer than five "
                "independent photographs were accepted.",
                saved.full_name,
            )

    for full_name, person_id, count in repository.embedding_counts():
        LOGGER.info(
            "DATABASE | %s | %s | embeddings=%d",
            full_name,
            person_id,
            count,
        )

    return 1 if failed_people else 0


if __name__ == "__main__":
    raise SystemExit(main())
