"""Validate enrolment photographs and store SFace embeddings in SQLite."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings  # noqa: E402
from src.face_engine import (  # noqa: E402
    FaceEngine,
    FaceEngineError,
    FaceTooSmallError,
    InvalidImageError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from src.repository import (  # noqa: E402
    ExistingEnrollmentError,
    FaceEmbeddingInput,
    FaceRepository,
    RepositoryError,
)


LOGGER = logging.getLogger("face_identity.enroll_person")
MODEL_NAME = "face_recognition_sface_2021dec.onnx"


@dataclass(frozen=True)
class RejectedImage:
    """An enrolment image rejected with a human-readable reason."""

    path: Path
    reason: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect exactly one face per photograph, generate normalized "
            "SFace embeddings, and store them in SQLite."
        )
    )
    parser.add_argument(
        "--name",
        required=True,
        help='Full display name, for example "Shrayan Sarkar".',
    )
    parser.add_argument(
        "--folder",
        required=True,
        type=Path,
        help="Directory containing approximately 10 to 15 photographs.",
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=("AI_TEAM", "VISITOR"),
        dest="person_type",
        help="Person classification stored in the database.",
    )
    parser.add_argument(
        "--phone",
        default=None,
        help="Optional phone number, maximum 20 characters.",
    )
    parser.add_argument(
        "--minimum-valid",
        type=int,
        default=5,
        help=(
            "Do not write to the database unless at least this many images "
            "are valid (default: 5)."
        ),
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete this person's old embeddings before saving the new set.",
    )
    mode_group.add_argument(
        "--append",
        action="store_true",
        help="Append new source images and skip already stored source paths.",
    )
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def discover_images(folder: Path) -> list[Path]:
    """Return supported images directly inside the enrolment folder."""

    if not folder.exists():
        raise FileNotFoundError(f"Enrolment folder not found: {folder}")

    if not folder.is_dir():
        raise NotADirectoryError(
            f"Enrolment path is not a directory: {folder}"
        )

    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def private_source_label(image_path: Path, settings: Settings) -> str:
    """Avoid storing an unnecessary absolute Windows user path."""

    try:
        return image_path.resolve().relative_to(
            settings.paths.root.resolve()
        ).as_posix()
    except ValueError:
        return f"{image_path.parent.name}/{image_path.name}"


def rejection_reason(exc: FaceEngineError) -> str:
    """Convert known face-processing exceptions into concise log text."""

    if isinstance(exc, NoFaceDetectedError):
        return "no face detected"
    if isinstance(exc, MultipleFacesDetectedError):
        return str(exc).lower()
    if isinstance(exc, FaceTooSmallError):
        return str(exc)
    if isinstance(exc, InvalidImageError):
        return str(exc)
    return f"face processing failed: {exc}"


def main() -> int:
    args = build_parser().parse_args()

    try:
        if args.minimum_valid < 1:
            raise ValueError("--minimum-valid must be at least 1.")

        settings = Settings.from_env()
        configure_logging(settings.log_level)

        folder = args.folder.expanduser().resolve()
        image_paths = discover_images(folder)

        if not image_paths:
            raise FileNotFoundError(
                f"No supported images found in {folder}. "
                "Supported extensions: "
                f"{sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
            )

        LOGGER.info(
            "Found %d enrolment image(s) in %s",
            len(image_paths),
            folder,
        )

        repository = FaceRepository(settings.database_path)
        repository.verify_schema()

        engine = FaceEngine(
            settings.paths.face_detector_model,
            settings.paths.face_recognizer_model,
            detector_backend=settings.face_detector_backend,
            low_light_enhancement=settings.face_low_light_enhancement,
            detection_score_threshold=(
                settings.face_detection_score_threshold
            ),
        )

        accepted: list[FaceEmbeddingInput] = []
        rejected: list[RejectedImage] = []

        for image_path in image_paths:
            try:
                image = engine.read_image(image_path)
                result = engine.extract_single_face_embedding(
                    image,
                    minimum_face_size=settings.min_face_size,
                )

                source_label = private_source_label(
                    image_path,
                    settings,
                )
                accepted.append(
                    FaceEmbeddingInput(
                        embedding=result.embedding,
                        model_name=MODEL_NAME,
                        source_image=source_label,
                        detection_score=result.detection.score,
                    )
                )
                LOGGER.info(
                    "ACCEPTED | %s | face=%dx%d | score=%.3f",
                    image_path.name,
                    result.detection.width,
                    result.detection.height,
                    result.detection.score,
                )
            except FaceEngineError as exc:
                reason = rejection_reason(exc)
                rejected.append(
                    RejectedImage(path=image_path, reason=reason)
                )
                LOGGER.warning(
                    "REJECTED | %s | %s",
                    image_path.name,
                    reason,
                )

        LOGGER.info(
            "Validation summary: accepted=%d rejected=%d total=%d",
            len(accepted),
            len(rejected),
            len(image_paths),
        )

        if len(accepted) < args.minimum_valid:
            LOGGER.error(
                "Only %d valid image(s) were found; at least %d are required. "
                "Nothing was written to SQLite.",
                len(accepted),
                args.minimum_valid,
            )
            return 1

        mode = "error"
        if args.replace_existing:
            mode = "replace"
        elif args.append:
            mode = "append"

        save_result = repository.save_enrollment(
            full_name=args.name,
            person_type=args.person_type,
            phone=args.phone,
            embeddings=accepted,
            mode=mode,
        )

        LOGGER.info("Enrolment saved successfully.")
        LOGGER.info(
            "Person: %s | person_id=%s | type=%s",
            save_result.person.full_name,
            save_result.person.person_id,
            save_result.person.person_type,
        )
        LOGGER.info(
            "Embeddings: inserted=%d skipped=%d total=%d",
            save_result.inserted_embeddings,
            save_result.skipped_embeddings,
            save_result.total_embeddings,
        )

        if save_result.total_embeddings < 5:
            LOGGER.warning(
                "This identity has fewer than five embeddings. "
                "Capture additional valid photographs before evaluation."
            )

        return 0

    except (
        ValueError,
        FileNotFoundError,
        NotADirectoryError,
        FaceEngineError,
        RepositoryError,
        ExistingEnrollmentError,
    ) as exc:
        logging.basicConfig(
            level=logging.ERROR,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
