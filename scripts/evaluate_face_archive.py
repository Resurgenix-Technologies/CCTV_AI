"""Evaluate unseen labelled CCTV faces in a ZIP without enrolling them."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402
from src.face_engine import (  # noqa: E402
    FaceEngine,
    FaceEngineError,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from src.recognizer import FaceRecognizer  # noqa: E402
from src.repository import FaceRepository  # noqa: E402


MAX_ARCHIVE_MEMBERS = 5_000
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024


class ArchiveValidationError(ValueError):
    """Raised when a supplied dataset archive is unsafe or malformed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate only new labelled images from a test_images ZIP. "
            "Exact copies of enrollment images are skipped to prevent "
            "training/evaluation leakage."
        )
    )
    parser.add_argument("archive", type=Path)
    return parser.parse_args()


def member_identity(
    filename: str,
    allowed_folders: set[str],
) -> tuple[str, str] | None:
    """Validate an archive member and return (identity folder, filename)."""

    normalized = filename.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ArchiveValidationError(
            f"Unsafe archive member path: {filename!r}"
        )
    if not path.parts or normalized.endswith("/"):
        return None
    if len(path.parts) != 3 or path.parts[0] != "test_images":
        raise ArchiveValidationError(
            "Image members must use test_images/<identity>/<file>: "
            f"{filename!r}"
        )
    folder, basename = path.parts[1], path.parts[2]
    if folder not in allowed_folders:
        raise ArchiveValidationError(
            f"Unknown identity folder in archive: {folder!r}"
        )
    if Path(basename).suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return None
    return folder, basename


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def existing_image_hashes(root: Path) -> set[str]:
    return {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    }


def main() -> int:
    args = parse_args()
    archive_path = args.archive.expanduser().resolve()
    if not archive_path.is_file():
        print(f"Archive not found: {archive_path}", file=sys.stderr)
        return 2

    settings = Settings.from_env()
    team = json.loads(
        settings.paths.team_config.read_text(encoding="utf-8")
    )
    folder_to_name = {
        str(person["folder"]): str(person["full_name"])
        for person in team
    }

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
        recognizer = FaceRecognizer(
            FaceRepository(settings.database_path),
            settings.face_match_threshold,
            settings.face_match_margin,
        )
    except (FaceEngineError, OSError, ValueError) as exc:
        print(f"Could not initialize face evaluation: {exc}", file=sys.stderr)
        return 2

    if recognizer.embedding_count == 0:
        print(
            "No enrolled embeddings found. Run scripts/enroll_ai_team.py.",
            file=sys.stderr,
        )
        return 2

    enrollment_hashes = existing_image_hashes(settings.paths.test_images)
    seen_hashes: set[str] = set()
    counts: Counter[str] = Counter()

    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise ArchiveValidationError(
                    f"Archive has too many members ({len(members)})."
                )
            total_size = sum(item.file_size for item in members)
            if total_size > MAX_TOTAL_BYTES:
                raise ArchiveValidationError(
                    "Archive uncompressed size exceeds 256 MiB."
                )

            for member in members:
                identity = member_identity(
                    member.filename,
                    set(folder_to_name),
                )
                if identity is None:
                    continue
                if member.file_size > MAX_IMAGE_BYTES:
                    raise ArchiveValidationError(
                        f"Archive image is too large: {member.filename!r}"
                    )

                folder, basename = identity
                data = archive.read(member)
                digest = sha256_bytes(data)
                if digest in enrollment_hashes:
                    counts["duplicate_enrollment"] += 1
                    continue
                if digest in seen_hashes:
                    counts["duplicate_archive"] += 1
                    continue
                seen_hashes.add(digest)
                counts["new"] += 1

                image = cv2.imdecode(
                    np.frombuffer(data, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if image is None or image.size == 0:
                    counts["invalid"] += 1
                    print(f"INVALID    {folder}/{basename} | decode failed")
                    continue

                try:
                    detections = engine.detect_faces(image)
                except FaceEngineError as exc:
                    counts["invalid"] += 1
                    print(f"INVALID    {folder}/{basename} | {exc}")
                    continue
                if len(detections) != 1:
                    counts["invalid"] += 1
                    print(
                        f"INVALID    {folder}/{basename} | "
                        f"expected 1 face, found {len(detections)}"
                    )
                    continue

                detection = detections[0]
                dimensions = f"{detection.width}x{detection.height}px"
                if min(detection.width, detection.height) < (
                    settings.live_face_identity_min_size
                ):
                    counts["too_small"] += 1
                    print(
                        f"TOO_SMALL  {folder}/{basename} | {dimensions} | "
                        f"requires {settings.live_face_identity_min_size}px"
                    )
                    continue

                try:
                    embedding = engine.extract_embedding(
                        image,
                        detection,
                    ).embedding
                    result = recognizer.recognize(embedding)
                except FaceEngineError as exc:
                    counts["invalid"] += 1
                    print(f"INVALID    {folder}/{basename} | {exc}")
                    continue

                expected = folder_to_name[folder]
                if not result.matched:
                    counts["unknown"] += 1
                    print(
                        f"UNKNOWN    {folder}/{basename} | {dimensions} | "
                        f"score={result.similarity:.3f} | "
                        f"reason={result.rejection_reason}"
                    )
                elif result.full_name == expected:
                    counts["correct"] += 1
                    print(
                        f"CORRECT    {folder}/{basename} | {dimensions} | "
                        f"score={result.similarity:.3f}"
                    )
                else:
                    counts["wrong"] += 1
                    print(
                        f"WRONG      {folder}/{basename} | {dimensions} | "
                        f"expected={expected!r} predicted={result.full_name!r} "
                        f"score={result.similarity:.3f}"
                    )

    except (ArchiveValidationError, OSError, zipfile.BadZipFile) as exc:
        print(f"Invalid archive: {exc}", file=sys.stderr)
        return 2

    eligible = counts["correct"] + counts["unknown"] + counts["wrong"]
    print(
        "SUMMARY | "
        f"new={counts['new']} | "
        f"enrollment_duplicates={counts['duplicate_enrollment']} | "
        f"eligible={eligible} | correct={counts['correct']} | "
        f"unknown={counts['unknown']} | wrong={counts['wrong']} | "
        f"too_small={counts['too_small']} | invalid={counts['invalid']}"
    )
    if counts["new"] == 0 or counts["invalid"] or counts["wrong"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
