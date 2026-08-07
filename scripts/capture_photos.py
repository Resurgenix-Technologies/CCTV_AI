"""Interactively capture validated enrolment photographs from a camera."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Union

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings  # noqa: E402
from src.face_engine import FaceEngine, FaceEngineError  # noqa: E402


LOGGER = logging.getLogger("face_identity.capture_photos")
WINDOW_NAME = "Face enrolment capture"


def parse_source(value: str) -> Union[int, str]:
    """Convert a numeric source to an OpenCV camera index."""

    stripped = value.strip()
    return int(stripped) if stripped.lstrip("-").isdigit() else stripped


def safe_directory_name(value: str) -> str:
    """Convert a person label to a safe lowercase folder name."""

    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    normalized = normalized.strip("_-").lower()

    if not normalized:
        raise ValueError(
            "--person must contain at least one letter or number."
        )

    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture enrolment photographs. Press SPACE to save a valid "
            "single-face frame and Q or ESC to stop."
        )
    )
    parser.add_argument(
        "--person",
        required=True,
        help="Folder-safe person label, for example shrayan_sarkar.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=12,
        help="Number of photographs to capture (default: 12).",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="OpenCV camera index. Defaults to CAMERA_SOURCE from .env.",
    )
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def draw_detection(
    frame,
    detections,
    minimum_face_size: int,
) -> tuple[bool, str]:
    """Draw current detections and return whether the frame can be saved."""

    for detection in detections:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(
            frame,
            f"face {detection.score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

    if not detections:
        return False, "NO FACE - look toward the camera"

    if len(detections) > 1:
        return False, f"REJECTED - {len(detections)} faces detected"

    detection = detections[0]
    if (
        detection.width < minimum_face_size
        or detection.height < minimum_face_size
    ):
        return (
            False,
            "FACE TOO SMALL - move closer "
            f"({detection.width}x{detection.height}px)",
        )

    return True, "READY - press SPACE to save"


def main() -> int:
    args = build_parser().parse_args()

    try:
        if args.count < 1:
            raise ValueError("--count must be at least 1.")

        settings = Settings.from_env()
        configure_logging(settings.log_level)

        person_directory = safe_directory_name(args.person)
        output_folder = settings.paths.enrollment / person_directory
        output_folder.mkdir(parents=True, exist_ok=True)

        engine = FaceEngine(
            settings.paths.face_detector_model,
            settings.paths.face_recognizer_model,
            detector_backend=settings.face_detector_backend,
            low_light_enhancement=settings.face_low_light_enhancement,
            detection_score_threshold=(
                settings.face_detection_score_threshold
            ),
        )

        source_value = (
            args.source
            if args.source is not None
            else settings.camera_source
        )
        source = parse_source(source_value)

        if isinstance(source, str) and not Path(source).exists():
            raise FileNotFoundError(
                f"Camera/video source not found: {source}"
            )

        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(
                f"OpenCV could not open camera source: {source}"
            )

        LOGGER.info("Saving photographs to %s", output_folder)
        LOGGER.info("Press SPACE to save; press Q or ESC to stop.")

        saved = 0

        try:
            while saved < args.count:
                success, frame = capture.read()
                if not success or frame is None:
                    raise RuntimeError(
                        "The camera stopped returning frames."
                    )

                detections = engine.detect_faces(frame)
                display = frame.copy()
                can_save, status = draw_detection(
                    display,
                    detections,
                    settings.min_face_size,
                )

                cv2.putText(
                    display,
                    f"Saved: {saved}/{args.count}",
                    (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    status,
                    (20, 64),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0) if can_save else (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    "SPACE: save | Q/ESC: quit",
                    (20, max(90, display.shape[0] - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    LOGGER.info("Capture stopped by the user.")
                    break

                if key == 32:
                    if not can_save:
                        LOGGER.warning("Image not saved: %s", status)
                        continue

                    timestamp = datetime.now().strftime(
                        "%Y%m%d_%H%M%S_%f"
                    )
                    destination = (
                        output_folder
                        / f"{person_directory}_{timestamp}.jpg"
                    )
                    FaceEngine.write_image(destination, frame)
                    saved += 1
                    LOGGER.info(
                        "Accepted %s (%d/%d)",
                        destination.name,
                        saved,
                        args.count,
                    )
        finally:
            capture.release()
            cv2.destroyAllWindows()

        LOGGER.info(
            "Capture complete: %d image(s) saved in %s",
            saved,
            output_folder,
        )
        return 0 if saved == args.count else 1

    except (
        ValueError,
        FileNotFoundError,
        RuntimeError,
        FaceEngineError,
    ) as exc:
        logging.basicConfig(
            level=logging.ERROR,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
