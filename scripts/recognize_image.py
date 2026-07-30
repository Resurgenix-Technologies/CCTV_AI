"""Recognize and annotate every visible face in one image."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402
from src.face_engine import FaceEngine, FaceEngineError  # noqa: E402
from src.recognizer import FaceRecognizer  # noqa: E402
from src.repository import FaceRepository  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = Settings.from_env()
    logging.basicConfig(level=getattr(logging, settings.log_level))

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
        image = engine.read_image(args.image)
    except FaceEngineError as exc:
        logging.error("%s", exc)
        return 1

    recognizer = FaceRecognizer(
        FaceRepository(settings.database_path),
        settings.face_match_threshold,
        settings.face_match_margin,
    )

    detections = engine.detect_faces(image)
    for detection in detections:
        try:
            result = engine.extract_embedding(image, detection)
            identity = recognizer.recognize(result.embedding)
        except FaceEngineError as exc:
            logging.warning("%s", exc)
            continue

        x1, y1, x2, y2 = detection.box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        decision = (
            identity.full_name
            if identity.matched
            else identity.rejection_reason or "UNKNOWN"
        )
        text = f"{decision} | similarity={identity.similarity:.3f}"
        cv2.putText(
            image,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        print(text)

    output = args.output or (
        settings.paths.outputs / f"recognized_{args.image.stem}.jpg"
    )
    engine.write_image(output, image)
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
