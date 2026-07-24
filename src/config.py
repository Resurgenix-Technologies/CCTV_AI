"""Validated application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _integer(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, received {raw!r}.") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, received {value}.")
    return value


def _floating(
    name: str,
    default: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric, received {raw!r}.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, received {value}."
        )
    return value


def _choice(
    name: str,
    default: str,
    choices: set[str],
) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in choices:
        raise ValueError(
            f"{name} must be one of {sorted(choices)}, received {value!r}."
        )
    return value


@dataclass(frozen=True)
class ProjectPaths:
    """Absolute paths derived from the project root."""

    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    enrollment: Path = PROJECT_ROOT / "data" / "enrollment"
    test_images: Path = PROJECT_ROOT / "test_images"
    models: Path = PROJECT_ROOT / "models"
    outputs: Path = PROJECT_ROOT / "outputs"
    team_config: Path = PROJECT_ROOT / "config" / "ai_team.json"
    face_detector_model: Path = (
        PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
    )
    face_recognizer_model: Path = (
        PROJECT_ROOT / "models" / "face_recognition_sface_2021dec.onnx"
    )


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for enrolment, recognition and tracking."""

    database_path: Path
    face_match_threshold: float
    face_match_margin: float
    face_detector_backend: str
    face_low_light_enhancement: str
    face_recognition_interval: int
    face_confirmation_votes: int
    face_vote_window: int
    track_ttl_frames: int
    min_face_size: int
    face_detection_score_threshold: float
    live_face_detection_min_size: int
    live_face_identity_min_size: int
    live_face_roi_upscale: float
    live_face_roi_ratio: float
    live_face_roi_score_threshold: float
    live_face_roi_max_dimension: int
    face_diagnostic_log_interval: int
    enrollment_variants: int
    max_embeddings_per_person: int
    yolo_model: str
    yolo_confidence: float
    yolo_image_size: int
    tracker_config: str
    camera_source: str
    ndi_source_name: str
    ndi_layout: str
    ndi_discovery_timeout_ms: int
    ndi_capture_timeout_ms: int
    log_level: str
    paths: ProjectPaths

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Settings":
        load_dotenv(env_file or PROJECT_ROOT / ".env")

        database_raw = os.getenv(
            "DATABASE_PATH", "data/face_identity.db"
        ).strip()
        database_path = Path(database_raw)
        if not database_path.is_absolute():
            database_path = PROJECT_ROOT / database_path

        vote_window = _integer("FACE_VOTE_WINDOW", 5, 1)
        confirmation_votes = _integer("FACE_CONFIRMATION_VOTES", 3, 1)
        if confirmation_votes > vote_window:
            raise ValueError(
                "FACE_CONFIRMATION_VOTES cannot exceed FACE_VOTE_WINDOW."
            )

        live_detection_min_size = _integer(
            "LIVE_FACE_DETECTION_MIN_SIZE",
            12,
            1,
        )
        live_identity_min_size = _integer(
            "LIVE_FACE_IDENTITY_MIN_SIZE",
            32,
            1,
        )
        if live_identity_min_size < live_detection_min_size:
            raise ValueError(
                "LIVE_FACE_IDENTITY_MIN_SIZE cannot be below "
                "LIVE_FACE_DETECTION_MIN_SIZE."
            )

        log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if log_level not in valid_levels:
            raise ValueError(
                f"LOG_LEVEL must be one of {sorted(valid_levels)}."
            )

        return cls(
            database_path=database_path.resolve(),
            face_match_threshold=_floating("FACE_MATCH_THRESHOLD", 0.45),
            face_match_margin=_floating("FACE_MATCH_MARGIN", 0.05),
            face_detector_backend=_choice(
                "FACE_DETECTOR_BACKEND", "yunet", {"yunet", "scrfd"}
            ),
            face_low_light_enhancement=_choice(
                "FACE_LOW_LIGHT_ENHANCEMENT", "none", {"none", "lime"}
            ),
            face_recognition_interval=_integer(
                "FACE_RECOGNITION_INTERVAL", 10, 1
            ),
            face_confirmation_votes=confirmation_votes,
            face_vote_window=vote_window,
            track_ttl_frames=_integer("TRACK_TTL_FRAMES", 150, 1),
            min_face_size=_integer("MIN_FACE_SIZE", 48, 1),
            face_detection_score_threshold=_floating(
                "FACE_DETECTION_SCORE_THRESHOLD", 0.72
            ),
            live_face_detection_min_size=live_detection_min_size,
            live_face_identity_min_size=live_identity_min_size,
            live_face_roi_upscale=_floating(
                "LIVE_FACE_ROI_UPSCALE",
                2.0,
                1.0,
                4.0,
            ),
            live_face_roi_ratio=_floating(
                "LIVE_FACE_ROI_RATIO",
                0.55,
                0.20,
                0.80,
            ),
            live_face_roi_score_threshold=_floating(
                "LIVE_FACE_ROI_SCORE_THRESHOLD",
                0.60,
            ),
            live_face_roi_max_dimension=_integer(
                "LIVE_FACE_ROI_MAX_DIMENSION",
                640,
                32,
            ),
            face_diagnostic_log_interval=_integer(
                "FACE_DIAGNOSTIC_LOG_INTERVAL",
                300,
                1,
            ),
            enrollment_variants=_integer("ENROLLMENT_VARIANTS", 5, 1),
            max_embeddings_per_person=_integer(
                "MAX_EMBEDDINGS_PER_PERSON", 10, 5
            ),
            yolo_model=os.getenv("YOLO_MODEL", "yolo11n.pt").strip()
            or "yolo11n.pt",
            yolo_confidence=_floating("YOLO_CONFIDENCE", 0.35),
            yolo_image_size=_integer("YOLO_IMAGE_SIZE", 640, 32),
            tracker_config=os.getenv(
                "TRACKER_CONFIG", "botsort.yaml"
            ).strip()
            or "bytetrack.yaml",
            camera_source=os.getenv("CAMERA_SOURCE", "0").strip() or "0",
            ndi_source_name=os.getenv("NDI_SOURCE_NAME", "").strip(),
            ndi_layout=_choice(
                "NDI_LAYOUT",
                "single",
                {"single", "2x2"},
            ),
            ndi_discovery_timeout_ms=_integer(
                "NDI_DISCOVERY_TIMEOUT_MS", 10000, 100
            ),
            ndi_capture_timeout_ms=_integer(
                "NDI_CAPTURE_TIMEOUT_MS", 1000, 1
            ),
            log_level=log_level,
            paths=ProjectPaths(),
        )

    def parsed_camera_source(self) -> Union[int, str]:
        source = self.camera_source.strip()
        return int(source) if source.lstrip("-").isdigit() else source

    def resolve_input_layout(
        self,
        *,
        is_ndi: bool,
        override: str | None = None,
    ) -> str:
        """Resolve CLI override, NDI default, and non-NDI compatibility."""

        if override is not None and override not in {"single", "2x2"}:
            raise ValueError("Input layout must be 'single' or '2x2'.")
        if override is not None:
            return override
        return self.ndi_layout if is_ndi else "single"
