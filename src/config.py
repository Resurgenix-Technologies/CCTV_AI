"""Application configuration loaded from environment variables."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _integer(name: str, default: int, *, minimum: int = 0) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, received {raw_value!r}.") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, received {value}.")
    return value

def _floating(name: str, default: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, received {raw_value!r}.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, received {value}.")
    return value

@dataclass(frozen=True)
class ProjectPaths:
    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    enrollment: Path = PROJECT_ROOT / "data" / "enrollment"
    models: Path = PROJECT_ROOT / "models"
    outputs: Path = PROJECT_ROOT / "outputs"
    face_detector_model: Path = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
    face_recognizer_model: Path = PROJECT_ROOT / "models" / "face_recognition_sface_2021dec.onnx"

@dataclass(frozen=True)
class Settings:
    database_path: Path
    face_match_threshold: float
    face_recognition_interval: int
    face_confirmation_votes: int
    face_vote_window: int
    min_face_size: int
    face_detection_score_threshold: float
    yolo_model: str
    yolo_confidence: float
    tracker_config: str
    track_ttl_frames: int
    camera_source: str
    log_level: str
    paths: ProjectPaths

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Settings":
        load_dotenv(env_file or PROJECT_ROOT / ".env")
        database_value = os.getenv("DATABASE_PATH", "data/face_identity.db").strip()
        database_path = Path(database_value)
        if not database_path.is_absolute():
            database_path = PROJECT_ROOT / database_path
        vote_window = _integer("FACE_VOTE_WINDOW", 5, minimum=1)
        confirmation_votes = _integer("FACE_CONFIRMATION_VOTES", 3, minimum=1)
        if confirmation_votes > vote_window:
            raise ValueError("FACE_CONFIRMATION_VOTES cannot be greater than FACE_VOTE_WINDOW.")
        log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if log_level not in valid:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(valid)}, received {log_level!r}.")
        return cls(
            database_path=database_path,
            face_match_threshold=_floating("FACE_MATCH_THRESHOLD", 0.45),
            face_recognition_interval=_integer("FACE_RECOGNITION_INTERVAL", 10, minimum=1),
            face_confirmation_votes=confirmation_votes,
            face_vote_window=vote_window,
            min_face_size=_integer("MIN_FACE_SIZE", 80, minimum=1),
            face_detection_score_threshold=_floating("FACE_DETECTION_SCORE_THRESHOLD", 0.90),
            yolo_model=os.getenv("YOLO_MODEL", "yolo11n.pt").strip() or "yolo11n.pt",
            yolo_confidence=_floating("YOLO_CONFIDENCE", 0.35),
            tracker_config=os.getenv("TRACKER_CONFIG", "bytetrack.yaml").strip() or "bytetrack.yaml",
            track_ttl_frames=_integer("TRACK_TTL_FRAMES", 60, minimum=1),
            camera_source=os.getenv("CAMERA_SOURCE", "0").strip() or "0",
            log_level=log_level,
            paths=ProjectPaths(),
        )

    def parsed_camera_source(self) -> Union[int, str]:
        source = self.camera_source.strip()
        return int(source) if source.lstrip("-").isdigit() else source
