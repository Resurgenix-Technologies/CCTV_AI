"""Live YOLO/ByteTrack face identity overlay for OpenCV or NDI input."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402
from src.embedding_fusion import FaceEmbeddingFusion  # noqa: E402
from src.face_engine import (  # noqa: E402
    FaceEngine,
    FaceEngineError,
)
from src.identity_manager import (  # noqa: E402
    IdentityManager,
    VoteProgress,
)
from src.activity_processor import ActivityProcessor  # noqa: E402
from src.activity_repository import ActivityRepository  # noqa: E402
from src.event_repository import (  # noqa: E402
    EventRepository,
    EventRepositoryError,
    RecognitionEvent,
)
from src.logging_config import configure_logging  # noqa: E402
from src.live_face_pipeline import (  # noqa: E402
    LiveFacePipeline,
    TrackFaceResult,
)
from src.ndi_source import (  # noqa: E402
    NDIFrameSource,
    ThreadedNDIFrameSource,
    NDISourceNotFoundError,
    NDIUnavailableError,
)
from src.recognizer import FaceRecognizer  # noqa: E402
from src.repository import FaceRepository  # noqa: E402
from src.tracker import (  # noqa: E402
    PersonTracker,
    QuadrantPersonTracker,
    TrackedPerson,
)
from src.interaction_engine import (  # noqa: E402
    InteractionConfig,
    ProximityInteractionEngine,
    TrackObservation,
)
from src.conversation_logger import ConversationLogger  # noqa: E402
from src.low_light import enhance_ndi_frame  # noqa: E402


LOGGER = logging.getLogger("live_recognition")

UNKNOWN_COLOR = (165, 165, 165)
PENDING_COLOR = (0, 191, 255)
CONFIRMED_COLOR = (0, 200, 0)
FACE_COLOR = (210, 210, 210)
REJECTED_FACE_COLOR = (0, 140, 255)
PANEL_BACKGROUND = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)


class OpenCVSource:
    """OpenCV webcam or video-file source."""

    def __init__(
        self,
        source: int | str,
        recording_start: datetime | None = None,
    ) -> None:
        self.is_recording = False

        if isinstance(source, str):
            path = Path(source)

            if path.suffix and not path.exists():
                raise FileNotFoundError(
                    f"Video file not found: {path}"
                )

            self.is_recording = path.is_file()

        self.capture = cv2.VideoCapture(source)

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open source: {source}"
            )

        self.source = source
        self.recording_start = (
            recording_start
            or datetime.now(timezone.utc)
        )

    def read(
        self,
    ) -> tuple[bool, np.ndarray | None]:
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()

    def capture_timestamp(self) -> str:
        """Return capture time, using the video timeline for recordings."""

        if self.is_recording:
            offset_ms = max(
                0.0,
                float(
                    self.capture.get(
                        cv2.CAP_PROP_POS_MSEC
                    )
                ),
            )
            if offset_ms <= 0.0:
                frame_position = max(
                    0.0,
                    float(
                        self.capture.get(
                            cv2.CAP_PROP_POS_FRAMES
                        )
                    )
                    - 1.0,
                )
                offset_ms = (
                    frame_position / self.fps * 1000.0
                )
            captured_at = self.recording_start + timedelta(
                milliseconds=offset_ms
            )
        else:
            captured_at = datetime.now(timezone.utc)

        return captured_at.isoformat(timespec="milliseconds")

    @property
    def fps(self) -> float:
        value = self.capture.get(
            cv2.CAP_PROP_FPS
        )
        return value if value > 0 else 30.0

    @property
    def label(self) -> str:
        if isinstance(self.source, int):
            return f"Webcam {self.source}"

        return f"Video: {Path(self.source).name}"


def parse_utc_datetime(value: str) -> datetime:
    """Parse an ISO-8601 timestamp and normalize it to UTC."""

    normalized = value.strip().replace("Z", "+00:00")

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "timestamp must be ISO-8601, for example "
            "2026-07-21T09:00:00+05:30"
        ) from exc

    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(
            "timestamp must include a UTC offset or Z"
        )

    return parsed.astimezone(timezone.utc)


def parse_positive_integer(value: str) -> int:
    """Parse a strictly positive CLI integer."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        )
    return parsed


def parse_nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative number") from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be a non-negative number")
    return parsed


def parse_normalized_roi(value: str) -> tuple[float, float, float, float]:
    """Parse an inclusive restricted rectangle as x1,y1,x2,y2 in 0..1."""

    try:
        parts = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "ROI must be four normalized numbers: x1,y1,x2,y2"
        ) from exc
    if len(parts) != 4 or not all(np.isfinite(part) for part in parts):
        raise argparse.ArgumentTypeError(
            "ROI must be four normalized numbers: x1,y1,x2,y2"
        )
    x1, y1, x2, y2 = parts
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise argparse.ArgumentTypeError(
            "ROI coordinates must satisfy 0 <= x1 < x2 <= 1 and "
            "0 <= y1 < y2 <= 1"
        )
    return x1, y1, x2, y2


def tracks_inside_roi(
    tracks: list[TrackedPerson],
    frame_shape: tuple[int, ...],
    roi: tuple[float, float, float, float]
    | list[tuple[float, float, float, float]]
    | None,
) -> list[TrackedPerson]:
    """Keep only tracks whose box center is within the restricted region."""

    if roi is None:
        return tracks
    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain height and width")
    height, width = frame_shape[:2]
    boxes = [roi] if isinstance(roi, tuple) else roi
    assert boxes is not None
    return [
        track
        for track in tracks
        if any(
            box[0] * width <= (track.box[0] + track.box[2]) / 2.0 <= box[2] * width
            and box[1] * height <= (track.box[1] + track.box[3]) / 2.0 <= box[3] * height
            for box in boxes
        )
    ]


def draw_restricted_roi(
    frame: np.ndarray,
    roi: tuple[float, float, float, float]
    | list[tuple[float, float, float, float]]
    | None,
) -> None:
    """Draw the active restricted rectangle on the live frame."""

    if roi is None:
        return
    height, width = frame.shape[:2]
    boxes = [roi] if isinstance(roi, tuple) else roi
    assert boxes is not None
    for index, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        left = max(0, min(width - 1, round(x1 * width)))
        top = max(0, min(height - 1, round(y1 * height)))
        right = max(left + 1, min(width - 1, round(x2 * width)))
        bottom = max(top + 1, min(height - 1, round(y2 * height)))
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 3)
        cv2.putText(
            frame,
            f"RESTRICTED ROI {index}",
            (left + 8, max(24, top + 26)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def normalized_roi_from_selection(
    selection: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
) -> tuple[float, float, float, float] | None:
    """Convert OpenCV's pixel ROI selection into normalized coordinates."""

    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain height and width")
    x, y, width, height = (int(value) for value in selection)
    frame_height, frame_width = frame_shape[:2]
    if width < 2 or height < 2 or frame_width < 1 or frame_height < 1:
        return None
    x = max(0, min(frame_width - 1, x))
    y = max(0, min(frame_height - 1, y))
    right = max(x + 1, min(frame_width, x + width))
    bottom = max(y + 1, min(frame_height, y + height))
    return (
        x / frame_width,
        y / frame_height,
        right / frame_width,
        bottom / frame_height,
    )


def should_process_source_frame(
    source_frame_index: int,
    frame_stride: int,
) -> bool:
    """Return whether a one-based source frame belongs to the stride."""

    if source_frame_index < 1:
        raise ValueError("source_frame_index must be positive.")
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive.")
    return (source_frame_index - 1) % frame_stride == 0


def validate_frame_stride(
    frame_stride: int,
    *,
    is_recording: bool,
) -> None:
    """Reject temporal subsampling for live cameras and NDI streams."""

    if frame_stride < 1:
        raise ValueError("frame_stride must be positive.")
    if frame_stride > 1 and not is_recording:
        raise ValueError(
            "--frame-stride is supported only for recorded videos."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track people and recognize visible faces "
            "from a webcam, video or NDI source."
        )
    )

    parser.add_argument(
        "--source",
        default=None,
        help="OpenCV camera index or video path.",
    )

    parser.add_argument(
        "--ndi",
        action="store_true",
        help="Use a direct NDI receiver.",
    )

    parser.add_argument(
        "--ndi-source",
        default=None,
        help=(
            "NDI source index or case-insensitive "
            "partial source name."
        ),
    )

    parser.add_argument(
        "--input-layout",
        choices=("single", "2x2"),
        default=None,
        help=(
            "Process one normal image or split a four-camera 2x2 mosaic. "
            "For NDI, the NDI_LAYOUT setting is used when omitted."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Optional processed MP4 output path.",
    )

    parser.add_argument(
        "--recording-start",
        type=parse_utc_datetime,
        help=(
            "UTC/offset ISO-8601 time of the first frame. "
            "Use with a recorded video so event times follow capture time."
        ),
    )

    parser.add_argument(
        "--frame-stride",
        type=parse_positive_integer,
        default=1,
        help=(
            "Process every Nth frame of a recorded video (default: 1). "
            "Output FPS and event timing preserve the source timeline."
        ),
    )

    parser.add_argument(
        "--restricted-roi",
        type=parse_normalized_roi,
        action="append",
        help=(
            "Only report tracks whose box center is inside this rectangle "
            "(normalized x1,y1,x2,y2). Repeat the option for multiple ROIs."
        ),
    )

    parser.add_argument(
        "--select-roi",
        action="store_true",
        help=(
            "Draw the restricted ROI interactively on the first frame. "
            "Press Enter/Space to confirm or Esc to disable it."
        ),
    )

    parser.add_argument(
        "--interaction-minimum-dwell",
        type=parse_nonnegative_float,
        default=None,
        help="Proximity dwell threshold in seconds; default 3.",
    )
    parser.add_argument(
        "--interaction-gap-grace",
        type=parse_nonnegative_float,
        default=None,
        help="Detector-gap grace period in seconds; default 1.",
    )
    parser.add_argument(
        "--interaction-enter-distance",
        type=parse_nonnegative_float,
        default=None,
        help="Normalized proximity distance to start an estimate; default 1.25.",
    )
    parser.add_argument(
        "--interaction-exit-distance",
        type=parse_nonnegative_float,
        default=None,
        help="Normalized proximity distance to end an estimate; default 1.50.",
    )

    parser.add_argument(
        "--show-faces",
        action="store_true",
        help="Draw accepted and quality-rejected YuNet face boxes.",
    )

    parser.add_argument(
        "--hide-status-panel",
        action="store_true",
        help="Hide the top-left runtime status panel (already hidden by default).",
    )
    parser.add_argument(
        "--show-status-panel",
        action="store_true",
        help="Show the detailed runtime status panel.",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without an OpenCV window or keyboard controls.",
    )

    parser.add_argument(
        "--disable-event-logs",
        action="store_true",
        help="Do not store structured recognition events in SQLite.",
    )

    return parser.parse_args()


def track_camera_label(
    source_label: str,
    track: TrackedPerson,
) -> str:
    """Return a stable camera namespace for logs and stored events."""

    if track.quadrant_index is None:
        return source_label
    return f"{source_label} [Q{track.quadrant_index}]"


def track_event_id(track: TrackedPerson) -> int:
    """Return the camera-local tracker ID used in event storage."""

    if track.local_track_id is None:
        return track.track_id
    return (track.quadrant_epoch << 32) | track.local_track_id


def track_overlay_line(track: TrackedPerson) -> str:
    """Format a compact local-track label for the video overlay."""

    if track.quadrant_index is None:
        return f"Local track: {track.track_id}"
    line = (
        f"Camera Q{track.quadrant_index} / track "
        f"{track.local_track_id}"
    )
    if track.quadrant_epoch > 0:
        line += f" / recovery {track.quadrant_epoch}"
    return line


def state_color(
    progress: VoteProgress,
) -> tuple[int, int, int]:
    """Return the BGR color for a recognition state."""

    if progress.status == "CONFIRMED":
        return CONFIRMED_COLOR

    if progress.status == "PENDING":
        return PENDING_COLOR

    return UNKNOWN_COLOR


def draw_text_block(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    lines: list[str],
    color: tuple[int, int, int],
) -> None:
    """Draw a readable translucent identity label."""

    x1, y1, x2, _ = box
    line_height = 21
    block_height = (
        line_height * len(lines) + 10
    )
    top = max(0, y1 - block_height)
    width = max(
        230,
        min(max(x2 - x1, 230), 370),
    )
    right = min(
        frame.shape[1] - 1,
        x1 + width,
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, top),
        (right, y1),
        PANEL_BACKGROUND,
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.68,
        frame,
        0.32,
        0,
        frame,
    )

    cv2.rectangle(
        frame,
        (x1, top),
        (right, y1),
        color,
        2,
    )

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (
                x1 + 6,
                top + 19 + index * line_height,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )


def draw_status_panel(
    frame: np.ndarray,
    *,
    source_label: str,
    fps: float,
    active_tracks: int,
    enrolled_people: int,
    stored_embeddings: int,
    threshold: float,
    recognition_interval: int,
    face_batch_summary: str,
    roi_upscale: float,
    identity_min_size: int,
    yolo_image_size: int,
    source_stride: int = 1,
) -> None:
    """Draw global runtime information at the top-left."""

    lines = [
        f"Source: {source_label}",
        f"FPS: {fps:.1f}",
        f"Active tracks: {active_tracks}",
        (
            "Enrolled: "
            f"{enrolled_people} people / "
            f"{stored_embeddings} embeddings"
        ),
        f"Face threshold: {threshold:.2f}",
        (
            "Recognition interval: "
            f"{recognition_interval} frames"
        ),
        f"Face batch: {face_batch_summary}",
        (
            "ROI fallback: "
            f"{roi_upscale:.1f}x | "
            f"ID floor: {identity_min_size}px"
        ),
        f"YOLO input size: {yolo_image_size}px",
    ]
    if source_stride > 1:
        lines.append(f"Offline source stride: {source_stride}")

    line_height = 22
    panel_width = min(
        440,
        max(300, frame.shape[1] - 20),
    )
    panel_height = (
        len(lines) * line_height + 14
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (8, 8),
        (8 + panel_width, 8 + panel_height),
        PANEL_BACKGROUND,
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.68,
        frame,
        0.32,
        0,
        frame,
    )

    cv2.rectangle(
        frame,
        (8, 8),
        (8 + panel_width, 8 + panel_height),
        TEXT_COLOR,
        1,
    )

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (
                16,
                29 + index * line_height,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )


def create_writer(
    output: Path,
    frame: np.ndarray,
    fps: float,
) -> cv2.VideoWriter:
    """Create and validate an MP4 writer."""

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    height, width = frame.shape[:2]

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open output writer: {output}"
        )

    return writer


def store_event_safely(
    repository: EventRepository | None,
    event: RecognitionEvent,
) -> None:
    """Store an event without terminating the video pipeline on log failure."""

    if repository is None:
        return

    try:
        repository.log_event(event)
    except (EventRepositoryError, ValueError) as exc:
        LOGGER.warning("Recognition event was not stored: %s", exc)


def face_stage_diagnostic(
    result: TrackFaceResult,
    identity_min_size: int,
) -> str:
    """Return a concise per-track explanation of the last face decision."""

    source = result.source.upper() if result.source else "ROI"
    size = result.face_size
    if result.status == "NO_FACE":
        return "Face: not detected (full + ROI)"
    if result.status == "INVALID_ROI":
        return "Face: invalid/off-frame person ROI"
    if result.status == "DETECTION_FAILED":
        return "Face: detector error"
    if result.status == "EMBEDDING_FAILED":
        return "Face: alignment/embedding failed"
    if size is None:
        return f"Face: {result.status.lower()}"

    width, height = size
    if result.status == "FACE_TOO_SMALL":
        return (
            f"Face: {width}x{height}px {source} TOO SMALL "
            f"(<{identity_min_size}px)"
        )
    return f"Face: {width}x{height}px {source} ready"


def draw_face_stage_result(
    frame: np.ndarray,
    result: TrackFaceResult,
) -> None:
    """Draw detected faces, including candidates rejected by quality gates."""

    if result.detection is None:
        return

    x1, y1, x2, y2 = result.detection.box
    color = (
        FACE_COLOR
        if result.status == "READY"
        else REJECTED_FACE_COLOR
    )
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    source = result.source.upper() if result.source else "FACE"
    status = result.status.replace("FACE_", "").replace("_", " ")
    label = (
        f"{source} {result.detection.width}x"
        f"{result.detection.height} {status}"
    )
    cv2.putText(
        frame,
        label,
        (x1, max(18, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


def log_face_diagnostic_summary(
    counts: Counter[str],
    *,
    session_id: str,
    source_label: str,
    reason: str,
) -> None:
    """Write a bounded aggregate instead of one rejection log per frame."""

    if not counts:
        return
    LOGGER.info(
        "FACE_DIAGNOSTICS | session=%s | source=%s | reason=%s | "
        "ready=%d | matched=%d | conflicts=%d | unknown=%d | ambiguous=%d | "
        "too_small=%d | no_face=%d | failures=%d",
        session_id[:8],
        source_label,
        reason,
        counts["READY"],
        counts["MATCHED"],
        counts["CONFLICT"],
        counts["UNKNOWN_MATCH"],
        counts["AMBIGUOUS"],
        counts["FACE_TOO_SMALL"],
        counts["NO_FACE"],
        (
            counts["DETECTION_FAILED"]
            + counts["EMBEDDING_FAILED"]
            + counts["INVALID_ROI"]
        ),
    )


def log_interaction_events(
    events: list[object],
    conversation_logger: ConversationLogger | None = None,
    person_names: dict[str, str] | None = None,
) -> None:
    """Log completed proximity estimates without claiming conversation."""

    for event in events:
        if conversation_logger is not None:
            conversation_logger.write(event, person_names)
        LOGGER.info(
            "INTERACTION_%s | camera=%s | person_a=%s | person_b=%s | "
            "duration=%.1fs | distance=%.2f | estimate=proximity",
            event.event_type,
            event.camera_id,
            event.person_a_id[:12],
            event.person_b_id[:12],
            event.duration_seconds,
            event.normalized_distance,
        )


def update_active_interactions(
    active: dict[tuple[str, str, str], float],
    events: list[object],
) -> None:
    """Maintain the currently active proximity-estimate pairs."""

    for event in events:
        key = (event.camera_id, event.person_a_id, event.person_b_id)
        if event.event_type in {"STARTED", "UPDATED"}:
            active[key] = float(event.duration_seconds)
        elif event.event_type == "ENDED":
            active.pop(key, None)


def draw_active_interactions(
    frame: np.ndarray,
    tracks: list[TrackedPerson],
    identity_manager: IdentityManager,
    active: dict[tuple[str, str, str], float],
    source_label: str,
) -> None:
    """Draw active estimated interactions between currently visible people."""

    by_camera_person: dict[tuple[str, str], TrackedPerson] = {}
    for track in tracks:
        state = identity_manager.get(track.track_id)
        camera = track_camera_label(source_label, track)
        # Visual interaction tracking also includes unrecognized people.
        # Use the same stable key as the interaction observation path.
        person_key = state.person_id or f"unknown_track_{track.track_id}"
        by_camera_person[(camera, person_key)] = track

    for (camera, person_a, person_b), duration in sorted(active.items()):
        first = by_camera_person.get((camera, person_a))
        second = by_camera_person.get((camera, person_b))
        if first is None or second is None:
            continue
        first_center = (
            (first.box[0] + first.box[2]) // 2,
            (first.box[1] + first.box[3]) // 2,
        )
        second_center = (
            (second.box[0] + second.box[2]) // 2,
            (second.box[1] + second.box[3]) // 2,
        )
        cv2.line(frame, first_center, second_center, (0, 255, 255), 3)
        midpoint = (
            (first_center[0] + second_center[0]) // 2,
            (first_center[1] + second_center[1]) // 2,
        )
        cv2.putText(
            frame,
            f"INTERACTION ESTIMATE {duration:.0f}s",
            midpoint,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main() -> int:
    args = parse_args()
    settings = Settings.from_env()
    input_layout = settings.resolve_input_layout(
        is_ndi=args.ndi,
        override=args.input_layout,
    )

    log_file = configure_logging(
        settings.log_level,
        settings.paths.root / "logs",
    )
    session_id = str(uuid.uuid4())

    if args.ndi:
        try:
            validate_frame_stride(
                args.frame_stride,
                is_recording=False,
            )
        except ValueError as exc:
            LOGGER.error("%s", exc)
            return 1

    try:
        engine = FaceEngine(
            settings.paths.face_detector_model,
            settings.paths.face_recognizer_model,
            detector_backend=settings.face_detector_backend,
            low_light_enhancement=settings.face_low_light_enhancement,
            detection_score_threshold=(
                settings
                .face_detection_score_threshold
            ),
        )

        repository = FaceRepository(
            settings.database_path
        )

        event_repository = (
            None
            if args.disable_event_logs
            else EventRepository(settings.database_path)
        )

        recognizer = FaceRecognizer(
            repository,
            settings.face_match_threshold,
            settings.face_match_margin,
        )

        if recognizer.embedding_count == 0:
            raise RuntimeError(
                "No enrolled embeddings found. Run: "
                "python scripts\\enroll_ai_team.py"
            )

        tracker_type = (
            QuadrantPersonTracker
            if input_layout == "2x2"
            else PersonTracker
        )
        tracker = tracker_type(
            settings.yolo_model,
            settings.yolo_confidence,
            settings.tracker_config,
            settings.yolo_image_size,
            **(
                {"quadrant_upscale": float(os.getenv("QUADRANT_DETECTION_UPSCALE", "2.0"))}
                if input_layout == "2x2"
                else {}
            ),
        )

        face_pipeline = LiveFacePipeline(
            engine,
            detection_min_size=(
                settings.live_face_detection_min_size
            ),
            identity_min_size=(
                settings.live_face_identity_min_size
            ),
            roi_upscale=settings.live_face_roi_upscale,
            roi_upper_body_ratio=(
                settings.live_face_roi_ratio
            ),
            roi_score_threshold=(
                settings.live_face_roi_score_threshold
            ),
            roi_max_dimension=(
                settings.live_face_roi_max_dimension
            ),
        )
        face_fusion = FaceEmbeddingFusion(max_samples=5)

    except (
        FaceEngineError,
        RuntimeError,
        ValueError,
    ) as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Runtime log file: %s", log_file)
    LOGGER.info("Recognition session ID: %s", session_id)
    LOGGER.info(
        "Structured SQLite event logs: %s",
        "disabled" if event_repository is None else "enabled",
    )
    LOGGER.info(
        "Loaded %d enrolled people.",
        recognizer.person_count,
    )
    LOGGER.info(
        "Loaded %d face embeddings.",
        recognizer.embedding_count,
    )
    LOGGER.info(
        "Face match threshold: %.3f",
        settings.face_match_threshold,
    )
    LOGGER.info(
        "Face detector backend: %s | low-light enhancement: %s",
        settings.face_detector_backend.upper(),
        settings.face_low_light_enhancement.upper(),
    )
    LOGGER.info(
        "Face ambiguity margin: %.3f",
        settings.face_match_margin,
    )
    LOGGER.info(
        "Voting rule: %d of last %d results.",
        settings.face_confirmation_votes,
        settings.face_vote_window,
    )
    LOGGER.info(
        "Recognition target: every %d source frames.",
        settings.face_recognition_interval,
    )
    LOGGER.info(
        "YOLO model: %s",
        settings.yolo_model,
    )
    LOGGER.info(
        "YOLO inference size: %d pixels.",
        settings.yolo_image_size,
    )
    LOGGER.info(
        "Input layout: %s%s.",
        input_layout,
        (
            " (four independent quadrant trackers)"
            if input_layout == "2x2"
            else ""
        ),
    )
    LOGGER.info(
        "Restricted ROI: %s.",
        args.restricted_roi or "disabled",
    )
    LOGGER.info(
        "Distant-face fallback: %.1fx upper-body ROI "
        "(ratio %.2f, score threshold %.2f, max enlarged side %dpx).",
        settings.live_face_roi_upscale,
        settings.live_face_roi_ratio,
        settings.live_face_roi_score_threshold,
        settings.live_face_roi_max_dimension,
    )
    LOGGER.info(
        "Live face quality: detection floor=%dpx, identity floor=%dpx.",
        settings.live_face_detection_min_size,
        settings.live_face_identity_min_size,
    )

    source = None
    writer = None
    identity_manager = None
    activity_processor: ActivityProcessor | None = None
    visual_interaction_engine: ProximityInteractionEngine | None = None
    conversation_logger = ConversationLogger(
        settings.paths.root / "conversation_logs",
        session_id,
    )
    active_interactions: dict[tuple[str, str, str], float] = {}
    conversation_person_names: dict[str, str] = {}
    output_path = args.output
    source_label = "Unknown source"
    restricted_roi = args.restricted_roi
    roi_selected = not args.select_roi
    track_last_seen_at: dict[int, str] = {}
    track_source_labels: dict[int, str] = {}
    track_event_ids: dict[int, int] = {}
    face_diagnostics: dict[int, str] = {}
    face_batch_summary = "waiting"
    face_diagnostic_counts: Counter[str] = Counter()
    last_face_diagnostic_log_frame = 0
    last_captured_at: str | None = None

    try:
        if args.ndi:
            selector = (
                args.ndi_source
                if args.ndi_source is not None
                else settings.ndi_source_name
            )

            source = ThreadedNDIFrameSource(
                selector,
                discovery_timeout_ms=(
                    settings
                    .ndi_discovery_timeout_ms
                ),
                capture_timeout_ms=(
                    settings
                    .ndi_capture_timeout_ms
                ),
            )

            source_label = (
                f"NDI: {source.connected_name}"
            )

            LOGGER.info(
                "Connected to NDI source: %s",
                source.connected_name,
            )

            fps_for_writer = 30.0

        else:
            source_value = (
                args.source
                if args.source is not None
                else settings.camera_source
            )

            parsed_source: int | str = (
                int(source_value)
                if str(source_value)
                .lstrip("-")
                .isdigit()
                else str(source_value)
            )

            source = OpenCVSource(
                parsed_source,
                recording_start=args.recording_start,
            )

            source_label = source.label
            validate_frame_stride(
                args.frame_stride,
                is_recording=source.is_recording,
            )
            fps_for_writer = source.fps / args.frame_stride

            LOGGER.info(
                "Opened source: %s",
                source_label,
            )
            if args.frame_stride > 1:
                LOGGER.info(
                    "Offline frame stride: %d (output %.3f FPS).",
                    args.frame_stride,
                    fps_for_writer,
                )

            if (
                source.is_recording
                and args.recording_start is None
            ):
                LOGGER.warning(
                    "No --recording-start was supplied. "
                    "Recorded-video event times use the current UTC time "
                    "as a synthetic epoch. Durations remain capture-based."
                )

            if (
                output_path is None
                and isinstance(
                    parsed_source,
                    str,
                )
            ):
                path = Path(parsed_source)

                if path.is_file():
                    output_path = (
                        settings.paths.outputs
                        / (
                            "tracked_"
                            f"{path.stem}.mp4"
                        )
                    )

        identity_manager = IdentityManager(
            settings.face_confirmation_votes,
            settings.face_vote_window,
            settings.track_ttl_frames,
        )
        interaction_config = InteractionConfig()
        interaction_changes: dict[str, float] = {}
        if args.interaction_minimum_dwell is not None:
            interaction_changes["minimum_dwell_seconds"] = args.interaction_minimum_dwell
        if args.interaction_gap_grace is not None:
            interaction_changes["gap_grace_seconds"] = args.interaction_gap_grace
        if args.interaction_enter_distance is not None:
            interaction_changes["enter_distance_threshold"] = args.interaction_enter_distance
        if args.interaction_exit_distance is not None:
            interaction_changes["exit_distance_threshold"] = args.interaction_exit_distance
        elif args.interaction_enter_distance is not None:
            interaction_changes["exit_distance_threshold"] = max(
                interaction_config.exit_distance_threshold,
                args.interaction_enter_distance,
            )
        if interaction_changes:
            interaction_config = replace(interaction_config, **interaction_changes)
        visual_interaction_engine = ProximityInteractionEngine(
            interaction_config
        )
        if not args.disable_event_logs:
            activity_processor = ActivityProcessor(
                ActivityRepository(settings.database_path),
                ProximityInteractionEngine(interaction_config),
            )

        source_frame_index = 0
        processed_frame_count = 0
        next_face_source_frame = settings.face_recognition_interval
        previous_tick = time.perf_counter()
        smoothed_fps = 0.0

        while True:
            ok, frame = source.read()

            if ok and frame is not None and settings.face_low_light_enhancement == "lime":
                frame = enhance_ndi_frame(frame)

            if not ok or frame is None:
                if args.ndi:
                    key = (
                        cv2.waitKey(1) & 0xFF
                        if not args.headless
                        else -1
                    )

                    if key in {
                        ord("q"),
                        27,
                    }:
                        break

                    continue

                break

            source_frame_index += 1

            if not roi_selected:
                if args.headless:
                    raise RuntimeError(
                        "--select-roi requires a display; remove --headless."
                    )
                selected_rois: list[tuple[float, float, float, float]] = []
                while True:
                    selection = cv2.selectROI(
                        "Draw restricted ROI; press Enter to confirm",
                        frame,
                        fromCenter=False,
                        showCrosshair=True,
                    )
                    cv2.destroyWindow(
                        "Draw restricted ROI; press Enter to confirm"
                    )
                    selected = normalized_roi_from_selection(
                        tuple(int(value) for value in selection),
                        frame.shape,
                    )
                    if selected is not None:
                        selected_rois.append(selected)
                    answer = input(
                        "Draw another restricted ROI? [y/N]: "
                    ).strip().lower()
                    if answer not in {"y", "yes"}:
                        break
                restricted_roi = selected_rois or None
                roi_selected = True
                LOGGER.info(
                    "Interactive restricted ROI: %s.",
                    restricted_roi or "disabled",
                )

            if not should_process_source_frame(
                source_frame_index,
                args.frame_stride,
            ):
                continue

            processed_frame_count += 1
            # Source indices keep TTLs, diagnostics and stored events on the
            # original recording timeline even when inference skips frames.
            frame_index = source_frame_index

            captured_at = (
                source.capture_timestamp()
                if isinstance(source, OpenCVSource)
                else datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"
                )
            )
            last_captured_at = captured_at

            current_tick = time.perf_counter()
            elapsed = max(
                current_tick - previous_tick,
                1e-6,
            )
            previous_tick = current_tick
            instantaneous_fps = 1.0 / elapsed

            smoothed_fps = (
                instantaneous_fps
                if smoothed_fps <= 0.0
                else (
                    0.90 * smoothed_fps
                    + 0.10
                    * instantaneous_fps
                )
            )

            tracks = tracks_inside_roi(
                tracker.track(frame),
                frame.shape,
                restricted_roi,
            )
            draw_restricted_roi(frame, restricted_roi)
            track_by_id = {
                track.track_id: track
                for track in tracks
            }
            active_ids = set(track_by_id)

            for track in tracks:
                track_last_seen_at[track.track_id] = captured_at
                track_source_labels[track.track_id] = track_camera_label(
                    source_label,
                    track,
                )
                track_event_ids[track.track_id] = track_event_id(track)

            started_states = (
                identity_manager.mark_seen(
                    active_ids,
                    frame_index,
                )
            )

            for state in started_states:
                current_track = track_by_id.get(state.track_id)
                detection_confidence = (
                    current_track.confidence
                    if current_track is not None
                    else None
                )
                event_source = track_source_labels.get(
                    state.track_id,
                    source_label,
                )
                event_track_id = track_event_ids.get(
                    state.track_id,
                    state.track_id,
                )

                LOGGER.info(
                    "TRACK_STARTED | "
                    "session=%s | source=%s | track=%d",
                    session_id[:8],
                    event_source,
                    event_track_id,
                )

                store_event_safely(
                    event_repository,
                    RecognitionEvent(
                        session_id=session_id,
                        event_type="TRACK_STARTED",
                        camera_source=event_source,
                        local_track_id=event_track_id,
                        detection_confidence=detection_confidence,
                        frame_index=frame_index,
                        occurred_at=captured_at,
                    ),
                )

            current_face_results: list[TrackFaceResult] = []
            if frame_index >= next_face_source_frame:
                missed_intervals = (
                    frame_index - next_face_source_frame
                ) // settings.face_recognition_interval
                next_face_source_frame += (
                    missed_intervals + 1
                ) * settings.face_recognition_interval
                current_face_results = face_pipeline.process(
                    frame,
                    tracks,
                )
                batch_counts: Counter[str] = Counter(
                    result.status for result in current_face_results
                )
                face_diagnostic_counts.update(batch_counts)
                matched_faces = 0
                conflicting_faces = 0
                unknown_faces = 0
                ambiguous_faces = 0

                for result in current_face_results:
                    face_diagnostics[result.track_id] = (
                        face_stage_diagnostic(
                            result,
                            settings.live_face_identity_min_size,
                        )
                    )

                    if result.status != "READY":
                        LOGGER.debug(
                            "FACE_STAGE_REJECTED | session=%s | "
                            "source=%s | track=%d | status=%s | "
                            "detail=%s",
                            session_id[:8],
                            track_source_labels.get(
                                result.track_id,
                                source_label,
                            ),
                            track_event_ids.get(
                                result.track_id,
                                result.track_id,
                            ),
                            result.status,
                            result.error or "none",
                        )
                        continue

                    if (
                        result.embedding is None
                        or result.detection is None
                    ):
                        LOGGER.warning(
                            "Face pipeline returned READY without "
                            "an embedding for track %d.",
                            result.track_id,
                        )
                        continue

                    fused_embedding = face_fusion.update(
                        result.track_id,
                        result.embedding,
                        detection_score=result.detection.score,
                        face_size=(
                            result.detection.width,
                            result.detection.height,
                        ),
                    )
                    identity = recognizer.recognize(
                        fused_embedding
                    )
                    width, height = result.face_size or (0, 0)
                    source_name = (
                        result.source.upper()
                        if result.source
                        else "FACE"
                    )
                    current_state = identity_manager.get(result.track_id)
                    conflicting_match = (
                        identity.matched
                        and current_state.confirmed
                        and current_state.person_id is not None
                        and identity.person_id != current_state.person_id
                    )
                    if conflicting_match:
                        conflicting_faces += 1
                        face_diagnostics[result.track_id] = (
                            f"Face: {width}x{height}px {source_name} "
                            f"CONFLICT REJECTED {identity.similarity:.3f}"
                        )
                    elif identity.matched:
                        matched_faces += 1
                        face_diagnostics[result.track_id] = (
                            f"Face: {width}x{height}px {source_name} "
                            f"MATCH {identity.similarity:.3f}"
                        )
                    else:
                        decision = (
                            identity.rejection_reason or "UNKNOWN"
                        )
                        if decision == "AMBIGUOUS":
                            ambiguous_faces += 1
                        else:
                            unknown_faces += 1
                        face_diagnostics[result.track_id] = (
                            f"Face: {width}x{height}px {source_name} "
                            f"{decision} {identity.similarity:.3f}"
                        )

                    before_confirmed = current_state.confirmed
                    state = identity_manager.observe(
                        result.track_id,
                        identity,
                        frame_index,
                    )

                    if (
                        state.confirmed
                        and not before_confirmed
                        and state.person_id
                    ):
                        track = track_by_id[result.track_id]
                        event_source = track_camera_label(
                            source_label,
                            track,
                        )
                        event_track_id = track_event_id(track)
                        LOGGER.info(
                            "IDENTITY_CONFIRMED | "
                            "session=%s | "
                            "source=%s | "
                            "track=%d | "
                            "person=%s | "
                            "person_id=%s | "
                            "similarity=%.3f",
                            session_id[:8],
                            event_source,
                            event_track_id,
                            state.full_name,
                            state.person_id[:8],
                            state.similarity,
                        )

                        store_event_safely(
                            event_repository,
                            RecognitionEvent(
                                session_id=session_id,
                                event_type="IDENTITY_CONFIRMED",
                                camera_source=event_source,
                                local_track_id=event_track_id,
                                person_id=state.person_id,
                                full_name=state.full_name,
                                similarity=state.similarity,
                                detection_confidence=track.confidence,
                                frame_index=frame_index,
                                occurred_at=captured_at,
                            ),
                        )

                detected_faces = sum(
                    result.detection is not None
                    for result in current_face_results
                )
                ready_faces = batch_counts["READY"]
                face_batch_summary = (
                    f"{detected_faces} found / {ready_faces} ready / "
                    f"{matched_faces} matched"
                )
                face_diagnostic_counts["MATCHED"] += matched_faces
                face_diagnostic_counts["CONFLICT"] += conflicting_faces
                face_diagnostic_counts["UNKNOWN_MATCH"] += unknown_faces
                face_diagnostic_counts["AMBIGUOUS"] += ambiguous_faces

                if (
                    frame_index - last_face_diagnostic_log_frame
                    >= settings.face_diagnostic_log_interval
                ):
                    log_face_diagnostic_summary(
                        face_diagnostic_counts,
                        session_id=session_id,
                        source_label=source_label,
                        reason="periodic",
                    )
                    face_diagnostic_counts.clear()
                    last_face_diagnostic_log_frame = frame_index

                if args.show_faces:
                    for result in current_face_results:
                        draw_face_stage_result(frame, result)

            if activity_processor is not None:
                capture_seconds = datetime.fromisoformat(
                    captured_at
                ).timestamp()
                camera_observations: dict[str, list[TrackObservation]] = {}
                camera_names = {
                    track_camera_label(source_label, track)
                    for track in tracks
                }
                if input_layout == "2x2":
                    camera_names.update(
                        f"{source_label} [Q{index}]"
                        for index in range(1, 5)
                    )
                else:
                    camera_names.add(source_label)
                for camera_name in camera_names:
                    camera_observations[camera_name] = []
                all_observations: dict[str, list[TrackObservation]] = {
                    camera_name: [] for camera_name in camera_names
                }
                for track in tracks:
                    state = identity_manager.get(track.track_id)
                    camera_name = track_camera_label(source_label, track)
                    person_key = state.person_id or (
                        f"unknown_track_{track.track_id}"
                    )
                    conversation_person_names[person_key] = (
                        state.full_name
                        if state.person_id and state.full_name
                        else (
                            state.person_id
                            if state.person_id
                            else f"UNKNOWN (track {track.track_id})"
                        )
                    )
                    observation = TrackObservation(
                        session_id=session_id,
                        camera_id=camera_name,
                        person_id=person_key,
                        timestamp=capture_seconds,
                        box=tuple(float(value) for value in track.box),
                    )
                    if state.person_id:
                        camera_observations[camera_name].append(observation)
                    all_observations[camera_name].append(observation)
                for camera_name, observations in camera_observations.items():
                    if activity_processor is not None:
                        interaction_events = activity_processor.process_frame(
                            session_id=session_id,
                            camera_id=camera_name,
                            timestamp=capture_seconds,
                            observations=observations,
                        )
                    visual_events = visual_interaction_engine.process_frame(
                        session_id=session_id,
                        camera_id=camera_name,
                        timestamp=capture_seconds,
                        observations=all_observations[camera_name],
                    )
                    update_active_interactions(active_interactions, visual_events)
                    log_interaction_events(
                        visual_events,
                        conversation_logger,
                        conversation_person_names,
                    )

            expired_states = (
                identity_manager.expire(
                    frame_index
                )
            )

            for state in expired_states:
                face_fusion.clear(state.track_id)
                ended_at = track_last_seen_at.pop(
                    state.track_id,
                    captured_at,
                )
                event_source = track_source_labels.pop(
                    state.track_id,
                    source_label,
                )
                event_track_id = track_event_ids.pop(
                    state.track_id,
                    state.track_id,
                )
                face_diagnostics.pop(state.track_id, None)

                LOGGER.info(
                    "TRACK_ENDED | "
                    "session=%s | "
                    "source=%s | "
                    "track=%d | "
                    "identity=%s | "
                    "person_id=%s",
                    session_id[:8],
                    event_source,
                    event_track_id,
                    state.full_name,
                    (
                        state.person_id[:8]
                        if state.person_id
                        else "none"
                    ),
                )

                store_event_safely(
                    event_repository,
                    RecognitionEvent(
                        session_id=session_id,
                        event_type="TRACK_ENDED",
                        camera_source=event_source,
                        local_track_id=event_track_id,
                        person_id=state.person_id,
                        full_name=(
                            state.full_name
                            if state.confirmed
                            else None
                        ),
                        similarity=(
                            state.similarity
                            if state.confirmed
                            else None
                        ),
                        frame_index=state.last_seen_frame,
                        occurred_at=ended_at,
                    ),
                )

            for track in tracks:
                x1, y1, x2, y2 = (
                    track.box
                )
                # OpenCV requires plain integer pixel coordinates. The
                # quadrant tracker may return NumPy scalar values after
                # resizing, which OpenCV rejects in rectangle().
                x1, y1, x2, y2 = (
                    int(round(float(x1))),
                    int(round(float(y1))),
                    int(round(float(x2))),
                    int(round(float(y2))),
                )

                state = (
                    identity_manager.get(
                        track.track_id
                    )
                )

                progress = (
                    identity_manager
                    .vote_progress(
                        track.track_id
                    )
                )

                color = state_color(progress)
                face_detail = face_diagnostics.get(
                    track.track_id,
                    "Face: waiting for recognition pass",
                )
                local_track_line = track_overlay_line(track)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )

                # Keep the camera view readable: identity and track only.
                display_name = (
                    state.full_name
                    if progress.status == "CONFIRMED"
                    else (
                        progress.candidate_name
                        if progress.status == "PENDING"
                        else "UNKNOWN"
                    )
                )
                lines = [display_name, local_track_line]

                draw_text_block(
                    frame,
                    track.box,
                    lines,
                    color,
                )

            if active_interactions:
                draw_active_interactions(
                    frame,
                    tracks,
                    identity_manager,
                    active_interactions,
                    source_label,
                )

            if args.show_status_panel and not args.hide_status_panel:
                draw_status_panel(
                    frame,
                    source_label=(
                        f"{source_label} [2x2 split]"
                        if input_layout == "2x2"
                        else source_label
                    ),
                    fps=smoothed_fps,
                    active_tracks=len(tracks),
                    enrolled_people=(
                        recognizer.person_count
                    ),
                    stored_embeddings=(
                        recognizer
                        .embedding_count
                    ),
                    threshold=(
                        settings
                        .face_match_threshold
                    ),
                    recognition_interval=(
                        settings
                        .face_recognition_interval
                    ),
                    face_batch_summary=face_batch_summary,
                    roi_upscale=(
                        settings.live_face_roi_upscale
                    ),
                    identity_min_size=(
                        settings.live_face_identity_min_size
                    ),
                    yolo_image_size=settings.yolo_image_size,
                    source_stride=args.frame_stride,
                )

            if output_path is not None:
                if writer is None:
                    writer = create_writer(
                        output_path,
                        frame,
                        fps_for_writer,
                    )

                writer.write(frame)

            if args.headless:
                key = -1
            else:
                cv2.imshow(
                    "CCTV Face Identity",
                    frame,
                )
                key = cv2.waitKey(1) & 0xFF

            if key in {
                ord("q"),
                27,
            }:
                break

            if key == ord("r"):
                recognizer.refresh()

                LOGGER.info(
                    "Embedding cache refreshed: "
                    "%d people / "
                    "%d embeddings.",
                    recognizer.person_count,
                    recognizer
                    .embedding_count,
                )

        LOGGER.info(
            "Source frames decoded=%d | processed=%d | stride=%d.",
            source_frame_index,
            processed_frame_count,
            args.frame_stride,
        )

    except (
        FileNotFoundError,
        RuntimeError,
        NDIUnavailableError,
        NDISourceNotFoundError,
    ) as exc:
        LOGGER.error("%s", exc)
        return 1

    finally:
        log_face_diagnostic_summary(
            face_diagnostic_counts,
            session_id=session_id,
            source_label=source_label,
            reason="stream_closed",
        )
        face_diagnostic_counts.clear()

        if identity_manager is not None:
            for state in identity_manager.close_all():
                face_fusion.clear(state.track_id)
                ended_at = track_last_seen_at.pop(
                    state.track_id,
                    last_captured_at
                    or datetime.now(timezone.utc).isoformat(
                        timespec="milliseconds"
                    ),
                )
                event_source = track_source_labels.pop(
                    state.track_id,
                    source_label,
                )
                event_track_id = track_event_ids.pop(
                    state.track_id,
                    state.track_id,
                )

                LOGGER.info(
                    "TRACK_ENDED | "
                    "session=%s | "
                    "source=%s | "
                    "track=%d | "
                    "identity=%s | "
                    "person_id=%s | "
                    "reason=stream_closed",
                    session_id[:8],
                    event_source,
                    event_track_id,
                    state.full_name,
                    (
                        state.person_id[:8]
                        if state.person_id
                        else "none"
                    ),
                )

                store_event_safely(
                    event_repository,
                    RecognitionEvent(
                        session_id=session_id,
                        event_type="TRACK_ENDED",
                        camera_source=event_source,
                        local_track_id=event_track_id,
                        person_id=state.person_id,
                        full_name=(
                            state.full_name
                            if state.confirmed
                            else None
                        ),
                        similarity=(
                            state.similarity
                            if state.confirmed
                            else None
                        ),
                        frame_index=state.last_seen_frame,
                        occurred_at=ended_at,
                    ),
                )

        if activity_processor is not None:
            log_interaction_events(
                activity_processor.flush(),
                conversation_logger,
                conversation_person_names,
            )

        if visual_interaction_engine is not None:
            log_interaction_events(
                visual_interaction_engine.flush(),
                conversation_logger,
                conversation_person_names,
            )

        conversation_logger.close()

        face_fusion.clear_all()

        if writer is not None:
            writer.release()

        if source is not None:
            source.release()

        if not args.headless:
            cv2.destroyAllWindows()

    if output_path is not None:
        LOGGER.info(
            "Output saved: %s",
            output_path,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
