"""Ultralytics YOLO person detection and ByteTrack wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import cv2
from ultralytics import YOLO


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackedPerson:
    track_id: int
    box: tuple[int, int, int, int]
    confidence: float
    local_track_id: int | None = None
    quadrant_index: int | None = None
    quadrant_epoch: int = 0


@dataclass(frozen=True)
class FrameQuadrant:
    """One row-major tile from a 2x2 camera mosaic."""

    index: int
    bounds: tuple[int, int, int, int]
    image: np.ndarray


class PersonTrackingBackend(Protocol):
    """Minimal tracker interface used by the quadrant orchestrator."""

    def track(self, frame: np.ndarray) -> list[TrackedPerson]: ...


def split_frame_2x2(frame: np.ndarray) -> list[FrameQuadrant]:
    """Split a BGR frame without dropping odd edge rows or columns."""

    if (
        not isinstance(frame, np.ndarray)
        or frame.ndim != 3
        or frame.shape[2] != 3
    ):
        raise ValueError("A 2x2 layout requires a three-channel BGR frame.")

    height, width = frame.shape[:2]
    if height < 2 or width < 2:
        raise ValueError("A 2x2 layout requires a frame of at least 2x2 pixels.")

    x_mid = width // 2
    y_mid = height // 2
    bounds = [
        (0, 0, x_mid, y_mid),
        (x_mid, 0, width, y_mid),
        (0, y_mid, x_mid, height),
        (x_mid, y_mid, width, height),
    ]

    return [
        FrameQuadrant(
            index=index,
            bounds=box,
            image=np.ascontiguousarray(
                frame[box[1] : box[3], box[0] : box[2]]
            ),
        )
        for index, box in enumerate(bounds, start=1)
    ]


def has_meaningful_video_content(
    image: np.ndarray,
    *,
    dark_luma_threshold: float = 24.0,
    maximum_dark_fraction: float = 0.90,
    minimum_luma_std: float = 3.0,
    maximum_dominant_fraction: float = 0.90,
) -> bool:
    """Cheaply distinguish live video from common signal-loss screens.

    CCTV recorders commonly fill a missing channel with black, a flat blue
    tile, or another nearly uniform color plus a small text label. Sampling
    and quantizing the tile catches those cases without decoding text or
    running a detector.
    """

    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.size == 0
    ):
        raise ValueError("Video-content checks require a non-empty BGR image.")
    if dark_luma_threshold < 0.0:
        raise ValueError("dark_luma_threshold cannot be negative.")
    if not 0.0 <= maximum_dark_fraction <= 1.0:
        raise ValueError("maximum_dark_fraction must be between 0 and 1.")
    if minimum_luma_std < 0.0:
        raise ValueError("minimum_luma_std cannot be negative.")
    if not 0.0 <= maximum_dominant_fraction <= 1.0:
        raise ValueError(
            "maximum_dominant_fraction must be between 0 and 1."
        )

    sampled = np.concatenate(
        [
            image[dy::8, dx::8].reshape(-1, 3)
            for dy, dx in ((0, 0), (1, 0), (0, 1), (1, 1))
        ],
        axis=0,
    )
    sampled_float = sampled.astype(np.float32, copy=False)
    luma = (
        0.114 * sampled_float[:, 0]
        + 0.587 * sampled_float[:, 1]
        + 0.299 * sampled_float[:, 2]
    )
    dark_fraction = float(np.mean(luma <= dark_luma_threshold))
    luma_std = float(np.std(luma))

    # A 4-bit value per BGR channel is sufficient to recognize a recorder's
    # flat fill color while tolerating compression noise around it.
    quantized = sampled.astype(np.uint16, copy=False) >> 4
    color_codes = (
        quantized[:, 0] * 256
        + quantized[:, 1] * 16
        + quantized[:, 2]
    )
    counts = np.bincount(color_codes.ravel(), minlength=4096)
    dominant_fraction = float(counts.max() / color_codes.size)

    unavailable = (
        dark_fraction >= maximum_dark_fraction
        or luma_std <= minimum_luma_std
        or dominant_fraction >= maximum_dominant_fraction
    )
    return not unavailable


class PersonTracker:
    """Detect only class 0 (person) and persist ByteTrack IDs."""

    def __init__(
        self,
        model_name: str,
        confidence: float,
        tracker_config: str,
        image_size: int = 640,
    ) -> None:
        if image_size < 32:
            raise ValueError("image_size must be at least 32 pixels.")
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.tracker_config = tracker_config
        self.image_size = image_size

    def track(self, frame: np.ndarray) -> list[TrackedPerson]:
        results = self.model.track(
            source=frame,
            persist=True,
            classes=[0],
            conf=self.confidence,
            tracker=self.tracker_config,
            imgsz=self.image_size,
            verbose=False,
        )

        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        track_ids = boxes.id.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy()

        tracked: list[TrackedPerson] = []
        for coords, track_id, confidence in zip(
            xyxy, track_ids, confidences, strict=True
        ):
            x1, y1, x2, y2 = (int(round(value)) for value in coords)
            tracked.append(
                TrackedPerson(
                    track_id=int(track_id),
                    box=(x1, y1, x2, y2),
                    confidence=float(confidence),
                )
            )
        return tracked


class QuadrantPersonTracker:
    """
    Track a 2x2 camera mosaic as four independent camera streams.

    Four persistent trackers are intentional. Feeding four NumPy crops to
    one Ultralytics ``track`` call uses one ByteTrack state and mixes camera
    timelines. Returned boxes use mosaic coordinates while internal IDs are
    mapped to collision-free session IDs.
    """

    def __init__(
        self,
        model_name: str,
        confidence: float,
        tracker_config: str,
        image_size: int = 640,
        *,
        tracker_factory: Callable[
            [str, float, str, int], PersonTrackingBackend
        ]
        | None = None,
        suppress_startup_inactive: bool = True,
        inactive_debounce_frames: int = 3,
        recovery_debounce_frames: int = 2,
        quadrant_upscale: float = 2.0,
    ) -> None:
        if inactive_debounce_frames < 1:
            raise ValueError("inactive_debounce_frames must be positive.")
        if recovery_debounce_frames < 1:
            raise ValueError("recovery_debounce_frames must be positive.")
        if quadrant_upscale < 1.0 or quadrant_upscale > 3.0:
            raise ValueError("quadrant_upscale must be between 1.0 and 3.0.")
        factory = tracker_factory or PersonTracker
        self.trackers = [
            factory(
                model_name,
                confidence,
                tracker_config,
                image_size,
            )
            for _ in range(4)
        ]
        self._initialized = False
        self._frame_shape: tuple[int, int] | None = None
        self.suppress_startup_inactive = suppress_startup_inactive
        self.inactive_debounce_frames = inactive_debounce_frames
        self.recovery_debounce_frames = recovery_debounce_frames
        self.quadrant_upscale = quadrant_upscale
        initial_state = (
            "STARTUP" if suppress_startup_inactive else "ONLINE"
        )
        self._quadrant_states = [initial_state for _ in range(4)]
        self._unavailable_streaks = [0] * 4
        self._available_streaks = [0] * 4
        self._quadrant_epochs = [0] * 4

    def _transition_quadrant(
        self,
        zero_index: int,
        quadrant: FrameQuadrant,
        tracker: PersonTrackingBackend,
    ) -> bool:
        """Update one camera-health state and return whether to infer it."""

        if not self.suppress_startup_inactive:
            return True

        meaningful = has_meaningful_video_content(quadrant.image)
        state = self._quadrant_states[zero_index]

        if state == "OFFLINE":
            self._unavailable_streaks[zero_index] = 0
            if not meaningful:
                self._available_streaks[zero_index] = 0
                return False

            self._available_streaks[zero_index] += 1
            if (
                self._available_streaks[zero_index]
                < self.recovery_debounce_frames
            ):
                return False

            self._available_streaks[zero_index] = 0
            self._quadrant_states[zero_index] = "ONLINE"
            self._quadrant_epochs[zero_index] += 1
            LOGGER.info(
                "Video content recovered in quadrant Q%d; person tracking "
                "activated with recovery epoch %d.",
                quadrant.index,
                self._quadrant_epochs[zero_index],
            )
            return True

        if meaningful:
            self._unavailable_streaks[zero_index] = 0
            self._available_streaks[zero_index] = 0
            if state == "STARTUP":
                self._quadrant_states[zero_index] = "ONLINE"
            return True

        self._available_streaks[zero_index] = 0
        self._unavailable_streaks[zero_index] += 1

        # Advance the underlying tracker's clock while a loss transition is
        # being confirmed, but never allow detections from a loss screen.
        tracker.track(np.zeros_like(quadrant.image))
        if (
            self._unavailable_streaks[zero_index]
            >= self.inactive_debounce_frames
        ):
            self._quadrant_states[zero_index] = "OFFLINE"
            self._unavailable_streaks[zero_index] = 0
            LOGGER.info(
                "Suppressing unavailable quadrant Q%d after %d consecutive "
                "loss frames; pixels remain monitored for recovery.",
                quadrant.index,
                self.inactive_debounce_frames,
            )
        return False

    def _initialize_trackers(
        self,
        quadrants: list[FrameQuadrant],
    ) -> None:
        """
        Initialize all Ultralytics tracker contexts before real detections.

        ByteTrack's underlying ID counter is class-global. Each newly created
        predictor resets it, so allowing Q1 to emit real IDs before Q2-Q4 are
        initialized can later reuse a Q1 ID. Blank first calls ensure every
        reset happens before any real track exists.
        """

        for quadrant, tracker in zip(
            quadrants,
            self.trackers,
            strict=True,
        ):
            tracker.track(np.zeros_like(quadrant.image))
        self._initialized = True

    def _global_track_id(
        self,
        quadrant_index: int,
        local_track_id: int,
        recovery_epoch: int = 0,
    ) -> int:
        if local_track_id < 1:
            raise ValueError("Quadrant tracker IDs must be positive.")
        if recovery_epoch < 0:
            raise ValueError("recovery_epoch cannot be negative.")
        base_track_id = (local_track_id - 1) * 4 + quadrant_index
        return (recovery_epoch << 32) | base_track_id

    def track(self, frame: np.ndarray) -> list[TrackedPerson]:
        quadrants = split_frame_2x2(frame)
        frame_shape = frame.shape[:2]
        if self._frame_shape is None:
            self._frame_shape = frame_shape
        elif frame_shape != self._frame_shape:
            raise RuntimeError(
                "The 2x2 source resolution changed from "
                f"{self._frame_shape[1]}x{self._frame_shape[0]} to "
                f"{frame_shape[1]}x{frame_shape[0]}. Restart quadrant "
                "tracking so stale camera coordinates cannot be reused."
            )
        if not self._initialized:
            self._initialize_trackers(quadrants)
        tracked: list[TrackedPerson] = []

        for zero_index, (quadrant, tracker) in enumerate(
            zip(
                quadrants,
                self.trackers,
                strict=True,
            )
        ):
            if not self._transition_quadrant(
                zero_index,
                quadrant,
                tracker,
            ):
                continue

            left, top, right, bottom = quadrant.bounds
            tile_width = right - left
            tile_height = bottom - top

            # Each NDI tile is inferred independently. Upscaling the tile
            # before YOLO gives distant people more pixels; boxes are mapped
            # back to the original mosaic coordinates below.
            tile_image = quadrant.image
            scale = self.quadrant_upscale
            if scale > 1.0:
                tile_image = cv2.resize(
                    tile_image,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
            for person in tracker.track(tile_image):
                local_track_id = int(person.track_id)
                x1, y1, x2, y2 = (value / scale for value in person.box)
                x1 = int(round(max(0, min(tile_width, x1))))
                y1 = int(round(max(0, min(tile_height, y1))))
                x2 = int(round(max(0, min(tile_width, x2))))
                y2 = int(round(max(0, min(tile_height, y2))))
                if x2 <= x1 or y2 <= y1:
                    continue

                tracked.append(
                    TrackedPerson(
                        track_id=self._global_track_id(
                            quadrant.index,
                            local_track_id,
                            self._quadrant_epochs[zero_index],
                        ),
                        box=(
                            x1 + left,
                            y1 + top,
                            x2 + left,
                            y2 + top,
                        ),
                        confidence=person.confidence,
                        local_track_id=local_track_id,
                        quadrant_index=quadrant.index,
                        quadrant_epoch=self._quadrant_epochs[zero_index],
                    )
                )

        return tracked
