"""Quality-gated face extraction for tracked people in live CCTV frames."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from src.association import (
    Box,
    HasPersonBox,
    assign_faces_to_tracks,
    box_iou,
    face_matches_person,
)
from src.face_engine import (
    FaceDetection,
    FaceEmbeddingResult,
    FaceEngineError,
)


FaceSource = Literal["full", "roi"]
FaceStageStatus = Literal[
    "READY",
    "NO_FACE",
    "FACE_TOO_SMALL",
    "INVALID_ROI",
    "DETECTION_FAILED",
    "EMBEDDING_FAILED",
]


class LiveFaceEngine(Protocol):
    """Face-engine operations needed by the live pipeline."""

    def detect_faces(self, image: np.ndarray) -> list[FaceDetection]: ...

    def detect_faces_in_region(
        self,
        image: np.ndarray,
        region: Box,
        *,
        upscale: float = 1.0,
        score_threshold: float | None = None,
        max_dimension: int | None = None,
    ) -> list[FaceDetection]: ...

    def extract_embedding(
        self,
        image: np.ndarray,
        detection: FaceDetection,
    ) -> FaceEmbeddingResult: ...


@dataclass(frozen=True)
class FaceCandidate:
    """One frame-coordinate face candidate and its detection source."""

    detection: FaceDetection
    source: FaceSource


@dataclass(frozen=True)
class TrackFaceResult:
    """Terminal face-stage outcome for one tracked person."""

    track_id: int
    status: FaceStageStatus
    detection: FaceDetection | None = None
    source: FaceSource | None = None
    embedding: np.ndarray | None = None
    error: str | None = None

    @property
    def face_size(self) -> tuple[int, int] | None:
        if self.detection is None:
            return None
        return self.detection.width, self.detection.height


def upper_body_region(
    frame_shape: tuple[int, ...],
    person_box: Box,
    *,
    upper_body_ratio: float,
    horizontal_padding_ratio: float = 0.12,
    top_padding_ratio: float = 0.05,
) -> Box | None:
    """Return a padded, frame-clamped upper-body search region."""

    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain height and width.")
    if not 0.0 < upper_body_ratio <= 1.0:
        raise ValueError("upper_body_ratio must be between 0 and 1.")
    if horizontal_padding_ratio < 0.0 or top_padding_ratio < 0.0:
        raise ValueError("ROI padding ratios cannot be negative.")

    frame_height, frame_width = frame_shape[:2]
    px1, py1, px2, py2 = person_box
    person_width = px2 - px1
    person_height = py2 - py1
    if person_width <= 0 or person_height <= 0:
        return None

    left = math.floor(px1 - person_width * horizontal_padding_ratio)
    right = math.ceil(px2 + person_width * horizontal_padding_ratio)
    top = math.floor(py1 - person_height * top_padding_ratio)
    bottom = math.ceil(py1 + person_height * upper_body_ratio)

    left = max(0, min(frame_width, left))
    top = max(0, min(frame_height, top))
    right = max(0, min(frame_width, right))
    bottom = max(0, min(frame_height, bottom))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def deduplicate_candidates(
    candidates: list[FaceCandidate],
    *,
    iou_threshold: float = 0.45,
    identity_min_size: int | None = None,
) -> list[FaceCandidate]:
    """Remove repeated ROI detections of the same global face."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1.")
    if identity_min_size is not None and identity_min_size < 1:
        raise ValueError("identity_min_size must be positive when provided.")

    def quality(candidate: FaceCandidate) -> tuple[int, float, int, int]:
        detection = candidate.detection
        minimum_dimension = min(detection.width, detection.height)
        identity_eligible = int(
            identity_min_size is not None
            and minimum_dimension >= identity_min_size
        )
        return (
            identity_eligible,
            detection.score,
            minimum_dimension,
            detection.width * detection.height,
        )

    ordered = sorted(
        candidates,
        key=quality,
        reverse=True,
    )
    kept: list[FaceCandidate] = []

    for candidate in ordered:
        if any(
            box_iou(candidate.detection.box, item.detection.box)
            >= iou_threshold
            for item in kept
        ):
            continue
        kept.append(candidate)

    return kept


class LiveFacePipeline:
    """
    Run full-frame detection, tracked-person ROI fallback and quality gates.

    ROI enlargement is a detection aid only. ``FaceEngine`` maps every ROI
    detection back to the source frame before SFace alignment, and the live
    identity floor is always evaluated in original frame pixels.
    """

    def __init__(
        self,
        engine: LiveFaceEngine,
        *,
        detection_min_size: int,
        identity_min_size: int,
        roi_upscale: float,
        roi_upper_body_ratio: float,
        roi_score_threshold: float,
        roi_max_dimension: int = 640,
        dedup_iou_threshold: float = 0.45,
    ) -> None:
        if detection_min_size < 1:
            raise ValueError("detection_min_size must be positive.")
        if identity_min_size < detection_min_size:
            raise ValueError(
                "identity_min_size cannot be below detection_min_size."
            )
        if not 1.0 <= roi_upscale <= 4.0:
            raise ValueError("roi_upscale must be between 1.0 and 4.0.")
        if not 0.0 < roi_upper_body_ratio <= 1.0:
            raise ValueError(
                "roi_upper_body_ratio must be between 0 and 1."
            )
        if not 0.0 <= roi_score_threshold <= 1.0:
            raise ValueError(
                "roi_score_threshold must be between 0 and 1."
            )
        if roi_max_dimension < 32:
            raise ValueError("roi_max_dimension must be at least 32.")
        if not 0.0 < dedup_iou_threshold <= 1.0:
            raise ValueError(
                "dedup_iou_threshold must be between 0 and 1."
            )

        self.engine = engine
        self.detection_min_size = detection_min_size
        self.identity_min_size = identity_min_size
        self.roi_upscale = roi_upscale
        self.roi_upper_body_ratio = roi_upper_body_ratio
        self.roi_score_threshold = roi_score_threshold
        self.roi_max_dimension = roi_max_dimension
        self.dedup_iou_threshold = dedup_iou_threshold

    def _priority(self, detection: FaceDetection) -> float:
        minimum_dimension = min(detection.width, detection.height)
        if minimum_dimension >= self.identity_min_size:
            return 2.0
        if minimum_dimension >= self.detection_min_size:
            return 1.0
        return 0.0

    def _assign(
        self,
        candidates: list[FaceCandidate],
        tracks: list[HasPersonBox],
    ) -> dict[int, FaceCandidate]:
        if not candidates or not tracks:
            return {}

        assignments = assign_faces_to_tracks(
            [item.detection.box for item in candidates],
            tracks,
            self.roi_upper_body_ratio,
            [self._priority(item.detection) for item in candidates],
        )
        return {
            track.track_id: candidates[face_index]
            for face_index, track in assignments.items()
        }

    def process(
        self,
        frame: np.ndarray,
        tracks: list[HasPersonBox],
    ) -> list[TrackFaceResult]:
        """Return exactly one terminal face-stage result per input track."""

        if not tracks:
            return []

        full_detection_error: str | None = None
        try:
            full_candidates = [
                FaceCandidate(detection, "full")
                for detection in self.engine.detect_faces(frame)
                if min(detection.width, detection.height)
                >= self.detection_min_size
            ]
        except FaceEngineError as exc:
            full_candidates = []
            full_detection_error = str(exc)

        assigned = self._assign(full_candidates, tracks)
        unresolved = [
            track
            for track in tracks
            if track.track_id not in assigned
            or self._priority(assigned[track.track_id].detection) < 2.0
        ]
        unresolved_ids = {track.track_id for track in unresolved}
        resolved_candidates = [
            candidate
            for track_id, candidate in assigned.items()
            if track_id not in unresolved_ids
        ]
        provisional_candidates = [
            assigned[track.track_id]
            for track in unresolved
            if track.track_id in assigned
        ]
        invalid_roi_ids: set[int] = set()
        roi_errors: dict[int, str] = {}
        roi_candidates: list[FaceCandidate] = []

        for track in unresolved:
            region = upper_body_region(
                frame.shape,
                track.box,
                upper_body_ratio=self.roi_upper_body_ratio,
            )
            if region is None:
                invalid_roi_ids.add(track.track_id)
                continue

            try:
                detections = self.engine.detect_faces_in_region(
                    frame,
                    region,
                    upscale=self.roi_upscale,
                    score_threshold=self.roi_score_threshold,
                    max_dimension=self.roi_max_dimension,
                )
            except FaceEngineError as exc:
                roi_errors[track.track_id] = str(exc)
                continue

            for detection in detections:
                if (
                    min(detection.width, detection.height)
                    < self.detection_min_size
                ):
                    continue
                if not face_matches_person(
                    detection.box,
                    track.box,
                    self.roi_upper_body_ratio,
                ):
                    continue
                if any(
                    box_iou(
                        detection.box,
                        resolved.detection.box,
                    )
                    >= self.dedup_iou_threshold
                    for resolved in resolved_candidates
                ):
                    continue
                roi_candidates.append(FaceCandidate(detection, "roi"))

        fallback_candidates = deduplicate_candidates(
            provisional_candidates + roi_candidates,
            iou_threshold=self.dedup_iou_threshold,
            identity_min_size=self.identity_min_size,
        )
        fallback_assignments = self._assign(
            fallback_candidates,
            unresolved,
        )
        for track in unresolved:
            assigned.pop(track.track_id, None)
        assigned.update(fallback_assignments)

        results: list[TrackFaceResult] = []

        for track in tracks:
            candidate = assigned.get(track.track_id)
            if candidate is None:
                if track.track_id in invalid_roi_ids:
                    status: FaceStageStatus = "INVALID_ROI"
                    error = "Tracked person box has no visible ROI."
                elif track.track_id in roi_errors:
                    status = "DETECTION_FAILED"
                    error = roi_errors[track.track_id]
                elif full_detection_error is not None:
                    status = "DETECTION_FAILED"
                    error = full_detection_error
                else:
                    status = "NO_FACE"
                    error = None

                results.append(
                    TrackFaceResult(
                        track_id=track.track_id,
                        status=status,
                        error=error,
                    )
                )
                continue

            detection = candidate.detection
            minimum_dimension = min(detection.width, detection.height)
            if minimum_dimension < self.identity_min_size:
                results.append(
                    TrackFaceResult(
                        track_id=track.track_id,
                        status="FACE_TOO_SMALL",
                        detection=detection,
                        source=candidate.source,
                        error=(
                            f"Original face is {detection.width}x"
                            f"{detection.height}px; identity requires "
                            f"{self.identity_min_size}px."
                        ),
                    )
                )
                continue

            try:
                extracted = self.engine.extract_embedding(frame, detection)
            except FaceEngineError as exc:
                results.append(
                    TrackFaceResult(
                        track_id=track.track_id,
                        status="EMBEDDING_FAILED",
                        detection=detection,
                        source=candidate.source,
                        error=str(exc),
                    )
                )
                continue

            results.append(
                TrackFaceResult(
                    track_id=track.track_id,
                    status="READY",
                    detection=detection,
                    source=candidate.source,
                    embedding=extracted.embedding,
                )
            )

        return results
