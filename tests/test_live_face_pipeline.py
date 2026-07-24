from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.face_engine import (
    EmbeddingGenerationError,
    FaceDetection,
    FaceEmbeddingResult,
)
from src.live_face_pipeline import LiveFacePipeline


@dataclass(frozen=True)
class Track:
    track_id: int
    box: tuple[int, int, int, int]


def face(
    box: tuple[int, int, int, int],
    score: float = 0.9,
) -> FaceDetection:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    raw = np.asarray(
        [
            x1,
            y1,
            width,
            height,
            x1 + width * 0.30,
            y1 + height * 0.35,
            x1 + width * 0.70,
            y1 + height * 0.35,
            x1 + width * 0.50,
            y1 + height * 0.55,
            x1 + width * 0.35,
            y1 + height * 0.75,
            x1 + width * 0.65,
            y1 + height * 0.75,
            score,
        ],
        dtype=np.float32,
    )
    return FaceDetection(box, score, raw)


class FakeEngine:
    def __init__(
        self,
        *,
        full: list[FaceDetection] | None = None,
        roi_batches: list[
            list[FaceDetection] | Exception
        ] | None = None,
        failing_boxes: set[tuple[int, int, int, int]] | None = None,
    ) -> None:
        self.full = full or []
        self.roi_batches = list(roi_batches or [])
        self.failing_boxes = failing_boxes or set()
        self.roi_calls: list[
            tuple[
                tuple[int, int, int, int],
                float,
                float | None,
                int | None,
            ]
        ] = []
        self.extract_calls: list[
            tuple[np.ndarray, FaceDetection]
        ] = []

    def detect_faces(self, image: np.ndarray) -> list[FaceDetection]:
        return self.full

    def detect_faces_in_region(
        self,
        image: np.ndarray,
        region: tuple[int, int, int, int],
        *,
        upscale: float = 1.0,
        score_threshold: float | None = None,
        max_dimension: int | None = None,
    ) -> list[FaceDetection]:
        self.roi_calls.append(
            (region, upscale, score_threshold, max_dimension)
        )
        batch = self.roi_batches.pop(0) if self.roi_batches else []
        if isinstance(batch, Exception):
            raise batch
        return batch

    def extract_embedding(
        self,
        image: np.ndarray,
        detection: FaceDetection,
    ) -> FaceEmbeddingResult:
        self.extract_calls.append((image, detection))
        if detection.box in self.failing_boxes:
            raise EmbeddingGenerationError("synthetic failure")
        vector = np.zeros(128, dtype=np.float32)
        vector[0] = 1.0
        return FaceEmbeddingResult(
            detection,
            np.ones((112, 112, 3), dtype=np.uint8),
            vector,
        )


def pipeline(engine: FakeEngine) -> LiveFacePipeline:
    return LiveFacePipeline(
        engine,
        detection_min_size=12,
        identity_min_size=32,
        roi_upscale=2.0,
        roi_upper_body_ratio=0.55,
        roi_score_threshold=0.60,
        roi_max_dimension=640,
    )


def test_full_frame_face_preserves_near_camera_path() -> None:
    detected = face((45, 20, 85, 65))
    engine = FakeEngine(full=[detected])
    frame = np.zeros((240, 180, 3), dtype=np.uint8)

    results = pipeline(engine).process(
        frame,
        [Track(1, (20, 10, 120, 220))],
    )

    assert len(results) == 1
    assert results[0].status == "READY"
    assert results[0].source == "full"
    assert engine.roi_calls == []
    assert engine.extract_calls[0][0] is frame


def test_roi_fallback_uses_configured_scale_and_score_threshold() -> None:
    detected = face((50, 24, 88, 66))
    engine = FakeEngine(roi_batches=[[detected]])
    frame = np.zeros((240, 180, 3), dtype=np.uint8)

    result = pipeline(engine).process(
        frame,
        [Track(7, (20, 10, 120, 220))],
    )[0]

    assert result.status == "READY"
    assert result.source == "roi"
    assert len(engine.roi_calls) == 1
    _, upscale, threshold, max_dimension = engine.roi_calls[0]
    assert upscale == 2.0
    assert threshold == 0.60
    assert max_dimension == 640
    assert engine.extract_calls[0][0] is frame


def test_original_pixel_floor_rejects_enlarged_tiny_face() -> None:
    detected = face((52, 22, 82, 60))
    engine = FakeEngine(roi_batches=[[detected]])
    frame = np.zeros((240, 180, 3), dtype=np.uint8)

    result = pipeline(engine).process(
        frame,
        [Track(2, (20, 10, 120, 220))],
    )[0]

    assert result.status == "FACE_TOO_SMALL"
    assert result.face_size == (30, 38)
    assert result.embedding is None
    assert engine.extract_calls == []


def test_off_frame_track_reports_invalid_roi_without_detector_call() -> None:
    engine = FakeEngine()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    result = pipeline(engine).process(
        frame,
        [Track(9, (140, 120, 180, 220))],
    )[0]

    assert result.status == "INVALID_ROI"
    assert engine.roi_calls == []


def test_identity_eligible_face_wins_over_tiny_high_score_face() -> None:
    tiny = face((58, 28, 70, 42), score=0.99)
    eligible = face((45, 20, 85, 62), score=0.82)
    engine = FakeEngine(full=[tiny, eligible])
    frame = np.zeros((240, 180, 3), dtype=np.uint8)

    result = pipeline(engine).process(
        frame,
        [Track(3, (20, 10, 120, 220))],
    )[0]

    assert result.status == "READY"
    assert result.detection is eligible
    assert engine.extract_calls[0][1] is eligible


def test_sub_detection_floor_face_does_not_block_roi_fallback() -> None:
    unreliable = face((58, 28, 68, 38), score=0.99)
    roi_face = face((45, 20, 85, 62), score=0.75)
    engine = FakeEngine(
        full=[unreliable],
        roi_batches=[[roi_face]],
    )
    frame = np.zeros((240, 180, 3), dtype=np.uint8)

    result = pipeline(engine).process(
        frame,
        [Track(4, (20, 10, 120, 220))],
    )[0]

    assert result.status == "READY"
    assert result.source == "roi"
    assert result.detection is roi_face
    assert len(engine.roi_calls) == 1


def test_identity_ineligible_full_face_still_gets_roi_retry() -> None:
    small_full = face((50, 20, 70, 40), score=0.95)
    usable_roi = face((43, 13, 77, 49), score=0.78)
    engine = FakeEngine(
        full=[small_full],
        roi_batches=[[usable_roi]],
    )
    frame = np.zeros((240, 180, 3), dtype=np.uint8)

    result = pipeline(engine).process(
        frame,
        [Track(5, (20, 10, 120, 220))],
    )[0]

    assert result.status == "READY"
    assert result.source == "roi"
    assert result.detection is usable_roi
    assert len(engine.roi_calls) == 1


def test_roi_face_for_overlapping_unresolved_track_is_not_discarded() -> None:
    first_face = face((20, 15, 60, 55))
    second_face = face((85, 15, 119, 55))
    engine = FakeEngine(
        full=[first_face],
        roi_batches=[[second_face]],
    )
    frame = np.zeros((240, 220, 3), dtype=np.uint8)
    tracks = [
        Track(1, (0, 0, 120, 220)),
        Track(2, (80, 0, 200, 220)),
    ]

    results = pipeline(engine).process(frame, tracks)

    assert [result.status for result in results] == ["READY", "READY"]
    assert results[0].detection is first_face
    assert results[1].detection is second_face
    assert results[1].source == "roi"


def test_roi_duplicate_of_resolved_full_face_is_not_reused() -> None:
    shared_face = face((65, 15, 99, 55))
    engine = FakeEngine(
        full=[shared_face],
        roi_batches=[[shared_face]],
    )
    frame = np.zeros((240, 220, 3), dtype=np.uint8)
    tracks = [
        Track(1, (0, 0, 120, 220)),
        Track(2, (80, 0, 200, 220)),
    ]

    results = pipeline(engine).process(frame, tracks)

    assert [result.status for result in results] == ["READY", "NO_FACE"]
    assert len(engine.extract_calls) == 1


def test_roi_dedup_prefers_identity_eligible_candidate() -> None:
    too_small = face((45, 15, 76, 50), score=0.99)
    eligible = face((44, 14, 78, 50), score=0.82)
    engine = FakeEngine(roi_batches=[[too_small, eligible]])
    frame = np.zeros((240, 180, 3), dtype=np.uint8)

    result = pipeline(engine).process(
        frame,
        [Track(6, (20, 10, 120, 220))],
    )[0]

    assert result.status == "READY"
    assert result.detection is eligible


def test_overlapping_rois_do_not_emit_duplicate_face_votes() -> None:
    repeated = face((75, 15, 110, 55))
    engine = FakeEngine(roi_batches=[[repeated], [repeated]])
    frame = np.zeros((240, 200, 3), dtype=np.uint8)
    tracks = [
        Track(1, (0, 0, 100, 220)),
        Track(2, (70, 0, 170, 220)),
    ]

    results = pipeline(engine).process(frame, tracks)

    statuses = [result.status for result in results]
    assert statuses.count("READY") == 1
    assert statuses.count("NO_FACE") == 1
    assert len(engine.extract_calls) == 1


def test_embedding_failure_does_not_abort_other_track() -> None:
    first = face((20, 15, 60, 60))
    second = face((130, 15, 170, 60))
    engine = FakeEngine(
        full=[first, second],
        failing_boxes={first.box},
    )
    frame = np.zeros((240, 220, 3), dtype=np.uint8)
    tracks = [
        Track(1, (0, 0, 90, 220)),
        Track(2, (110, 0, 200, 220)),
    ]

    results = pipeline(engine).process(frame, tracks)

    assert [result.status for result in results] == [
        "EMBEDDING_FAILED",
        "READY",
    ]
    assert len(engine.extract_calls) == 2
