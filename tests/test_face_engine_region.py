from __future__ import annotations

import types

import numpy as np
import pytest

from src.face_engine import FaceDetection, FaceEngine, FaceEngineError


def detection(
    box: tuple[int, int, int, int],
    raw: list[float],
    score: float = 0.9,
) -> FaceDetection:
    return FaceDetection(
        box=box,
        score=score,
        raw=np.asarray(raw, dtype=np.float32),
    )


def test_region_detection_maps_box_and_landmarks_to_frame() -> None:
    engine = FaceEngine.__new__(FaceEngine)
    local = detection(
        (20, 10, 60, 50),
        [
            20,
            10,
            40,
            40,
            30,
            20,
            50,
            20,
            40,
            30,
            32,
            42,
            48,
            42,
            0.9,
        ],
    )
    seen_shapes: list[tuple[int, ...]] = []

    def fake_detect(image: np.ndarray) -> list[FaceDetection]:
        seen_shapes.append(image.shape)
        return [local]

    engine.detect_faces = fake_detect  # type: ignore[method-assign]
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    results = engine.detect_faces_in_region(
        frame,
        (50, 20, 150, 80),
        upscale=2.0,
    )

    assert seen_shapes == [(120, 200, 3)]
    assert len(results) == 1
    mapped = results[0]
    assert mapped.box == (60, 25, 80, 45)
    np.testing.assert_allclose(
        mapped.raw[:14],
        [
            60,
            25,
            20,
            20,
            65,
            30,
            75,
            30,
            70,
            35,
            66,
            41,
            74,
            41,
        ],
    )


@pytest.mark.parametrize(
    ("region", "upscale", "max_dimension", "expected_shape"),
    [
        ((0, 0, 200, 100), 2.0, 320, (160, 320, 3)),
        ((0, 0, 100, 80), 2.0, 640, (160, 200, 3)),
        ((0, 0, 800, 400), 2.0, 640, (400, 800, 3)),
        ((-100, -50, 200, 100), 2.0, 320, (160, 320, 3)),
    ],
)
def test_region_enlargement_honors_dimension_budget_without_downscaling(
    region: tuple[int, int, int, int],
    upscale: float,
    max_dimension: int,
    expected_shape: tuple[int, int, int],
) -> None:
    engine = FaceEngine.__new__(FaceEngine)
    seen_shapes: list[tuple[int, ...]] = []

    def fake_detect(image: np.ndarray) -> list[FaceDetection]:
        seen_shapes.append(image.shape)
        return []

    engine.detect_faces = fake_detect  # type: ignore[method-assign]
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)

    engine.detect_faces_in_region(
        frame,
        region,
        upscale=upscale,
        max_dimension=max_dimension,
    )

    assert seen_shapes == [expected_shape]


class FakeDetector:
    def __init__(self) -> None:
        self.threshold = 0.72
        self.changes: list[float] = []

    def getScoreThreshold(self) -> float:
        return self.threshold

    def setScoreThreshold(self, value: float) -> None:
        self.threshold = value
        self.changes.append(value)


def test_region_specific_score_threshold_is_restored() -> None:
    engine = FaceEngine.__new__(FaceEngine)
    detector = FakeDetector()
    engine._detector = detector

    def fake_detect(image: np.ndarray) -> list[FaceDetection]:
        assert detector.threshold == pytest.approx(0.60)
        return []

    engine.detect_faces = fake_detect  # type: ignore[method-assign]
    frame = np.zeros((40, 40, 3), dtype=np.uint8)

    assert engine.detect_faces_in_region(
        frame,
        (0, 0, 20, 20),
        upscale=2.0,
        score_threshold=0.60,
    ) == []
    assert detector.threshold == pytest.approx(0.72)
    assert detector.changes == [0.60, 0.72]


def test_region_score_threshold_is_restored_after_detector_error() -> None:
    engine = FaceEngine.__new__(FaceEngine)
    detector = FakeDetector()
    engine._detector = detector

    def fake_detect(image: np.ndarray) -> list[FaceDetection]:
        raise FaceEngineError("synthetic detector error")

    engine.detect_faces = fake_detect  # type: ignore[method-assign]
    frame = np.zeros((40, 40, 3), dtype=np.uint8)

    with pytest.raises(FaceEngineError, match="synthetic"):
        engine.detect_faces_in_region(
            frame,
            (0, 0, 20, 20),
            upscale=2.0,
            score_threshold=0.60,
        )
    assert detector.threshold == pytest.approx(0.72)
    assert detector.changes == [0.60, 0.72]


class FakeRecognizer:
    def __init__(self) -> None:
        self.aligned_image: np.ndarray | None = None
        self.aligned_raw: np.ndarray | None = None

    def alignCrop(
        self,
        image: np.ndarray,
        raw: np.ndarray,
    ) -> np.ndarray:
        self.aligned_image = image
        self.aligned_raw = raw.copy()
        return np.ones((112, 112, 3), dtype=np.uint8)

    def feature(self, image: np.ndarray) -> np.ndarray:
        return np.ones((1, 128), dtype=np.float32)


def test_mapped_detection_aligns_from_original_frame() -> None:
    engine = FaceEngine.__new__(FaceEngine)
    recognizer = FakeRecognizer()
    engine._recognizer = recognizer
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    raw = np.asarray(
        [10, 12, 20, 24, 15, 18, 25, 18, 20, 24, 16, 30, 24, 30, 0.9],
        dtype=np.float32,
    )
    mapped = FaceDetection((10, 12, 30, 36), 0.9, raw)

    result = engine.extract_embedding(frame, mapped)

    assert recognizer.aligned_image is frame
    np.testing.assert_array_equal(recognizer.aligned_raw, raw)
    assert result.embedding.shape == (128,)
    assert np.linalg.norm(result.embedding) == pytest.approx(1.0)


def test_invalid_region_and_upscale_are_rejected() -> None:
    engine = FaceEngine.__new__(FaceEngine)
    called = False

    def fake_detect(image: np.ndarray) -> list[FaceDetection]:
        nonlocal called
        called = True
        return []

    engine.detect_faces = types.MethodType(  # type: ignore[method-assign]
        lambda self, image: fake_detect(image),
        engine,
    )
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    assert engine.detect_faces_in_region(
        frame,
        (30, 30, 40, 40),
    ) == []
    assert not called
    with pytest.raises(ValueError, match="upscale"):
        engine.detect_faces_in_region(frame, (0, 0, 10, 10), upscale=0.5)
    with pytest.raises(ValueError, match="max_dimension"):
        engine.detect_faces_in_region(
            frame,
            (0, 0, 10, 10),
            max_dimension=16,
        )
