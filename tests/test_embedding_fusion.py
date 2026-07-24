import numpy as np
import pytest

from src.embedding_fusion import FaceEmbeddingFusion
from src.face_engine import EMBEDDING_DIMENSION


def vector(index: int, scale: float = 1.0) -> np.ndarray:
    embedding = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    embedding[index] = scale
    return embedding


def test_first_observation_is_normalized() -> None:
    fusion = FaceEmbeddingFusion()

    fused = fusion.update(
        7,
        vector(3, 8.0),
        detection_score=0.8,
        face_size=(32, 40),
    )

    np.testing.assert_allclose(fused, vector(3))
    assert fusion.sample_count(7) == 1


def test_quality_weight_uses_score_and_capped_squared_face_size() -> None:
    fusion = FaceEmbeddingFusion()

    fusion.update(
        1,
        vector(0),
        detection_score=1.0,
        face_size=(32, 32),
    )
    fused = fusion.update(
        1,
        vector(1),
        detection_score=0.5,
        face_size=(96, 96),
    )

    # First vector weight: 1 * 1. Second: .5 * capped size weight 4.
    expected = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    expected[0] = 1.0
    expected[1] = 2.0
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(fused, expected, atol=1e-7)


def test_window_drops_samples_older_than_five_observations() -> None:
    fusion = FaceEmbeddingFusion(max_samples=5)

    for index in range(6):
        fused = fusion.update(
            4,
            vector(index),
            detection_score=1.0,
            face_size=(32, 32),
        )

    expected = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    expected[1:6] = 1.0
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(fused, expected, atol=1e-7)
    assert fused[0] == pytest.approx(0.0)
    assert fusion.sample_count(4) == 5


def test_tracks_are_isolated_and_can_be_cleared_independently() -> None:
    fusion = FaceEmbeddingFusion()

    fusion.update(
        2,
        vector(2),
        detection_score=0.9,
        face_size=(40, 42),
    )
    fused = fusion.update(
        9,
        vector(9),
        detection_score=0.9,
        face_size=(40, 42),
    )

    np.testing.assert_allclose(fused, vector(9))
    assert fusion.track_count == 2

    fusion.clear(2)

    assert fusion.sample_count(2) == 0
    assert fusion.sample_count(9) == 1
    assert fusion.track_count == 1


def test_clear_many_and_clear_all_are_idempotent() -> None:
    fusion = FaceEmbeddingFusion()
    for track_id in (1, 2, 3):
        fusion.update(
            track_id,
            vector(track_id),
            detection_score=0.9,
            face_size=(32, 32),
        )

    fusion.clear_many([1, 3, 99])

    assert fusion.track_count == 1
    assert fusion.sample_count(2) == 1

    fusion.clear_all()
    fusion.clear_all()

    assert fusion.track_count == 0


@pytest.mark.parametrize(
    ("embedding", "score", "face_size"),
    [
        (np.zeros(EMBEDDING_DIMENSION, dtype=np.float32), 0.9, (32, 32)),
        (np.ones(12, dtype=np.float32), 0.9, (32, 32)),
        (np.full(EMBEDDING_DIMENSION, np.nan, dtype=np.float32), 0.9, (32, 32)),
        (vector(0), 0.0, (32, 32)),
        (vector(0), float("nan"), (32, 32)),
        (vector(0), 0.9, (0, 32)),
    ],
)
def test_invalid_observation_is_rejected_without_allocating_a_track(
    embedding: np.ndarray,
    score: float,
    face_size: tuple[int, int],
) -> None:
    fusion = FaceEmbeddingFusion()

    with pytest.raises(ValueError):
        fusion.update(
            5,
            embedding,
            detection_score=score,
            face_size=face_size,
        )

    assert fusion.track_count == 0


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("max_samples", 0),
        ("reference_face_size", 0),
        ("maximum_size_weight", 0.5),
        ("maximum_size_weight", float("inf")),
    ],
)
def test_invalid_configuration_is_rejected(
    keyword: str,
    value: int | float,
) -> None:
    with pytest.raises(ValueError):
        FaceEmbeddingFusion(**{keyword: value})
