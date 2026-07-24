from __future__ import annotations

import numpy as np
import pytest

from scripts.enroll_ai_team import Candidate, select_candidates
from src.face_engine import FaceEngine


def candidate(source: str, variant: str) -> Candidate:
    return Candidate(
        source_image=source,
        augmentation_name=variant,
        detection_score=0.9,
        embedding=np.ones(128, dtype=np.float32),
        is_original=variant == "original",
    )


def test_candidate_selection_balances_variants_across_sources() -> None:
    originals = [
        candidate("one.png", "original"),
        candidate("two.png", "original"),
        candidate("three.png", "original"),
    ]
    augmented = [
        candidate(source, variant)
        for source in ("one.png", "two.png", "three.png")
        for variant in ("cctv_jpeg_50", "horizontal_flip")
    ]

    selected = select_candidates(originals, augmented, 7)

    assert [
        (item.source_image, item.augmentation_name)
        for item in selected
    ] == [
        ("one.png", "original"),
        ("two.png", "original"),
        ("three.png", "original"),
        ("one.png", "cctv_jpeg_50"),
        ("two.png", "cctv_jpeg_50"),
        ("three.png", "cctv_jpeg_50"),
        ("one.png", "horizontal_flip"),
    ]


def test_candidate_selection_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError, match="positive"):
        select_candidates([], [], 0)


def test_embedding_variants_prioritize_cctv_compression() -> None:
    engine = FaceEngine.__new__(FaceEngine)
    observed: list[np.ndarray] = []

    def fake_embedding(image: np.ndarray) -> np.ndarray:
        observed.append(image.copy())
        result = np.zeros(128, dtype=np.float32)
        result[len(observed) - 1] = 1.0
        return result

    engine.embedding_from_aligned = fake_embedding  # type: ignore[method-assign]
    gradient = np.tile(
        np.arange(112, dtype=np.uint8),
        (112, 1),
    )
    aligned = np.dstack((gradient, gradient.T, gradient))

    variants = engine.embedding_variants(aligned, count=5)

    assert [name for name, _ in variants] == [
        "original",
        "cctv_jpeg_50",
        "horizontal_flip",
        "slightly_brighter",
        "slightly_darker",
    ]
    assert all(image.shape == aligned.shape for image in observed)
    assert all(image.dtype == np.uint8 for image in observed)
    assert not np.array_equal(observed[0], observed[1])
