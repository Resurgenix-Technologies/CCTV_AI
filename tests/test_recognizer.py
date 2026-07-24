from __future__ import annotations

import math

import numpy as np
import pytest

from src.recognizer import FaceRecognizer
from src.repository import StoredEmbedding


def vector(cosine_to_first_axis: float) -> np.ndarray:
    result = np.zeros(128, dtype=np.float32)
    result[0] = cosine_to_first_axis
    result[1] = math.sqrt(1.0 - cosine_to_first_axis**2)
    return result


def record(
    embedding_id: int,
    person_id: str,
    name: str,
    embedding: np.ndarray,
) -> StoredEmbedding:
    return StoredEmbedding(
        embedding_id=embedding_id,
        person_id=person_id,
        full_name=name,
        person_type="AI_TEAM",
        embedding=embedding,
        source_image=f"{person_id}.jpg",
        augmentation_name="original",
        detection_score=0.9,
    )


class FakeRepository:
    def __init__(self, records: list[StoredEmbedding]) -> None:
        self.records = records

    def load_all_embeddings(self) -> list[StoredEmbedding]:
        return self.records


def test_ambiguity_margin_rejects_close_second_person() -> None:
    repository = FakeRepository(
        [
            record(1, "a", "Person A", vector(0.80)),
            record(2, "b", "Person B", vector(0.77)),
        ]
    )
    recognizer = FaceRecognizer(  # type: ignore[arg-type]
        repository,
        threshold=0.70,
        minimum_margin=0.05,
    )
    query = np.zeros(128, dtype=np.float32)
    query[0] = 1.0

    result = recognizer.recognize(query)

    assert not result.matched
    assert result.full_name == "UNKNOWN"
    assert result.similarity == pytest.approx(0.80)
    assert result.rejection_reason == "AMBIGUOUS"


def test_margin_uses_next_distinct_person_not_same_person_embedding() -> None:
    repository = FakeRepository(
        [
            record(1, "a", "Person A", vector(0.82)),
            record(2, "a", "Person A", vector(0.81)),
            record(3, "b", "Person B", vector(0.60)),
        ]
    )
    recognizer = FaceRecognizer(  # type: ignore[arg-type]
        repository,
        threshold=0.70,
        minimum_margin=0.10,
    )
    query = np.zeros(128, dtype=np.float32)
    query[0] = 1.0

    result = recognizer.recognize(query)

    assert result.matched
    assert result.person_id == "a"
    assert result.similarity == pytest.approx(0.82)
    assert result.rejection_reason is None


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_non_finite_query_can_never_match(invalid_value: float) -> None:
    repository = FakeRepository(
        [record(1, "a", "Person A", vector(0.90))]
    )
    recognizer = FaceRecognizer(  # type: ignore[arg-type]
        repository,
        threshold=0.70,
    )
    query = np.zeros(128, dtype=np.float32)
    query[0] = invalid_value

    result = recognizer.recognize(query)

    assert not result.matched
    assert result.rejection_reason == "INVALID_EMBEDDING"


def test_non_finite_gallery_record_is_ignored() -> None:
    invalid = np.full(128, np.nan, dtype=np.float32)
    repository = FakeRepository(
        [
            record(1, "bad", "Bad Record", invalid),
            record(2, "a", "Person A", vector(0.90)),
        ]
    )
    recognizer = FaceRecognizer(  # type: ignore[arg-type]
        repository,
        threshold=0.70,
    )
    query = np.zeros(128, dtype=np.float32)
    query[0] = 1.0

    result = recognizer.recognize(query)

    assert result.matched
    assert result.person_id == "a"
