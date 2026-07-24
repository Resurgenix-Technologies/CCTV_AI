"""In-memory cosine-similarity face recognition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.face_engine import EMBEDDING_DIMENSION
from src.repository import FaceRepository, StoredEmbedding


@dataclass(frozen=True)
class RecognitionResult:
    matched: bool
    person_id: str | None
    full_name: str
    similarity: float
    embedding_id: int | None
    rejection_reason: str | None = None

    @classmethod
    def unknown(
        cls,
        similarity: float = 0.0,
        reason: str = "UNKNOWN",
    ) -> "RecognitionResult":
        return cls(
            False,
            None,
            "UNKNOWN",
            similarity,
            None,
            reason,
        )


class FaceRecognizer:
    """Nearest-vector recognition over the SQLite enrolment set."""

    def __init__(
        self,
        repository: FaceRepository,
        threshold: float,
        minimum_margin: float = 0.0,
    ) -> None:
        if minimum_margin < 0.0:
            raise ValueError("minimum_margin cannot be negative.")
        self.repository = repository
        self.threshold = threshold
        self.minimum_margin = minimum_margin
        self._records: list[StoredEmbedding] = []
        self._matrix = np.empty(
            (0, EMBEDDING_DIMENSION),
            dtype=np.float32,
        )
        self.refresh()

    @property
    def embedding_count(self) -> int:
        """Number of face vectors currently cached in memory."""

        return len(self._records)

    @property
    def person_count(self) -> int:
        """Number of distinct global identities in the cache."""

        return len({
            record.person_id
            for record in self._records
        })

    def refresh(self) -> None:
        """Reload all enrolled vectors from SQLite."""

        self._records = (
            self.repository.load_all_embeddings()
        )

        if self._records:
            self._matrix = np.vstack([
                record.embedding
                for record in self._records
            ]).astype(np.float32)
        else:
            self._matrix = np.empty(
                (0, EMBEDDING_DIMENSION),
                dtype=np.float32,
            )

    def recognize(
        self,
        embedding: np.ndarray,
    ) -> RecognitionResult:
        vector = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        if vector.size != EMBEDDING_DIMENSION:
            raise ValueError(
                "Recognition embedding must contain "
                f"{EMBEDDING_DIMENSION} values."
            )

        if not np.all(np.isfinite(vector)):
            return RecognitionResult.unknown(
                reason="INVALID_EMBEDDING"
            )

        norm = float(np.linalg.norm(vector))

        if norm <= 1e-12:
            return RecognitionResult.unknown(
                reason="INVALID_EMBEDDING"
            )

        if self._matrix.shape[0] == 0:
            return RecognitionResult.unknown(
                reason="EMPTY_GALLERY"
            )

        normalized = vector / norm
        similarities = self._matrix @ normalized
        finite_scores = np.isfinite(similarities)
        if not np.any(finite_scores):
            return RecognitionResult.unknown(
                reason="INVALID_GALLERY"
            )
        safe_similarities = np.where(
            finite_scores,
            similarities,
            float("-inf"),
        )
        best_index = int(np.argmax(safe_similarities))
        similarity = float(safe_similarities[best_index])
        record = self._records[best_index]

        if similarity < self.threshold:
            return RecognitionResult.unknown(
                similarity,
                reason="BELOW_THRESHOLD",
            )

        if self.minimum_margin > 0.0:
            runner_up = max(
                (
                    float(score)
                    for score, candidate in zip(
                        safe_similarities,
                        self._records,
                        strict=True,
                    )
                    if candidate.person_id != record.person_id
                ),
                default=float("-inf"),
            )
            if similarity - runner_up < self.minimum_margin:
                return RecognitionResult.unknown(
                    similarity,
                    reason="AMBIGUOUS",
                )

        return RecognitionResult(
            matched=True,
            person_id=record.person_id,
            full_name=record.full_name,
            similarity=similarity,
            embedding_id=record.embedding_id,
        )
