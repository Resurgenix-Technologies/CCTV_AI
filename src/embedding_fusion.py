"""Bounded, track-level temporal fusion for live face embeddings."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from src.face_engine import EMBEDDING_DIMENSION


@dataclass
class _TrackBuffer:
    """The recent weighted vectors retained for one temporary track."""

    samples: deque[tuple[np.ndarray, float]] = field(default_factory=deque)


class FaceEmbeddingFusion:
    """
    Fuse a bounded window of normalized face vectors for each local track.

    Larger source-frame faces carry more evidence than distant faces. Their
    squared size factor is capped so one close-up cannot completely suppress
    the other recent observations. Detection confidence supplies the second
    part of the weight.

    The buffer deliberately stores no cumulative history: expired tracks can
    be cleared, and even an active track is represented by at most
    ``max_samples`` vectors.
    """

    def __init__(
        self,
        *,
        max_samples: int = 5,
        reference_face_size: int = 32,
        maximum_size_weight: float = 4.0,
    ) -> None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive.")
        if reference_face_size < 1:
            raise ValueError("reference_face_size must be positive.")
        if not math.isfinite(maximum_size_weight) or maximum_size_weight < 1.0:
            raise ValueError("maximum_size_weight must be finite and at least 1.")

        self.max_samples = max_samples
        self.reference_face_size = reference_face_size
        self.maximum_size_weight = maximum_size_weight
        self._tracks: dict[int, _TrackBuffer] = {}

    @property
    def track_count(self) -> int:
        """Number of tracks currently holding temporal face evidence."""

        return len(self._tracks)

    def sample_count(self, track_id: int) -> int:
        """Return the number of retained vectors for ``track_id``."""

        track = self._tracks.get(track_id)
        return len(track.samples) if track is not None else 0

    def _normalized(self, embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size != EMBEDDING_DIMENSION:
            raise ValueError(
                "Face embedding must contain "
                f"{EMBEDDING_DIMENSION} values."
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("Face embedding contains non-finite values.")

        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("Face embedding has zero length.")
        return np.ascontiguousarray(vector / norm, dtype=np.float32)

    def _weight(
        self,
        *,
        detection_score: float,
        face_size: tuple[int, int],
    ) -> float:
        score = float(detection_score)
        if not math.isfinite(score) or score <= 0.0:
            raise ValueError("detection_score must be finite and positive.")

        width, height = face_size
        minimum_dimension = min(int(width), int(height))
        if minimum_dimension < 1:
            raise ValueError("face_size dimensions must be positive.")

        size_ratio = minimum_dimension / self.reference_face_size
        size_weight = min(
            self.maximum_size_weight,
            max(1.0, size_ratio * size_ratio),
        )
        return score * size_weight

    def update(
        self,
        track_id: int,
        embedding: np.ndarray,
        *,
        detection_score: float,
        face_size: tuple[int, int],
    ) -> np.ndarray:
        """Add one READY observation and return its normalized fused vector."""

        vector = self._normalized(embedding)
        weight = self._weight(
            detection_score=detection_score,
            face_size=face_size,
        )
        track = self._tracks.get(track_id)
        if track is None:
            track = _TrackBuffer(samples=deque(maxlen=self.max_samples))
            self._tracks[track_id] = track
        track.samples.append((vector, weight))

        fused = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        for sample, sample_weight in track.samples:
            fused += sample * sample_weight

        fused_norm = float(np.linalg.norm(fused))
        if not math.isfinite(fused_norm) or fused_norm <= 1e-12:
            raise ValueError("Fused face embedding has zero or invalid length.")
        return np.ascontiguousarray(fused / fused_norm, dtype=np.float32)

    def clear(self, track_id: int) -> None:
        """Discard all temporal evidence for one expired track."""

        self._tracks.pop(track_id, None)

    def clear_many(self, track_ids: Iterable[int]) -> None:
        """Discard temporal evidence for several expired tracks."""

        for track_id in track_ids:
            self.clear(track_id)

    def clear_all(self) -> None:
        """Discard every buffered face vector at stream shutdown."""

        self._tracks.clear()
