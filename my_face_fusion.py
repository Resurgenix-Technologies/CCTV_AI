"""Bounded, track-level temporal fusion for live face embeddings."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from my_face_engine import EMBEDDING_DIMENSION


@dataclass
class _TrackBuffer:
    """The recent weighted vectors retained for one temporary track."""

    samples: deque = field(default_factory=deque)


class FaceEmbeddingFusion:
    """
    Fuse a bounded window of normalized face vectors for each local track.
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
        return len(self._tracks)

    def sample_count(self, track_id: int) -> int:
        track = self._tracks.get(track_id)
        return len(track.samples) if track is not None else 0

    def _normalized(self, embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size not in (128, 512):
            raise ValueError(
                "Face embedding vector must be either 128 or 512 dimensions, "
                f"got {vector.size} values."
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
        sharpness_score: float = 100.0,
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
        sharpness = max(1.0, float(sharpness_score))
        sharpness_weight = min(2.5, max(0.5, math.sqrt(sharpness / 100.0)))
        return score * size_weight * sharpness_weight

    def update(
        self,
        track_id: int,
        embedding: np.ndarray,
        *,
        detection_score: float,
        face_size: tuple[int, int],
        sharpness_score: float = 100.0,
        min_cosine_similarity: float = 0.35,
    ) -> np.ndarray:
        """Add one observation and return its normalized fused vector. Rejects outlier embeddings."""

        vector = self._normalized(embedding)
        dim = vector.size
        weight = self._weight(
            detection_score=detection_score,
            face_size=face_size,
            sharpness_score=sharpness_score,
        )
        track = self._tracks.get(track_id)
        if track is None:
            track = _TrackBuffer(samples=deque(maxlen=self.max_samples))
            self._tracks[track_id] = track

        # Outlier Rejection: Compare candidate against current running fused vector if buffer is non-empty
        if len(track.samples) > 0:
            current_fused = np.zeros(dim, dtype=np.float32)
            for sample, sample_w in track.samples:
                if sample.size == dim:
                    current_fused += sample * sample_w
            curr_norm = float(np.linalg.norm(current_fused))
            if curr_norm > 1e-12:
                curr_fused_norm = current_fused / curr_norm
                cos_sim = float(np.dot(vector, curr_fused_norm))
                # Reject candidate if it deviates strongly from established track representation
                if cos_sim < min_cosine_similarity:
                    return np.ascontiguousarray(curr_fused_norm, dtype=np.float32)

        track.samples.append((vector, weight))

        fused = np.zeros(dim, dtype=np.float32)
        for sample, sample_weight in track.samples:
            if sample.size == dim:
                fused += sample * sample_weight

        fused_norm = float(np.linalg.norm(fused))
        if not math.isfinite(fused_norm) or fused_norm <= 1e-12:
            raise ValueError("Fused face embedding has zero or invalid length.")
        return np.ascontiguousarray(fused / fused_norm, dtype=np.float32)

    def clear(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)

    def clear_many(self, track_ids: Iterable[int]) -> None:
        for track_id in track_ids:
            self.clear(track_id)

    def clear_all(self) -> None:
        self._tracks.clear()

    def solve(self) ->None:
        self