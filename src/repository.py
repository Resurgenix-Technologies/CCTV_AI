"""SQLite repository for people and SFace embeddings."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.database import database_connection
from src.face_engine import EMBEDDING_DIMENSION


PERSON_TYPES = {"AI_TEAM", "VISITOR"}


class RepositoryError(RuntimeError):
    """Base repository exception."""


class EmptyEmbeddingsError(RepositoryError):
    """Raised when recognition is requested before enrolment."""


@dataclass(frozen=True)
class EmbeddingInput:
    embedding: np.ndarray
    model_name: str
    source_image: str
    augmentation_name: str
    detection_score: float


@dataclass(frozen=True)
class StoredEmbedding:
    embedding_id: int
    person_id: str
    full_name: str
    person_type: str
    embedding: np.ndarray
    source_image: str
    augmentation_name: str
    detection_score: float


@dataclass(frozen=True)
class EnrollmentResult:
    person_id: str
    full_name: str
    stored_embeddings: int


class FaceRepository:
    """Persist people and embeddings in SQLite."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.resolve()

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding must have {EMBEDDING_DIMENSION} values."
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("Embedding contains invalid values.")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("Embedding has zero length.")
        return np.ascontiguousarray(vector / norm, dtype=np.float32).tobytes()

    @staticmethod
    def deserialize_embedding(blob: bytes) -> np.ndarray:
        vector = np.frombuffer(blob, dtype=np.float32).copy()
        if vector.size != EMBEDDING_DIMENSION:
            raise RepositoryError(
                f"Stored embedding dimension is {vector.size}, expected 128."
            )
        return vector

    def replace_enrollment(
        self,
        *,
        full_name: str,
        person_type: str,
        embeddings: Iterable[EmbeddingInput],
        phone: str | None = None,
    ) -> EnrollmentResult:
        name = full_name.strip()
        kind = person_type.strip().upper()
        if not name:
            raise ValueError("full_name cannot be empty.")
        if kind not in PERSON_TYPES:
            raise ValueError(f"person_type must be one of {sorted(PERSON_TYPES)}.")

        prepared = list(embeddings)
        if not prepared:
            raise ValueError("At least one embedding is required.")

        try:
            with database_connection(self.database_path) as connection:
                row = connection.execute(
                    """
                    SELECT person_id
                    FROM people
                    WHERE full_name = ? AND person_type = ?;
                    """,
                    (name, kind),
                ).fetchone()

                if row:
                    person_id = str(row["person_id"])
                    connection.execute(
                        "DELETE FROM face_embeddings WHERE person_id = ?;",
                        (person_id,),
                    )
                    if phone:
                        connection.execute(
                            "UPDATE people SET phone = ? WHERE person_id = ?;",
                            (phone, person_id),
                        )
                else:
                    person_id = str(uuid.uuid4())
                    connection.execute(
                        """
                        INSERT INTO people (
                            person_id, full_name, person_type, phone
                        ) VALUES (?, ?, ?, ?);
                        """,
                        (person_id, name, kind, phone),
                    )

                for item in prepared:
                    connection.execute(
                        """
                        INSERT INTO face_embeddings (
                            person_id,
                            embedding,
                            embedding_dimension,
                            embedding_dtype,
                            model_name,
                            source_image,
                            augmentation_name,
                            detection_score
                        ) VALUES (?, ?, 128, 'float32', ?, ?, ?, ?);
                        """,
                        (
                            person_id,
                            sqlite3.Binary(
                                self.serialize_embedding(item.embedding)
                            ),
                            item.model_name,
                            item.source_image,
                            item.augmentation_name,
                            item.detection_score,
                        ),
                    )
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to save enrolment: {exc}") from exc

        return EnrollmentResult(person_id, name, len(prepared))

    def load_all_embeddings(self) -> list[StoredEmbedding]:
        try:
            with database_connection(self.database_path) as connection:
                rows = connection.execute(
                    """
                    SELECT
                        fe.embedding_id,
                        fe.person_id,
                        p.full_name,
                        p.person_type,
                        fe.embedding,
                        fe.source_image,
                        COALESCE(fe.augmentation_name, 'original')
                            AS augmentation_name,
                        fe.detection_score
                    FROM face_embeddings AS fe
                    JOIN people AS p ON p.person_id = fe.person_id
                    ORDER BY fe.embedding_id;
                    """
                ).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to load embeddings: {exc}") from exc

        return [
            StoredEmbedding(
                embedding_id=int(row["embedding_id"]),
                person_id=str(row["person_id"]),
                full_name=str(row["full_name"]),
                person_type=str(row["person_type"]),
                embedding=self.deserialize_embedding(row["embedding"]),
                source_image=str(row["source_image"] or ""),
                augmentation_name=str(row["augmentation_name"]),
                detection_score=float(row["detection_score"] or 0.0),
            )
            for row in rows
        ]

    def embedding_counts(self) -> list[tuple[str, str, int]]:
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT
                    p.full_name,
                    p.person_id,
                    COUNT(fe.embedding_id) AS count
                FROM people AS p
                LEFT JOIN face_embeddings AS fe
                    ON fe.person_id = p.person_id
                GROUP BY p.person_id, p.full_name
                ORDER BY p.full_name;
                """
            ).fetchall()
        return [
            (str(row["full_name"]), str(row["person_id"]), int(row["count"]))
            for row in rows
        ]
