"""Initialize and migrate the SQLite schema."""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402
from src.activity_repository import ActivityRepository  # noqa: E402
from src.database import database_connection  # noqa: E402
from src.event_repository import (  # noqa: E402
    migrate_recognition_events_schema,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    person_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    person_type TEXT NOT NULL
        CHECK (person_type IN ('AI_TEAM', 'VISITOR')),
    phone TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (full_name, person_type)
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dimension INTEGER NOT NULL DEFAULT 128
        CHECK (embedding_dimension = 128),
    embedding_dtype TEXT NOT NULL DEFAULT 'float32'
        CHECK (embedding_dtype = 'float32'),
    model_name TEXT NOT NULL,
    source_image TEXT,
    augmentation_name TEXT DEFAULT 'original',
    detection_score REAL
        CHECK (
            detection_score IS NULL
            OR detection_score BETWEEN 0.0 AND 1.0
        ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id)
        REFERENCES people(person_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS face_embeddings_person_id_idx
ON face_embeddings (person_id);
"""


def initialize(settings: Settings) -> None:
    """Create base tables and apply non-destructive migrations."""

    with database_connection(settings.database_path) as connection:
        connection.executescript(SCHEMA)

        embedding_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(face_embeddings);"
            ).fetchall()
        }

        if "augmentation_name" not in embedding_columns:
            connection.execute(
                """
                ALTER TABLE face_embeddings
                ADD COLUMN augmentation_name TEXT
                DEFAULT 'original';
                """
            )

        migrate_recognition_events_schema(connection)

        required_tables = {
            "people",
            "face_embeddings",
            "recognition_events",
        }

        tables = {
            str(row["name"])
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name IN (
                      'people',
                      'face_embeddings',
                      'recognition_events'
                  );
                """
            ).fetchall()
        }

        if tables != required_tables:
            raise RuntimeError(
                "Required tables were not created."
            )

    # ActivityRepository owns its additive schema and validates every required
    # column. Run it only after the base people table transaction is committed.
    ActivityRepository(settings.database_path)


def main() -> int:
    settings = Settings.from_env()

    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(message)s"
        ),
    )

    try:
        initialize(settings)
    except (
        sqlite3.Error,
        RuntimeError,
        ValueError,
    ) as exc:
        logging.error("%s", exc)
        return 1

    logging.info(
        "SQLite initialized and migrated: %s",
        settings.database_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
