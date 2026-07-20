"""Create the SQLite database schema for the MVP."""
from __future__ import annotations
import logging
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings
from src.database import DatabaseUnavailableError, database_connection

LOGGER = logging.getLogger("face_identity.init_db")
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS people (
    person_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    person_type TEXT NOT NULL CHECK (person_type IN ('AI_TEAM', 'VISITOR')),
    phone TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (full_name, person_type)
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dimension INTEGER NOT NULL DEFAULT 128 CHECK (embedding_dimension = 128),
    embedding_dtype TEXT NOT NULL DEFAULT 'float32' CHECK (embedding_dtype = 'float32'),
    model_name TEXT NOT NULL,
    source_image TEXT,
    detection_score REAL CHECK (
        detection_score IS NULL OR (detection_score >= 0.0 AND detection_score <= 1.0)
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES people(person_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS face_embeddings_person_id_idx
ON face_embeddings (person_id);
"""

def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def initialize_database(settings: Settings) -> None:
    with database_connection(settings.database_path) as connection:
        try:
            connection.executescript(SCHEMA_SQL)
            rows = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('people','face_embeddings') ORDER BY name;").fetchall()
        except sqlite3.Error as exc:
            raise RuntimeError(f"Database schema creation failed: {exc}") from exc
    names = {row["name"] for row in rows}
    if names != {"people", "face_embeddings"}:
        raise RuntimeError("One or more required SQLite tables were not created.")
    LOGGER.info("SQLite database initialized at %s", settings.database_path)
    LOGGER.info("Database schema initialized successfully.")

def main() -> int:
    try:
        settings = Settings.from_env()
        configure_logging(settings.log_level)
        initialize_database(settings)
    except (ValueError, DatabaseUnavailableError, RuntimeError) as exc:
        logging.basicConfig(level=logging.ERROR, format="%(asctime)s | %(levelname)s | %(message)s")
        LOGGER.error("%s", exc)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
