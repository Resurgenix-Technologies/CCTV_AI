"""SQLite storage for significant recognition events."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from src.database import database_connection


EventType = Literal[
    "TRACK_STARTED",
    "IDENTITY_CONFIRMED",
    "TRACK_ENDED",
]

VALID_EVENT_TYPES = {
    "TRACK_STARTED",
    "IDENTITY_CONFIRMED",
    "TRACK_ENDED",
}

# Keep table creation and index creation separate. SQLite's
# CREATE TABLE IF NOT EXISTS does not add new columns to an existing table.
RECOGNITION_EVENTS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS recognition_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL
        CHECK (
            event_type IN (
                'TRACK_STARTED',
                'IDENTITY_CONFIRMED',
                'TRACK_ENDED'
            )
        ),
    camera_source TEXT NOT NULL,
    local_track_id INTEGER NOT NULL,
    person_id TEXT,
    full_name TEXT,
    similarity REAL,
    detection_confidence REAL,
    frame_index INTEGER,
    occurred_at TEXT NOT NULL,
    FOREIGN KEY (person_id)
        REFERENCES people(person_id)
        ON DELETE SET NULL,
    CHECK (
        similarity IS NULL
        OR similarity BETWEEN -1.0 AND 1.0
    ),
    CHECK (
        detection_confidence IS NULL
        OR detection_confidence BETWEEN 0.0 AND 1.0
    ),
    CHECK (
        frame_index IS NULL
        OR frame_index >= 0
    )
);
"""

RECOGNITION_EVENTS_INDEX_SCHEMA = """
CREATE INDEX IF NOT EXISTS recognition_events_session_track_idx
ON recognition_events (
    session_id,
    camera_source,
    local_track_id
);

CREATE INDEX IF NOT EXISTS recognition_events_person_idx
ON recognition_events (person_id);

CREATE INDEX IF NOT EXISTS recognition_events_time_idx
ON recognition_events (occurred_at);
"""

# Kept for compatibility with any older imports.
RECOGNITION_EVENTS_SCHEMA = (
    RECOGNITION_EVENTS_TABLE_SCHEMA
    + RECOGNITION_EVENTS_INDEX_SCHEMA
)


class EventRepositoryError(RuntimeError):
    """Raised when a recognition event cannot be stored."""


def utc_timestamp() -> str:
    """Return an unambiguous UTC ISO-8601 timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def migrate_recognition_events_schema(
    connection: sqlite3.Connection,
) -> None:
    """
    Create or upgrade the recognition_events table without deleting rows.

    Existing rows receive session_id='legacy'. New application runs use a
    generated UUID session ID.
    """

    connection.executescript(RECOGNITION_EVENTS_TABLE_SCHEMA)

    columns = {
        str(row["name"])
        for row in connection.execute(
            "PRAGMA table_info(recognition_events);"
        ).fetchall()
    }

    if "session_id" not in columns:
        connection.execute(
            """
            ALTER TABLE recognition_events
            ADD COLUMN session_id TEXT
            NOT NULL DEFAULT 'legacy';
            """
        )

    if "frame_index" not in columns:
        connection.execute(
            """
            ALTER TABLE recognition_events
            ADD COLUMN frame_index INTEGER;
            """
        )

    # Re-read after migration and fail clearly if the table is unexpected.
    columns = {
        str(row["name"])
        for row in connection.execute(
            "PRAGMA table_info(recognition_events);"
        ).fetchall()
    }

    required_columns = {
        "event_id",
        "session_id",
        "event_type",
        "camera_source",
        "local_track_id",
        "person_id",
        "full_name",
        "similarity",
        "detection_confidence",
        "frame_index",
        "occurred_at",
    }

    missing = required_columns - columns

    if missing:
        raise EventRepositoryError(
            "recognition_events is missing columns after migration: "
            + ", ".join(sorted(missing))
        )

    connection.executescript(RECOGNITION_EVENTS_INDEX_SCHEMA)


@dataclass(frozen=True)
class RecognitionEvent:
    """One significant tracking or identity event."""

    session_id: str
    event_type: EventType
    camera_source: str
    local_track_id: int
    person_id: str | None = None
    full_name: str | None = None
    similarity: float | None = None
    detection_confidence: float | None = None
    frame_index: int | None = None
    occurred_at: str = field(default_factory=utc_timestamp)


class EventRepository:
    """Persist significant CCTV events in SQLite."""

    def __init__(
        self,
        database_path: Path,
        *,
        initialize_schema: bool = True,
    ) -> None:
        self.database_path = database_path.resolve()

        if initialize_schema:
            self.ensure_schema()

    def ensure_schema(self) -> None:
        """Create or migrate the event table and its indexes."""

        try:
            with database_connection(self.database_path) as connection:
                migrate_recognition_events_schema(connection)
        except (sqlite3.Error, EventRepositoryError) as exc:
            raise EventRepositoryError(
                f"Failed to initialize recognition events: {exc}"
            ) from exc

    @staticmethod
    def _validate(event: RecognitionEvent) -> None:
        if not event.session_id.strip():
            raise ValueError("session_id cannot be empty.")

        if event.event_type not in VALID_EVENT_TYPES:
            raise ValueError(
                f"Unsupported event_type: {event.event_type!r}."
            )

        if not event.camera_source.strip():
            raise ValueError("camera_source cannot be empty.")

        if event.local_track_id < 0:
            raise ValueError("local_track_id cannot be negative.")

        if event.frame_index is not None and event.frame_index < 0:
            raise ValueError("frame_index cannot be negative.")

        if (
            event.similarity is not None
            and not -1.0 <= event.similarity <= 1.0
        ):
            raise ValueError(
                "similarity must be between -1 and 1."
            )

        if (
            event.detection_confidence is not None
            and not 0.0 <= event.detection_confidence <= 1.0
        ):
            raise ValueError(
                "detection_confidence must be between 0 and 1."
            )

    def log_event(self, event: RecognitionEvent) -> int:
        """Insert one event and return its database ID."""

        self._validate(event)

        try:
            with database_connection(self.database_path) as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO recognition_events (
                        session_id,
                        event_type,
                        camera_source,
                        local_track_id,
                        person_id,
                        full_name,
                        similarity,
                        detection_confidence,
                        frame_index,
                        occurred_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        event.session_id,
                        event.event_type,
                        event.camera_source,
                        event.local_track_id,
                        event.person_id,
                        event.full_name,
                        event.similarity,
                        event.detection_confidence,
                        event.frame_index,
                        event.occurred_at,
                    ),
                )
                event_id = cursor.lastrowid
        except sqlite3.Error as exc:
            raise EventRepositoryError(
                f"Failed to store recognition event: {exc}"
            ) from exc

        if event_id is None:
            raise EventRepositoryError(
                "SQLite did not return an event ID."
            )

        return int(event_id)
