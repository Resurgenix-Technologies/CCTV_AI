import sqlite3
from pathlib import Path

from src.event_repository import (
    EventRepository,
    RecognitionEvent,
)


def create_legacy_database(path: Path) -> None:
    connection = sqlite3.connect(path)

    try:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE people (
                person_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                person_type TEXT NOT NULL,
                phone TEXT,
                created_at TEXT NOT NULL
                    DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE recognition_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                camera_source TEXT NOT NULL,
                local_track_id INTEGER NOT NULL,
                person_id TEXT,
                full_name TEXT,
                similarity REAL,
                detection_confidence REAL,
                occurred_at TEXT NOT NULL
                    DEFAULT CURRENT_TIMESTAMP
            );

            INSERT INTO recognition_events (
                event_type,
                camera_source,
                local_track_id
            )
            VALUES (
                'TRACK_STARTED',
                'Webcam 0',
                1
            );
            """
        )
        connection.commit()
    finally:
        connection.close()


def test_legacy_table_is_migrated_without_data_loss(
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy.db"
    create_legacy_database(database)

    repository = EventRepository(database)

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row

    try:
        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(recognition_events);"
            ).fetchall()
        }

        legacy = connection.execute(
            """
            SELECT session_id, frame_index
            FROM recognition_events
            WHERE event_id = 1;
            """
        ).fetchone()

    finally:
        connection.close()

    assert "session_id" in columns
    assert "frame_index" in columns
    assert legacy["session_id"] == "legacy"
    assert legacy["frame_index"] is None

    new_id = repository.log_event(
        RecognitionEvent(
            session_id="new-session",
            event_type="TRACK_STARTED",
            camera_source="Webcam 0",
            local_track_id=2,
            frame_index=10,
        )
    )

    assert new_id == 2
