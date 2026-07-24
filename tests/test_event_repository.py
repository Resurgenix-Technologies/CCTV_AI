import sqlite3
from pathlib import Path

from src.database import database_connection
from src.event_repository import EventRepository, RecognitionEvent


PEOPLE_SCHEMA = """
CREATE TABLE people (
    person_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    person_type TEXT NOT NULL
);
"""


def prepare_database(path: Path) -> None:
    with database_connection(path) as connection:
        connection.executescript(PEOPLE_SCHEMA)
        connection.execute(
            """
            INSERT INTO people (
                person_id,
                full_name,
                person_type
            ) VALUES (?, ?, ?);
            """,
            ("person-a", "A", "AI_TEAM"),
        )


def test_event_repository_creates_table_and_stores_event(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "events.db"
    prepare_database(database_path)
    repository = EventRepository(database_path)

    event_id = repository.log_event(
        RecognitionEvent(
            session_id="session-1",
            event_type="IDENTITY_CONFIRMED",
            camera_source="Webcam 0",
            local_track_id=7,
            person_id="person-a",
            full_name="A",
            similarity=0.72,
            detection_confidence=0.91,
            frame_index=30,
        )
    )

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            """
            SELECT *
            FROM recognition_events
            WHERE event_id = ?;
            """,
            (event_id,),
        ).fetchone()
    finally:
        connection.close()

    assert row is not None
    assert row["event_type"] == "IDENTITY_CONFIRMED"
    assert row["person_id"] == "person-a"
    assert row["frame_index"] == 30


def test_event_repository_rejects_invalid_confidence(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "events.db"
    prepare_database(database_path)
    repository = EventRepository(database_path)

    try:
        repository.log_event(
            RecognitionEvent(
                session_id="session-1",
                event_type="TRACK_STARTED",
                camera_source="Webcam 0",
                local_track_id=1,
                detection_confidence=1.5,
            )
        )
    except ValueError as exc:
        assert "detection_confidence" in str(exc)
    else:
        raise AssertionError("Expected ValueError.")
