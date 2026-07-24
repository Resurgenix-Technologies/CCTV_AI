from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.activity_processor import (
    ActivityProcessor,
    ActivityProcessorError,
    PROXIMITY_ESTIMATION_METHOD,
)
from src.activity_repository import ActivityRepository, ActivityRepositoryError
from src.database import database_connection
from src.interaction_engine import (
    InteractionConfig,
    ProximityInteractionEngine,
    TrackObservation,
)


PEOPLE_SCHEMA = """
CREATE TABLE people (
    person_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    person_type TEXT NOT NULL
);
"""

BASE_TIME = datetime(2026, 7, 21, 9, 0, tzinfo=timezone.utc)


def prepare_repository(path: Path) -> ActivityRepository:
    with database_connection(path) as connection:
        connection.executescript(PEOPLE_SCHEMA)
        connection.executemany(
            """
            INSERT INTO people (person_id, full_name, person_type)
            VALUES (?, ?, ?);
            """,
            [
                ("person-a", "Person A", "VISITOR"),
                ("person-b", "Person B", "VISITOR"),
            ],
        )
    return ActivityRepository(path)


def captured_after(seconds: float) -> datetime:
    return BASE_TIME + timedelta(seconds=seconds)


def observation(
    person_id: str,
    seconds: float,
    center_x: float,
    *,
    session_id: str = "office-day-1",
    camera_id: str = "room-2",
) -> TrackObservation:
    return TrackObservation(
        session_id=session_id,
        camera_id=camera_id,
        person_id=person_id,
        timestamp=BASE_TIME.timestamp() + seconds,
        box=(center_x - 20.0, 0.0, center_x + 20.0, 100.0),
    )


def pair(seconds: float) -> list[TrackObservation]:
    return [
        observation("person-b", seconds, 70.0),
        observation("person-a", seconds, 0.0),
    ]


def interaction_engine(
    *,
    dwell: float = 2.0,
    grace: float = 3.0,
) -> ProximityInteractionEngine:
    return ProximityInteractionEngine(
        InteractionConfig(
            minimum_dwell_seconds=dwell,
            gap_grace_seconds=grace,
            enter_distance_threshold=1.0,
            exit_distance_threshold=1.25,
        )
    )


def test_two_camera_vertical_slice_uses_gate_visits_and_proximity_estimate(
    tmp_path: Path,
) -> None:
    repository = prepare_repository(tmp_path / "vertical-slice.db")
    processor = ActivityProcessor(repository, interaction_engine())

    processor.record_entry(
        "person-a",
        captured_after(0),
        "lobby-gate",
        "gate-entry-a",
    )
    processor.record_entry(
        "person-b",
        captured_after(12),
        "lobby-gate",
        "gate-entry-b",
    )

    # The same already-resolved identity appears in another camera namespace.
    # The processor consumes that identity; it does not implement body ReID.
    room_one_a = observation(
        "person-a",
        5,
        0.0,
        camera_id="room-1",
    )
    assert processor.process_observations([room_one_a]) == []

    for second in (14, 16, 18, 20, 22):
        processor.process_observations(pair(second))

    ended = processor.flush()
    assert [event.event_type for event in ended] == ["ENDED"]
    assert ended[0].pair == ("person-a", "person-b")
    assert ended[0].camera_id == "room-2"
    assert ended[0].duration_seconds == 8.0

    processor.record_exit(
        "person-b",
        captured_after(24),
        "lobby-gate",
    )
    processor.record_exit(
        "person-a",
        captured_after(30),
        "lobby-gate",
    )

    a_summary = repository.get_portal_summary(
        "person-a",
        as_of=captured_after(31),
    )
    b_summary = repository.get_portal_summary(
        "person-b",
        as_of=captured_after(31),
    )

    assert a_summary.total_presence_seconds == 30.0
    assert b_summary.total_presence_seconds == 12.0
    assert a_summary.total_interaction_seconds == 8.0
    assert b_summary.total_interaction_seconds == 8.0
    assert a_summary.interactions[0].counterpart_person_id == "person-b"

    with database_connection(repository.database_path) as connection:
        stored = connection.execute(
            """
            SELECT
                person_a_id,
                person_b_id,
                started_at,
                ended_at,
                camera_source,
                estimation_method
            FROM proximity_interactions;
            """
        ).fetchone()

    assert stored is not None
    assert (stored["person_a_id"], stored["person_b_id"]) == (
        "person-a",
        "person-b",
    )
    assert stored["started_at"] == "2026-07-21T09:00:14.000+00:00"
    assert stored["ended_at"] == "2026-07-21T09:00:22.000+00:00"
    assert stored["camera_source"] == "room-2"
    assert stored["estimation_method"] == PROXIMITY_ESTIMATION_METHOD


def test_visibility_never_opens_visit_and_gap_end_is_persisted(
    tmp_path: Path,
) -> None:
    repository = prepare_repository(tmp_path / "no-inferred-visits.db")
    processor = ActivityProcessor(
        repository,
        interaction_engine(dwell=1.0, grace=1.0),
    )
    processor.process_observations(pair(0))
    processor.process_observations(pair(1))

    ended = processor.process_frame(
        session_id="office-day-1",
        camera_id="room-2",
        timestamp=BASE_TIME.timestamp() + 3,
    )

    assert [event.event_type for event in ended] == ["ENDED"]
    with database_connection(repository.database_path) as connection:
        visit_count = connection.execute(
            "SELECT COUNT(*) AS count FROM observed_visits;"
        ).fetchone()
        interaction_count = connection.execute(
            "SELECT COUNT(*) AS count FROM proximity_interactions;"
        ).fetchone()

    assert visit_count is not None
    assert interaction_count is not None
    assert visit_count["count"] == 0
    assert interaction_count["count"] == 1


def test_gate_event_replay_is_idempotent_and_normalized_to_utc(
    tmp_path: Path,
) -> None:
    repository = prepare_repository(tmp_path / "gate-replay.db")
    processor = ActivityProcessor(repository)

    first = processor.record_entry(
        "person-a",
        "2026-07-21T14:30:00+05:30",
        "entry-gate",
        "reader-event-123",
    )
    replay = processor.record_entry(
        "person-a",
        "2026-07-21T09:00:00Z",
        "entry-gate",
        "reader-event-123",
    )
    assert first == replay

    processor.record_exit(
        "person-a",
        "2026-07-21T14:30:30+05:30",
        "exit-gate",
    )
    # Identical close replay reaches ActivityRepository's no-op path.
    processor.record_exit(
        "person-a",
        "2026-07-21T09:00:30Z",
        "exit-gate",
    )

    with database_connection(repository.database_path) as connection:
        rows = connection.execute(
            """
            SELECT visit_id, entered_at, exited_at
            FROM observed_visits;
            """
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["visit_id"] == first
    assert rows[0]["entered_at"] == "2026-07-21T09:00:00.000+00:00"
    assert rows[0]["exited_at"] == "2026-07-21T09:00:30.000+00:00"


def test_completed_interaction_id_is_deterministic_across_replay(
    tmp_path: Path,
) -> None:
    repository = prepare_repository(tmp_path / "interaction-replay.db")

    for _ in range(2):
        processor = ActivityProcessor(
            repository,
            interaction_engine(dwell=1.0, grace=2.0),
        )
        processor.process_observations(pair(0))
        processor.process_observations(pair(1))
        processor.flush()

    with database_connection(repository.database_path) as connection:
        rows = connection.execute(
            """
            SELECT interaction_id
            FROM proximity_interactions;
            """
        ).fetchall()

    assert len(rows) == 1
    assert str(rows[0]["interaction_id"]).startswith("proximity-v1-")


def test_database_failure_propagates_and_pending_end_can_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = prepare_repository(tmp_path / "failure-retry.db")
    processor = ActivityProcessor(
        repository,
        interaction_engine(dwell=1.0, grace=2.0),
    )
    processor.process_observations(pair(0))
    processor.process_observations(pair(1))

    original_record = repository.record_proximity_interaction

    def fail_write(*args: object, **kwargs: object) -> str:
        raise ActivityRepositoryError("database unavailable")

    monkeypatch.setattr(repository, "record_proximity_interaction", fail_write)
    with pytest.raises(ActivityRepositoryError, match="database unavailable"):
        processor.flush()
    assert processor.pending_interaction_count == 1

    monkeypatch.setattr(
        repository,
        "record_proximity_interaction",
        original_record,
    )
    assert processor.flush() == []
    assert processor.pending_interaction_count == 0

    with database_connection(repository.database_path) as connection:
        stored = connection.execute(
            "SELECT COUNT(*) AS count FROM proximity_interactions;"
        ).fetchone()
    assert stored is not None
    assert stored["count"] == 1


def test_gate_failures_and_missing_entry_are_not_hidden(tmp_path: Path) -> None:
    repository = prepare_repository(tmp_path / "gate-errors.db")
    processor = ActivityProcessor(repository)

    with pytest.raises(ActivityProcessorError, match="explicit entry"):
        processor.record_exit(
            "person-a",
            captured_after(1),
            "exit-gate",
        )

    with pytest.raises(ActivityRepositoryError):
        processor.record_entry(
            "unknown-person",
            captured_after(0),
            "entry-gate",
            "unknown-person-entry",
        )
