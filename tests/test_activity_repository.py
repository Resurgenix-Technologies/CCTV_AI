import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from src.activity_repository import (
    ActivityRepository,
    ActivityRepositoryError,
    InvalidPortalCredentialError,
    utc_iso_timestamp,
)
from src.database import database_connection


PEOPLE_SCHEMA = """
CREATE TABLE people (
    person_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    person_type TEXT NOT NULL
);

CREATE TABLE preexisting_audit_marker (
    marker TEXT PRIMARY KEY
);
"""


def prepare_database(path: Path) -> None:
    with database_connection(path) as connection:
        connection.executescript(PEOPLE_SCHEMA)
        connection.executemany(
            """
            INSERT INTO people (person_id, full_name, person_type)
            VALUES (?, ?, ?);
            """,
            [
                ("person-a", "Person A", "VISITOR"),
                ("person-b", "Person B", "AI_TEAM"),
                ("person-c", "Person C", "AI_TEAM"),
            ],
        )
        connection.execute(
            "INSERT INTO preexisting_audit_marker (marker) VALUES ('keep');"
        )


def test_schema_is_idempotent_and_preserves_existing_data(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "activity.db"
    prepare_database(database_path)

    repository = ActivityRepository(database_path)
    repository.ensure_schema()

    with database_connection(database_path) as connection:
        tables = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table';"
            ).fetchall()
        }
        marker = connection.execute(
            "SELECT marker FROM preexisting_audit_marker;"
        ).fetchone()
        foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(proximity_interactions);"
        ).fetchall()

    assert {
        "person_activity_profiles",
        "observed_visits",
        "proximity_interactions",
        "portal_credentials",
    }.issubset(tables)
    assert marker is not None
    assert marker["marker"] == "keep"
    assert {row["table"] for row in foreign_keys} == {"people"}


def test_portal_summary_aggregates_visits_interactions_and_photo(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "activity.db"
    prepare_database(database_path)
    repository = ActivityRepository(database_path)

    repository.set_profile_photo(
        "person-a",
        "profiles/person-a.jpg",
        captured_at="2026-07-21T08:55:00Z",
        updated_at="2026-07-21T08:56:00Z",
    )
    visit_id = repository.start_visit(
        "person-a",
        "2026-07-21T09:00:00+00:00",
        entry_camera_source="lobby-entry",
        visit_id="visit-a-1",
    )
    repository.end_visit(
        visit_id,
        "2026-07-21T12:00:00+00:00",
        exit_camera_source="lobby-exit",
    )
    # Reversed arguments exercise canonical pair storage.
    repository.record_proximity_interaction(
        "person-b",
        "person-a",
        "2026-07-21T09:30:00Z",
        "2026-07-21T09:45:00Z",
        confidence=0.91,
        camera_source="room-1-camera",
        zone_label="meeting-table",
        interaction_id="interaction-ab-1",
    )
    repository.record_proximity_interaction(
        "person-a",
        "person-b",
        "2026-07-21T10:00:00Z",
        "2026-07-21T10:05:00Z",
        confidence=0.82,
        interaction_id="interaction-ab-2",
    )
    repository.record_proximity_interaction(
        "person-c",
        "person-a",
        "2026-07-21T11:00:00Z",
        "2026-07-21T11:08:00Z",
        interaction_id="interaction-ac-1",
    )

    summary = repository.get_portal_summary(
        "person-a",
        as_of="2026-07-21T13:00:00Z",
    )
    timeline = repository.get_portal_timeline(
        "person-a",
        as_of="2026-07-21T13:00:00Z",
    )

    assert summary.full_name == "Person A"
    assert summary.profile_photo is not None
    assert summary.profile_photo.path == "profiles/person-a.jpg"
    assert summary.profile_photo.captured_at == "2026-07-21T08:55:00.000+00:00"
    assert summary.total_presence_seconds == 3 * 60 * 60
    assert summary.total_interaction_seconds == 28 * 60
    assert summary.first_entry_at == "2026-07-21T09:00:00.000+00:00"
    assert summary.latest_exit_at == "2026-07-21T12:00:00.000+00:00"
    assert summary.visits[0].entry_camera_source == "lobby-entry"

    by_counterpart = {
        item.counterpart_person_id: item
        for item in summary.interactions
    }
    assert by_counterpart["person-b"].interaction_count == 2
    assert by_counterpart["person-b"].total_duration_seconds == 20 * 60
    assert by_counterpart["person-c"].total_duration_seconds == 8 * 60
    assert timeline[0].activity_id == "interaction-ac-1"
    assert timeline[0].counterpart_full_name == "Person C"

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        stored_pair = connection.execute(
            """
            SELECT person_a_id, person_b_id
            FROM proximity_interactions
            WHERE interaction_id = 'interaction-ab-1';
            """
        ).fetchone()
    finally:
        connection.close()

    assert stored_pair is not None
    assert (stored_pair["person_a_id"], stored_pair["person_b_id"]) == (
        "person-a",
        "person-b",
    )


def test_open_visit_is_measured_only_until_query_time(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "activity.db"
    prepare_database(database_path)
    repository = ActivityRepository(database_path)
    repository.start_visit(
        "person-a",
        "2026-07-21T09:00:00Z",
        visit_id="open-visit",
    )

    summary = repository.get_portal_summary(
        "person-a",
        as_of="2026-07-21T09:20:00Z",
    )

    assert summary.total_presence_seconds == 20 * 60
    assert summary.visits[0].exited_at is None
    assert summary.latest_exit_at is None


def test_visit_ids_are_replay_safe_and_only_one_visit_can_be_open(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "activity.db"
    prepare_database(database_path)
    repository = ActivityRepository(database_path)

    first = repository.start_visit(
        "person-a",
        "2026-07-21T09:00:00Z",
        entry_camera_source="entry",
        visit_id="source-event-visit-1",
    )
    replay = repository.start_visit(
        "person-a",
        "2026-07-21T09:00:00+00:00",
        entry_camera_source="entry",
        visit_id="source-event-visit-1",
    )

    assert first == replay == "source-event-visit-1"
    with pytest.raises(ActivityRepositoryError, match="Conflicting replay"):
        repository.start_visit(
            "person-a",
            "2026-07-21T09:01:00Z",
            entry_camera_source="entry",
            visit_id="source-event-visit-1",
        )
    with pytest.raises(ActivityRepositoryError):
        repository.start_visit(
            "person-a",
            "2026-07-21T10:00:00Z",
            visit_id="source-event-visit-2",
        )

    repository.end_visit(
        first,
        "2026-07-21T11:00:00Z",
        exit_camera_source="exit",
    )
    # A replay of the start event remains valid after a later close event.
    assert repository.start_visit(
        "person-a",
        "2026-07-21T09:00:00Z",
        entry_camera_source="entry",
        visit_id="source-event-visit-1",
    ) == first
    assert repository.start_visit(
        "person-a",
        "2026-07-21T12:00:00Z",
        visit_id="source-event-visit-2",
    ) == "source-event-visit-2"


def test_interaction_ids_are_replay_safe(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "activity.db"
    prepare_database(database_path)
    repository = ActivityRepository(database_path)

    first = repository.record_proximity_interaction(
        "person-b",
        "person-a",
        "2026-07-21T09:00:00Z",
        "2026-07-21T09:10:00Z",
        confidence=0.9,
        camera_source="room-1",
        zone_label="desk",
        interaction_id="source-event-interaction-1",
    )
    replay = repository.record_proximity_interaction(
        "person-a",
        "person-b",
        "2026-07-21T09:00:00+00:00",
        "2026-07-21T09:10:00+00:00",
        confidence=0.9,
        camera_source="room-1",
        zone_label="desk",
        interaction_id="source-event-interaction-1",
    )

    assert first == replay == "source-event-interaction-1"
    with pytest.raises(ActivityRepositoryError, match="Conflicting replay"):
        repository.record_proximity_interaction(
            "person-a",
            "person-b",
            "2026-07-21T09:00:00Z",
            "2026-07-21T09:11:00Z",
            confidence=0.9,
            camera_source="room-1",
            zone_label="desk",
            interaction_id="source-event-interaction-1",
        )

    with database_connection(database_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) AS count FROM proximity_interactions;"
        ).fetchone()
    assert count is not None
    assert count["count"] == 1


def test_portal_tokens_are_hashed_expiring_and_revocable(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "activity.db"
    prepare_database(database_path)
    repository = ActivityRepository(database_path)
    issued = repository.issue_portal_credential(
        "person-a",
        expires_at="2026-07-21T14:00:00Z",
        issued_at="2026-07-21T08:00:00Z",
    )

    with database_connection(database_path) as connection:
        stored = connection.execute(
            """
            SELECT token_hash, expires_at, revoked_at
            FROM portal_credentials
            WHERE credential_id = ?;
            """,
            (issued.credential_id,),
        ).fetchone()

    assert stored is not None
    assert stored["token_hash"] != issued.token
    assert stored["token_hash"] == hashlib.sha256(
        issued.token.encode("utf-8")
    ).hexdigest()
    assert len(stored["token_hash"]) == 64
    assert stored["revoked_at"] is None
    assert repository.resolve_portal_credential(
        issued.token,
        at="2026-07-21T09:00:00Z",
    ) == "person-a"
    assert repository.get_portal_summary_for_token(
        issued.token,
        as_of="2026-07-21T09:00:00Z",
    ).person_id == "person-a"

    with pytest.raises(InvalidPortalCredentialError):
        repository.resolve_portal_credential(
            issued.token,
            at="2026-07-21T14:00:00Z",
        )

    assert repository.revoke_portal_credential(
        issued.credential_id,
        revoked_at="2026-07-21T10:00:00Z",
    )
    # Repeated revocation remains idempotent.
    assert repository.revoke_portal_credential(
        issued.credential_id,
        revoked_at="2026-07-21T11:00:00Z",
    )
    with pytest.raises(InvalidPortalCredentialError):
        repository.resolve_portal_credential(
            issued.token,
            at="2026-07-21T09:00:00Z",
        )


def test_validation_rejects_ambiguous_time_and_invalid_interaction(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "activity.db"
    prepare_database(database_path)
    repository = ActivityRepository(database_path)

    with pytest.raises(ValueError, match="UTC offset"):
        utc_iso_timestamp(datetime(2026, 7, 21, 9, 0))
    assert utc_iso_timestamp("2026-07-21T14:30:00+05:30") == (
        "2026-07-21T09:00:00.000+00:00"
    )
    with pytest.raises(ValueError, match="different people"):
        repository.record_proximity_interaction(
            "person-a",
            "person-a",
            "2026-07-21T09:00:00Z",
            "2026-07-21T09:01:00Z",
        )
    with pytest.raises(ValueError, match="later than"):
        repository.record_proximity_interaction(
            "person-a",
            "person-b",
            "2026-07-21T09:01:00Z",
            "2026-07-21T09:00:00Z",
        )
    with pytest.raises(ActivityRepositoryError):
        repository.start_visit(
            "missing-person",
            "2026-07-21T09:00:00Z",
        )
