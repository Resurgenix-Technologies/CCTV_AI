"""SQLite storage and portal read models for observed person activity.

This module is deliberately independent of the live recognition pipeline.  It
adds projections that a future visit/interaction processor can populate and a
web layer can read after exchanging an opaque QR credential for its owner.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from src.database import database_connection


Timestamp = datetime | str
TimelineEventType = Literal["VISIT", "PROXIMITY_INTERACTION"]


ACTIVITY_SCHEMA = """
CREATE TABLE IF NOT EXISTS person_activity_profiles (
    person_id TEXT PRIMARY KEY,
    profile_photo_path TEXT,
    profile_photo_captured_at TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (person_id)
        REFERENCES people(person_id)
        ON DELETE CASCADE,
    CHECK (
        profile_photo_path IS NOT NULL
        OR profile_photo_captured_at IS NULL
    )
);

CREATE TABLE IF NOT EXISTS observed_visits (
    visit_id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL,
    entered_at TEXT NOT NULL,
    exited_at TEXT,
    entry_camera_source TEXT,
    exit_camera_source TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (person_id)
        REFERENCES people(person_id)
        ON DELETE CASCADE,
    CHECK (
        exited_at IS NULL
        OR exited_at >= entered_at
    )
);

CREATE INDEX IF NOT EXISTS observed_visits_person_time_idx
ON observed_visits (person_id, entered_at);

CREATE UNIQUE INDEX IF NOT EXISTS observed_visits_one_open_person_idx
ON observed_visits (person_id)
WHERE exited_at IS NULL;

CREATE TABLE IF NOT EXISTS proximity_interactions (
    interaction_id TEXT PRIMARY KEY,
    person_a_id TEXT NOT NULL,
    person_b_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    confidence REAL,
    camera_source TEXT,
    zone_label TEXT,
    estimation_method TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (person_a_id)
        REFERENCES people(person_id)
        ON DELETE CASCADE,
    FOREIGN KEY (person_b_id)
        REFERENCES people(person_id)
        ON DELETE CASCADE,
    CHECK (person_a_id < person_b_id),
    CHECK (ended_at > started_at),
    CHECK (
        confidence IS NULL
        OR confidence BETWEEN 0.0 AND 1.0
    )
);

CREATE INDEX IF NOT EXISTS proximity_interactions_person_a_time_idx
ON proximity_interactions (person_a_id, started_at);

CREATE INDEX IF NOT EXISTS proximity_interactions_person_b_time_idx
ON proximity_interactions (person_b_id, started_at);

CREATE TABLE IF NOT EXISTS portal_credentials (
    credential_id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    issued_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked_at TEXT,
    FOREIGN KEY (person_id)
        REFERENCES people(person_id)
        ON DELETE CASCADE,
    CHECK (length(token_hash) = 64),
    CHECK (expires_at > issued_at),
    CHECK (
        revoked_at IS NULL
        OR revoked_at >= issued_at
    )
);

CREATE INDEX IF NOT EXISTS portal_credentials_person_idx
ON portal_credentials (person_id);
"""


_REQUIRED_COLUMNS = {
    "person_activity_profiles": {
        "person_id",
        "profile_photo_path",
        "profile_photo_captured_at",
        "updated_at",
    },
    "observed_visits": {
        "visit_id",
        "person_id",
        "entered_at",
        "exited_at",
        "entry_camera_source",
        "exit_camera_source",
        "created_at",
        "updated_at",
    },
    "proximity_interactions": {
        "interaction_id",
        "person_a_id",
        "person_b_id",
        "started_at",
        "ended_at",
        "confidence",
        "camera_source",
        "zone_label",
        "estimation_method",
        "created_at",
    },
    "portal_credentials": {
        "credential_id",
        "person_id",
        "token_hash",
        "issued_at",
        "expires_at",
        "revoked_at",
    },
}


class ActivityRepositoryError(RuntimeError):
    """Raised when the activity store cannot complete an operation."""


class PersonNotFoundError(ActivityRepositoryError):
    """Raised when a portal query references an unknown person."""


class InvalidPortalCredentialError(ActivityRepositoryError):
    """Raised for an unknown, expired, or revoked portal credential."""


@dataclass(frozen=True)
class IssuedPortalCredential:
    """A newly issued credential; ``token`` is returned only at creation."""

    credential_id: str
    person_id: str
    token: str
    issued_at: str
    expires_at: str


@dataclass(frozen=True)
class PortalProfilePhoto:
    """Metadata for a photo stored outside SQLite."""

    path: str
    captured_at: str | None
    updated_at: str


@dataclass(frozen=True)
class PortalVisit:
    visit_id: str
    entered_at: str
    exited_at: str | None
    duration_seconds: float
    entry_camera_source: str | None
    exit_camera_source: str | None


@dataclass(frozen=True)
class PortalInteractionTotal:
    counterpart_person_id: str
    counterpart_full_name: str
    total_duration_seconds: float
    interaction_count: int
    last_interaction_at: str


@dataclass(frozen=True)
class PortalSummary:
    person_id: str
    full_name: str
    profile_photo: PortalProfilePhoto | None
    total_presence_seconds: float
    total_interaction_seconds: float
    first_entry_at: str | None
    latest_exit_at: str | None
    visits: tuple[PortalVisit, ...]
    interactions: tuple[PortalInteractionTotal, ...]


@dataclass(frozen=True)
class PortalTimelineEvent:
    event_type: TimelineEventType
    activity_id: str
    started_at: str
    ended_at: str | None
    duration_seconds: float
    counterpart_person_id: str | None = None
    counterpart_full_name: str | None = None
    camera_source: str | None = None
    zone_label: str | None = None
    confidence: float | None = None


def utc_iso_timestamp(value: Timestamp | None = None) -> str:
    """Normalize an aware timestamp to fixed-width UTC ISO-8601."""

    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("timestamp cannot be empty.")
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid ISO-8601 timestamp: {value!r}."
            ) from exc
    else:
        raise TypeError("timestamp must be a datetime or ISO-8601 string.")

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a UTC offset.")

    return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def _parse_stored_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _duration_seconds(start: str, end: str) -> float:
    delta = _parse_stored_timestamp(end) - _parse_stored_timestamp(start)
    return round(delta.total_seconds(), 3)


def _required_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty.")
    return normalized


def _optional_text(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty when provided.")
    return normalized


def _canonical_pair(person_one_id: str, person_two_id: str) -> tuple[str, str]:
    first = _required_text(person_one_id, "person_one_id")
    second = _required_text(person_two_id, "person_two_id")
    if first == second:
        raise ValueError("An interaction requires two different people.")
    return (first, second) if first < second else (second, first)


def _token_digest(token: str) -> str:
    normalized = _required_text(token, "token")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class ActivityRepository:
    """Persist visits/interactions and expose portal-safe read models."""

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
        """Add activity tables and indexes without replacing existing data."""

        try:
            with database_connection(self.database_path) as connection:
                people_columns = {
                    str(row["name"])
                    for row in connection.execute(
                        "PRAGMA table_info(people);"
                    ).fetchall()
                }
                if not {"person_id", "full_name"}.issubset(people_columns):
                    raise ActivityRepositoryError(
                        "The existing people table must be initialized first."
                    )

                connection.executescript(ACTIVITY_SCHEMA)

                for table_name, expected in _REQUIRED_COLUMNS.items():
                    actual = {
                        str(row["name"])
                        for row in connection.execute(
                            f"PRAGMA table_info({table_name});"
                        ).fetchall()
                    }
                    missing = expected - actual
                    if missing:
                        raise ActivityRepositoryError(
                            f"{table_name} is missing columns: "
                            + ", ".join(sorted(missing))
                        )
        except ActivityRepositoryError:
            raise
        except sqlite3.Error as exc:
            raise ActivityRepositoryError(
                f"Failed to initialize activity storage: {exc}"
            ) from exc

    def set_profile_photo(
        self,
        person_id: str,
        profile_photo_path: str | None,
        *,
        captured_at: Timestamp | None = None,
        updated_at: Timestamp | None = None,
    ) -> None:
        """Set external photo metadata, or clear it when path is ``None``."""

        owner = _required_text(person_id, "person_id")
        changed_at = utc_iso_timestamp(updated_at)

        try:
            with database_connection(self.database_path) as connection:
                if profile_photo_path is None:
                    connection.execute(
                        "DELETE FROM person_activity_profiles "
                        "WHERE person_id = ?;",
                        (owner,),
                    )
                    return

                path = _required_text(
                    profile_photo_path,
                    "profile_photo_path",
                )
                captured = (
                    utc_iso_timestamp(captured_at)
                    if captured_at is not None
                    else None
                )
                connection.execute(
                    """
                    INSERT INTO person_activity_profiles (
                        person_id,
                        profile_photo_path,
                        profile_photo_captured_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(person_id) DO UPDATE SET
                        profile_photo_path = excluded.profile_photo_path,
                        profile_photo_captured_at =
                            excluded.profile_photo_captured_at,
                        updated_at = excluded.updated_at;
                    """,
                    (owner, path, captured, changed_at),
                )
        except sqlite3.IntegrityError as exc:
            raise PersonNotFoundError(
                f"Unknown person_id: {owner}."
            ) from exc
        except sqlite3.Error as exc:
            raise ActivityRepositoryError(
                f"Failed to store profile photo metadata: {exc}"
            ) from exc

    def start_visit(
        self,
        person_id: str,
        entered_at: Timestamp,
        *,
        entry_camera_source: str | None = None,
        visit_id: str | None = None,
    ) -> str:
        """Create an open observed visit and return its ID."""

        owner = _required_text(person_id, "person_id")
        caller_supplied_id = visit_id is not None
        identifier = (
            _required_text(visit_id, "visit_id")
            if caller_supplied_id
            else str(uuid.uuid4())
        )
        entered = utc_iso_timestamp(entered_at)
        camera = _optional_text(
            entry_camera_source,
            "entry_camera_source",
        )
        created = utc_iso_timestamp()

        try:
            with database_connection(self.database_path) as connection:
                if caller_supplied_id:
                    existing = connection.execute(
                        """
                        SELECT person_id, entered_at, entry_camera_source
                        FROM observed_visits
                        WHERE visit_id = ?;
                        """,
                        (identifier,),
                    ).fetchone()
                    if existing is not None:
                        matches = (
                            str(existing["person_id"]) == owner
                            and str(existing["entered_at"]) == entered
                            and existing["entry_camera_source"] == camera
                        )
                        if matches:
                            return identifier
                        raise ActivityRepositoryError(
                            f"Conflicting replay for visit {identifier}."
                        )

                connection.execute(
                    """
                    INSERT INTO observed_visits (
                        visit_id,
                        person_id,
                        entered_at,
                        entry_camera_source,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (identifier, owner, entered, camera, created, created),
                )
        except ActivityRepositoryError:
            raise
        except sqlite3.IntegrityError as exc:
            raise ActivityRepositoryError(
                f"Could not start visit {identifier}: {exc}"
            ) from exc
        except sqlite3.Error as exc:
            raise ActivityRepositoryError(
                f"Failed to start visit: {exc}"
            ) from exc

        return identifier

    def end_visit(
        self,
        visit_id: str,
        exited_at: Timestamp,
        *,
        exit_camera_source: str | None = None,
    ) -> None:
        """Close an open visit; an identical replay is a no-op."""

        identifier = _required_text(visit_id, "visit_id")
        exited = utc_iso_timestamp(exited_at)
        camera = _optional_text(
            exit_camera_source,
            "exit_camera_source",
        )

        try:
            with database_connection(self.database_path) as connection:
                row = connection.execute(
                    """
                    SELECT entered_at, exited_at, exit_camera_source
                    FROM observed_visits
                    WHERE visit_id = ?;
                    """,
                    (identifier,),
                ).fetchone()
                if row is None:
                    raise ActivityRepositoryError(
                        f"Unknown visit_id: {identifier}."
                    )
                if exited < str(row["entered_at"]):
                    raise ValueError("exited_at must not precede entered_at.")
                if row["exited_at"] is not None:
                    if (
                        str(row["exited_at"]) == exited
                        and row["exit_camera_source"] == camera
                    ):
                        return
                    raise ActivityRepositoryError(
                        f"Visit {identifier} is already closed."
                    )

                connection.execute(
                    """
                    UPDATE observed_visits
                    SET exited_at = ?,
                        exit_camera_source = ?,
                        updated_at = ?
                    WHERE visit_id = ?;
                    """,
                    (exited, camera, utc_iso_timestamp(), identifier),
                )
        except (ActivityRepositoryError, ValueError):
            raise
        except sqlite3.Error as exc:
            raise ActivityRepositoryError(
                f"Failed to end visit: {exc}"
            ) from exc

    def record_proximity_interaction(
        self,
        person_one_id: str,
        person_two_id: str,
        started_at: Timestamp,
        ended_at: Timestamp,
        *,
        confidence: float | None = None,
        camera_source: str | None = None,
        zone_label: str | None = None,
        estimation_method: str = "CALIBRATED_PROXIMITY",
        interaction_id: str | None = None,
    ) -> str:
        """Store one completed estimated proximity episode."""

        person_a, person_b = _canonical_pair(
            person_one_id,
            person_two_id,
        )
        caller_supplied_id = interaction_id is not None
        identifier = (
            _required_text(interaction_id, "interaction_id")
            if caller_supplied_id
            else str(uuid.uuid4())
        )
        started = utc_iso_timestamp(started_at)
        ended = utc_iso_timestamp(ended_at)
        if ended <= started:
            raise ValueError("ended_at must be later than started_at.")
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")

        camera = _optional_text(camera_source, "camera_source")
        zone = _optional_text(zone_label, "zone_label")
        method = _required_text(estimation_method, "estimation_method")

        try:
            with database_connection(self.database_path) as connection:
                if caller_supplied_id:
                    existing = connection.execute(
                        """
                        SELECT
                            person_a_id,
                            person_b_id,
                            started_at,
                            ended_at,
                            confidence,
                            camera_source,
                            zone_label,
                            estimation_method
                        FROM proximity_interactions
                        WHERE interaction_id = ?;
                        """,
                        (identifier,),
                    ).fetchone()
                    if existing is not None:
                        stored_confidence = (
                            float(existing["confidence"])
                            if existing["confidence"] is not None
                            else None
                        )
                        matches = (
                            str(existing["person_a_id"]) == person_a
                            and str(existing["person_b_id"]) == person_b
                            and str(existing["started_at"]) == started
                            and str(existing["ended_at"]) == ended
                            and stored_confidence == confidence
                            and existing["camera_source"] == camera
                            and existing["zone_label"] == zone
                            and str(existing["estimation_method"]) == method
                        )
                        if matches:
                            return identifier
                        raise ActivityRepositoryError(
                            "Conflicting replay for interaction "
                            f"{identifier}."
                        )

                connection.execute(
                    """
                    INSERT INTO proximity_interactions (
                        interaction_id,
                        person_a_id,
                        person_b_id,
                        started_at,
                        ended_at,
                        confidence,
                        camera_source,
                        zone_label,
                        estimation_method,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        identifier,
                        person_a,
                        person_b,
                        started,
                        ended,
                        confidence,
                        camera,
                        zone,
                        method,
                        utc_iso_timestamp(),
                    ),
                )
        except ActivityRepositoryError:
            raise
        except sqlite3.IntegrityError as exc:
            raise ActivityRepositoryError(
                f"Could not store interaction {identifier}: {exc}"
            ) from exc
        except sqlite3.Error as exc:
            raise ActivityRepositoryError(
                f"Failed to store proximity interaction: {exc}"
            ) from exc

        return identifier

    def issue_portal_credential(
        self,
        person_id: str,
        expires_at: Timestamp,
        *,
        issued_at: Timestamp | None = None,
    ) -> IssuedPortalCredential:
        """Issue a random bearer token and persist only its SHA-256 digest."""

        owner = _required_text(person_id, "person_id")
        issued = utc_iso_timestamp(issued_at)
        expires = utc_iso_timestamp(expires_at)
        if expires <= issued:
            raise ValueError("expires_at must be later than issued_at.")

        identifier = str(uuid.uuid4())
        token = secrets.token_urlsafe(32)
        digest = _token_digest(token)

        try:
            with database_connection(self.database_path) as connection:
                connection.execute(
                    """
                    INSERT INTO portal_credentials (
                        credential_id,
                        person_id,
                        token_hash,
                        issued_at,
                        expires_at
                    ) VALUES (?, ?, ?, ?, ?);
                    """,
                    (identifier, owner, digest, issued, expires),
                )
        except sqlite3.IntegrityError as exc:
            raise PersonNotFoundError(
                f"Unknown person_id: {owner}."
            ) from exc
        except sqlite3.Error as exc:
            raise ActivityRepositoryError(
                f"Failed to issue portal credential: {exc}"
            ) from exc

        return IssuedPortalCredential(
            credential_id=identifier,
            person_id=owner,
            token=token,
            issued_at=issued,
            expires_at=expires,
        )

    def resolve_portal_credential(
        self,
        token: str,
        *,
        at: Timestamp | None = None,
    ) -> str:
        """Resolve an active opaque credential to its owner for web login."""

        digest = _token_digest(token)
        checked_at = utc_iso_timestamp(at)

        try:
            with database_connection(self.database_path) as connection:
                row = connection.execute(
                    """
                    SELECT person_id
                    FROM portal_credentials
                    WHERE token_hash = ?
                      AND revoked_at IS NULL
                      AND issued_at <= ?
                      AND expires_at > ?;
                    """,
                    (digest, checked_at, checked_at),
                ).fetchone()
        except sqlite3.Error as exc:
            raise ActivityRepositoryError(
                f"Failed to resolve portal credential: {exc}"
            ) from exc

        if row is None:
            raise InvalidPortalCredentialError(
                "Portal credential is invalid, expired, or revoked."
            )
        return str(row["person_id"])

    def revoke_portal_credential(
        self,
        credential_id: str,
        *,
        revoked_at: Timestamp | None = None,
    ) -> bool:
        """Revoke a credential. Return ``False`` if its ID is unknown."""

        identifier = _required_text(credential_id, "credential_id")
        revoked = utc_iso_timestamp(revoked_at)

        try:
            with database_connection(self.database_path) as connection:
                row = connection.execute(
                    """
                    SELECT issued_at, revoked_at
                    FROM portal_credentials
                    WHERE credential_id = ?;
                    """,
                    (identifier,),
                ).fetchone()
                if row is None:
                    return False
                if revoked < str(row["issued_at"]):
                    raise ValueError(
                        "revoked_at must not precede issued_at."
                    )
                if row["revoked_at"] is None:
                    connection.execute(
                        """
                        UPDATE portal_credentials
                        SET revoked_at = ?
                        WHERE credential_id = ?;
                        """,
                        (revoked, identifier),
                    )
        except ValueError:
            raise
        except sqlite3.Error as exc:
            raise ActivityRepositoryError(
                f"Failed to revoke portal credential: {exc}"
            ) from exc

        return True

    def get_portal_summary(
        self,
        person_id: str,
        *,
        as_of: Timestamp | None = None,
    ) -> PortalSummary:
        """Build the portal summary for a session-authenticated person."""

        owner = _required_text(person_id, "person_id")
        cutoff = utc_iso_timestamp(as_of)

        try:
            with database_connection(self.database_path) as connection:
                identity = connection.execute(
                    """
                    SELECT
                        p.person_id,
                        p.full_name,
                        ap.profile_photo_path,
                        ap.profile_photo_captured_at,
                        ap.updated_at AS profile_photo_updated_at
                    FROM people AS p
                    LEFT JOIN person_activity_profiles AS ap
                        ON ap.person_id = p.person_id
                    WHERE p.person_id = ?;
                    """,
                    (owner,),
                ).fetchone()
                if identity is None:
                    raise PersonNotFoundError(
                        f"Unknown person_id: {owner}."
                    )

                visit_rows = connection.execute(
                    """
                    SELECT
                        visit_id,
                        entered_at,
                        exited_at,
                        entry_camera_source,
                        exit_camera_source
                    FROM observed_visits
                    WHERE person_id = ?
                      AND entered_at <= ?
                    ORDER BY entered_at DESC, visit_id DESC;
                    """,
                    (owner, cutoff),
                ).fetchall()

                interaction_rows = connection.execute(
                    """
                    SELECT
                        i.interaction_id,
                        i.started_at,
                        i.ended_at,
                        i.confidence,
                        i.camera_source,
                        i.zone_label,
                        CASE
                            WHEN i.person_a_id = ? THEN i.person_b_id
                            ELSE i.person_a_id
                        END AS counterpart_person_id,
                        CASE
                            WHEN i.person_a_id = ? THEN person_b.full_name
                            ELSE person_a.full_name
                        END AS counterpart_full_name
                    FROM proximity_interactions AS i
                    JOIN people AS person_a
                        ON person_a.person_id = i.person_a_id
                    JOIN people AS person_b
                        ON person_b.person_id = i.person_b_id
                    WHERE (i.person_a_id = ? OR i.person_b_id = ?)
                      AND i.started_at <= ?
                    ORDER BY i.started_at DESC, i.interaction_id DESC;
                    """,
                    (owner, owner, owner, owner, cutoff),
                ).fetchall()
        except (ActivityRepositoryError, PersonNotFoundError):
            raise
        except sqlite3.Error as exc:
            raise ActivityRepositoryError(
                f"Failed to build portal summary: {exc}"
            ) from exc

        visits: list[PortalVisit] = []
        presence_total = 0.0
        for row in visit_rows:
            stored_exit = (
                str(row["exited_at"])
                if row["exited_at"] is not None
                else None
            )
            duration_end = (
                min(stored_exit, cutoff)
                if stored_exit is not None
                else cutoff
            )
            duration = _duration_seconds(
                str(row["entered_at"]),
                duration_end,
            )
            presence_total += duration
            visits.append(
                PortalVisit(
                    visit_id=str(row["visit_id"]),
                    entered_at=str(row["entered_at"]),
                    exited_at=stored_exit,
                    duration_seconds=duration,
                    entry_camera_source=(
                        str(row["entry_camera_source"])
                        if row["entry_camera_source"] is not None
                        else None
                    ),
                    exit_camera_source=(
                        str(row["exit_camera_source"])
                        if row["exit_camera_source"] is not None
                        else None
                    ),
                )
            )

        totals: dict[str, dict[str, object]] = {}
        interaction_total = 0.0
        for row in interaction_rows:
            duration_end = min(str(row["ended_at"]), cutoff)
            duration = _duration_seconds(
                str(row["started_at"]),
                duration_end,
            )
            interaction_total += duration
            counterpart = str(row["counterpart_person_id"])
            current = totals.get(counterpart)
            if current is None:
                totals[counterpart] = {
                    "full_name": str(row["counterpart_full_name"]),
                    "duration": duration,
                    "count": 1,
                    "last": str(row["started_at"]),
                }
            else:
                current["duration"] = float(current["duration"]) + duration
                current["count"] = int(current["count"]) + 1

        interaction_totals = tuple(
            sorted(
                (
                    PortalInteractionTotal(
                        counterpart_person_id=counterpart,
                        counterpart_full_name=str(values["full_name"]),
                        total_duration_seconds=round(
                            float(values["duration"]),
                            3,
                        ),
                        interaction_count=int(values["count"]),
                        last_interaction_at=str(values["last"]),
                    )
                    for counterpart, values in totals.items()
                ),
                key=lambda item: (
                    -item.total_duration_seconds,
                    item.counterpart_full_name,
                    item.counterpart_person_id,
                ),
            )
        )

        photo = None
        if identity["profile_photo_path"] is not None:
            photo = PortalProfilePhoto(
                path=str(identity["profile_photo_path"]),
                captured_at=(
                    str(identity["profile_photo_captured_at"])
                    if identity["profile_photo_captured_at"] is not None
                    else None
                ),
                updated_at=str(identity["profile_photo_updated_at"]),
            )

        completed_exits = [
            visit.exited_at
            for visit in visits
            if visit.exited_at is not None and visit.exited_at <= cutoff
        ]
        return PortalSummary(
            person_id=str(identity["person_id"]),
            full_name=str(identity["full_name"]),
            profile_photo=photo,
            total_presence_seconds=round(presence_total, 3),
            total_interaction_seconds=round(interaction_total, 3),
            first_entry_at=(
                min(visit.entered_at for visit in visits)
                if visits
                else None
            ),
            latest_exit_at=max(completed_exits) if completed_exits else None,
            visits=tuple(visits),
            interactions=interaction_totals,
        )

    def get_portal_summary_for_token(
        self,
        token: str,
        *,
        as_of: Timestamp | None = None,
    ) -> PortalSummary:
        """Credential-authenticated convenience wrapper for simple portals."""

        owner = self.resolve_portal_credential(token, at=as_of)
        return self.get_portal_summary(owner, as_of=as_of)

    def get_portal_timeline(
        self,
        person_id: str,
        *,
        as_of: Timestamp | None = None,
    ) -> tuple[PortalTimelineEvent, ...]:
        """Return visits and proximity episodes newest-first."""

        owner = _required_text(person_id, "person_id")
        cutoff = utc_iso_timestamp(as_of)
        # Validate ownership and reuse exactly the same visit duration semantics.
        summary = self.get_portal_summary(owner, as_of=cutoff)

        timeline: list[PortalTimelineEvent] = [
            PortalTimelineEvent(
                event_type="VISIT",
                activity_id=visit.visit_id,
                started_at=visit.entered_at,
                ended_at=visit.exited_at,
                duration_seconds=visit.duration_seconds,
                camera_source=visit.entry_camera_source,
            )
            for visit in summary.visits
        ]

        try:
            with database_connection(self.database_path) as connection:
                rows = connection.execute(
                    """
                    SELECT
                        i.interaction_id,
                        i.started_at,
                        i.ended_at,
                        i.confidence,
                        i.camera_source,
                        i.zone_label,
                        CASE
                            WHEN i.person_a_id = ? THEN i.person_b_id
                            ELSE i.person_a_id
                        END AS counterpart_person_id,
                        CASE
                            WHEN i.person_a_id = ? THEN person_b.full_name
                            ELSE person_a.full_name
                        END AS counterpart_full_name
                    FROM proximity_interactions AS i
                    JOIN people AS person_a
                        ON person_a.person_id = i.person_a_id
                    JOIN people AS person_b
                        ON person_b.person_id = i.person_b_id
                    WHERE (i.person_a_id = ? OR i.person_b_id = ?)
                      AND i.started_at <= ?;
                    """,
                    (owner, owner, owner, owner, cutoff),
                ).fetchall()
        except sqlite3.Error as exc:
            raise ActivityRepositoryError(
                f"Failed to build portal timeline: {exc}"
            ) from exc

        for row in rows:
            ended = str(row["ended_at"])
            duration_end = min(ended, cutoff)
            timeline.append(
                PortalTimelineEvent(
                    event_type="PROXIMITY_INTERACTION",
                    activity_id=str(row["interaction_id"]),
                    started_at=str(row["started_at"]),
                    ended_at=ended,
                    duration_seconds=_duration_seconds(
                        str(row["started_at"]),
                        duration_end,
                    ),
                    counterpart_person_id=str(
                        row["counterpart_person_id"]
                    ),
                    counterpart_full_name=str(
                        row["counterpart_full_name"]
                    ),
                    camera_source=(
                        str(row["camera_source"])
                        if row["camera_source"] is not None
                        else None
                    ),
                    zone_label=(
                        str(row["zone_label"])
                        if row["zone_label"] is not None
                        else None
                    ),
                    confidence=(
                        float(row["confidence"])
                        if row["confidence"] is not None
                        else None
                    ),
                )
            )

        timeline.sort(
            key=lambda item: (item.started_at, item.activity_id),
            reverse=True,
        )
        return tuple(timeline)

    def get_portal_timeline_for_token(
        self,
        token: str,
        *,
        as_of: Timestamp | None = None,
    ) -> tuple[PortalTimelineEvent, ...]:
        """Credential-authenticated timeline convenience wrapper."""

        owner = self.resolve_portal_credential(token, at=as_of)
        return self.get_portal_timeline(owner, as_of=as_of)
