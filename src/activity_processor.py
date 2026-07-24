"""Bridge captured activity events to durable portal projections.

The processor has two deliberately separate inputs:

* instrumented gate events explicitly open and close whole-site visits; and
* already-identified camera observations feed a proximity interaction estimate.

Mere camera visibility never creates a visit.  This module also does not
perform body re-identification: ``person_id`` must already be stable when an
observation arrives.  Sustained proximity is an interaction estimate, not
proof that a conversation occurred.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from src.activity_repository import ActivityRepository, Timestamp, utc_iso_timestamp
from src.interaction_engine import (
    InteractionEvent,
    ProximityInteractionEngine,
    TrackObservation,
)


PROXIMITY_ESTIMATION_METHOD = "NORMALIZED_FOOTPOINT_PROXIMITY_ESTIMATE_V1"


class ActivityProcessorError(RuntimeError):
    """Raised when captured events cannot form a valid processing sequence."""


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _stable_id(prefix: str, components: list[str]) -> str:
    payload = json.dumps(
        [prefix, *components],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()}"


def _epoch_capture_time(timestamp: float) -> str:
    """Convert an engine's Unix capture timestamp to UTC, never wall time."""

    try:
        captured = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (OSError, OverflowError, ValueError) as exc:
        raise ValueError(
            f"timestamp is outside the supported Unix capture range: {timestamp}"
        ) from exc
    return utc_iso_timestamp(captured)


@dataclass(frozen=True, slots=True)
class _PendingInteraction:
    interaction_id: str
    person_a_id: str
    person_b_id: str
    started_at: str
    ended_at: str
    camera_source: str


class ActivityProcessor:
    """Coordinate gate visits, proximity estimates, and activity storage.

    Observation timestamps are Unix capture timestamps in seconds.  They are
    converted to aware UTC values from the event itself when an ended estimate
    is stored; processing time is never substituted for capture time.
    """

    def __init__(
        self,
        repository: ActivityRepository,
        interaction_engine: ProximityInteractionEngine | None = None,
    ) -> None:
        self.repository = repository
        self.interaction_engine = (
            interaction_engine or ProximityInteractionEngine()
        )
        self._latest_visit_by_person: dict[str, str] = {}
        self._pending_interactions: dict[str, _PendingInteraction] = {}

    @property
    def pending_interaction_count(self) -> int:
        """Number of ended estimates waiting for a successful DB write."""

        return len(self._pending_interactions)

    def record_entry(
        self,
        person_id: str,
        captured_at: Timestamp,
        camera_source: str,
        event_id: str,
    ) -> str:
        """Open a visit from an explicit, instrumented gate-entry event.

        The source event ID deterministically supplies the visit idempotency
        key.  An identical replay is therefore a no-op in ``ActivityRepository``.
        """

        owner = _required_text(person_id, "person_id")
        source_event_id = _required_text(event_id, "event_id")
        captured = utc_iso_timestamp(captured_at)
        visit_id = _stable_id("gate-visit-v1", [source_event_id])

        # Do not catch repository failures: callers must see and handle them.
        stored_id = self.repository.start_visit(
            owner,
            captured,
            entry_camera_source=camera_source,
            visit_id=visit_id,
        )
        self._latest_visit_by_person[owner] = stored_id
        return stored_id

    def record_exit(
        self,
        person_id: str,
        captured_at: Timestamp,
        camera_source: str,
    ) -> str:
        """Close the visit opened by this processor for a gate-exit event."""

        owner = _required_text(person_id, "person_id")
        visit_id = self._latest_visit_by_person.get(owner)
        if visit_id is None:
            raise ActivityProcessorError(
                "Cannot record an exit without an explicit entry handled by "
                f"this processor for person_id {owner!r}."
            )

        captured = utc_iso_timestamp(captured_at)
        # Retain the mapping after success so an identical exit replay reaches
        # ActivityRepository's idempotent end_visit path.
        self.repository.end_visit(
            visit_id,
            captured,
            exit_camera_source=camera_source,
        )
        return visit_id

    def process_observations(
        self,
        observations: Iterable[TrackObservation],
    ) -> list[InteractionEvent]:
        """Run captured observations and persist any newly ended estimates."""

        self._drain_pending_interactions()
        events = self.interaction_engine.process(observations)
        self._enqueue_ended_interactions(events)
        self._drain_pending_interactions()
        return events

    def process_frame(
        self,
        *,
        session_id: str,
        camera_id: str,
        timestamp: float,
        observations: Iterable[TrackObservation] = (),
    ) -> list[InteractionEvent]:
        """Process one camera frame, including an explicit empty frame."""

        self._drain_pending_interactions()
        events = self.interaction_engine.process_frame(
            session_id=session_id,
            camera_id=camera_id,
            timestamp=timestamp,
            observations=observations,
        )
        self._enqueue_ended_interactions(events)
        self._drain_pending_interactions()
        return events

    def flush(
        self,
        *,
        session_id: str | None = None,
        camera_id: str | None = None,
    ) -> list[InteractionEvent]:
        """Flush engine state and persist every resulting ended estimate."""

        self._drain_pending_interactions()
        events = self.interaction_engine.flush(
            session_id=session_id,
            camera_id=camera_id,
        )
        self._enqueue_ended_interactions(events)
        self._drain_pending_interactions()
        return events

    def _enqueue_ended_interactions(
        self,
        events: Iterable[InteractionEvent],
    ) -> None:
        for event in events:
            if event.event_type != "ENDED":
                continue

            started_at = _epoch_capture_time(event.started_at)
            ended_at = _epoch_capture_time(event.event_at)
            if ended_at <= started_at:
                raise ActivityProcessorError(
                    "A completed proximity estimate must have positive duration."
                )

            interaction_id = _stable_id(
                "proximity-v1",
                [
                    event.session_id,
                    event.camera_id,
                    event.person_a_id,
                    event.person_b_id,
                    started_at,
                    ended_at,
                ],
            )
            pending = _PendingInteraction(
                interaction_id=interaction_id,
                person_a_id=event.person_a_id,
                person_b_id=event.person_b_id,
                started_at=started_at,
                ended_at=ended_at,
                camera_source=event.camera_id,
            )
            existing = self._pending_interactions.get(interaction_id)
            if existing is not None and existing != pending:
                raise ActivityProcessorError(
                    f"Conflicting deterministic interaction: {interaction_id}."
                )
            self._pending_interactions[interaction_id] = pending

    def _drain_pending_interactions(self) -> None:
        for interaction_id in sorted(tuple(self._pending_interactions)):
            pending = self._pending_interactions[interaction_id]
            # The repository exception intentionally propagates.  The pending
            # item remains queued until a later retry succeeds.
            self.repository.record_proximity_interaction(
                pending.person_a_id,
                pending.person_b_id,
                pending.started_at,
                pending.ended_at,
                camera_source=pending.camera_source,
                estimation_method=PROXIMITY_ESTIMATION_METHOD,
                interaction_id=pending.interaction_id,
            )
            del self._pending_interactions[interaction_id]


__all__ = [
    "ActivityProcessor",
    "ActivityProcessorError",
    "PROXIMITY_ESTIMATION_METHOD",
]
