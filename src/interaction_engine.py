"""Deterministic proximity-based interaction estimates.

This module deliberately makes a narrow claim: it estimates an interaction
from sustained spatial proximity.  It does not prove that people spoke, paid
attention to one another, or had a conversation.

Detections are namespaced by both session and camera.  Identity values are
therefore expected to be stable across cameras, while proximity is evaluated
only for people visible in the same camera frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import hypot, isfinite
from typing import Iterable, Literal, Sequence


Box = tuple[float, float, float, float]
Pair = tuple[str, str]
EventType = Literal["STARTED", "UPDATED", "ENDED"]


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


def _validated_box(box: Sequence[float]) -> Box:
    if len(box) != 4:
        raise ValueError("box must contain exactly (x1, y1, x2, y2)")
    x1, y1, x2, y2 = (
        _finite_number(value, "box coordinate") for value in box
    )
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must have positive width and height")
    return (x1, y1, x2, y2)


def normalized_footpoint_distance(
    first_box: Sequence[float],
    second_box: Sequence[float],
) -> float:
    """Return footpoint distance divided by the boxes' mean height.

    A box footpoint is the bottom-center point ``((x1 + x2) / 2, y2)``.
    Normalizing by mean person-box height makes a threshold useful across
    image resolutions and moderately different camera perspectives.
    """

    first = _validated_box(first_box)
    second = _validated_box(second_box)

    first_x = (first[0] + first[2]) / 2.0
    second_x = (second[0] + second[2]) / 2.0
    first_y = first[3]
    second_y = second[3]
    mean_height = ((first[3] - first[1]) + (second[3] - second[1])) / 2.0

    return hypot(first_x - second_x, first_y - second_y) / mean_height


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """One identified person box from one timestamped camera frame."""

    session_id: str
    camera_id: str
    person_id: str
    timestamp: float
    box: Box

    def __post_init__(self) -> None:
        for field_name in ("session_id", "camera_id", "person_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        object.__setattr__(
            self,
            "timestamp",
            _finite_number(self.timestamp, "timestamp"),
        )
        object.__setattr__(self, "box", _validated_box(self.box))

    @property
    def pair_namespace(self) -> tuple[str, str]:
        return (self.session_id, self.camera_id)


@dataclass(frozen=True, slots=True)
class InteractionConfig:
    """Thresholds for the proximity interaction state machine.

    ``enter_distance_threshold`` starts a candidate.  The equal-or-larger
    ``exit_distance_threshold`` supplies distance hysteresis once a candidate
    exists.  A missing or too-distant pair retains its state for
    ``gap_grace_seconds`` so short detector misses do not split an estimate.
    """

    minimum_dwell_seconds: float = 3.0
    gap_grace_seconds: float = 1.0
    enter_distance_threshold: float = 1.25
    exit_distance_threshold: float = 1.50

    def __post_init__(self) -> None:
        for field_name in (
            "minimum_dwell_seconds",
            "gap_grace_seconds",
            "enter_distance_threshold",
            "exit_distance_threshold",
        ):
            value = _finite_number(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, value)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")

        if self.enter_distance_threshold == 0:
            raise ValueError("enter_distance_threshold must be greater than zero")
        if self.exit_distance_threshold < self.enter_distance_threshold:
            raise ValueError(
                "exit_distance_threshold must be greater than or equal to "
                "enter_distance_threshold"
            )


@dataclass(frozen=True, slots=True)
class InteractionEvent:
    """A lifecycle event for a proximity-based interaction estimate.

    ``started_at`` is the first qualifying proximity observation, even though
    ``STARTED`` is emitted only after the configured dwell time.  ``event_at``
    is the confirming sample for ``STARTED``/``UPDATED`` and the last
    qualifying sample for ``ENDED``.  Thus grace time is not added to the
    reported ``duration_seconds``.
    """

    event_type: EventType
    session_id: str
    camera_id: str
    person_a_id: str
    person_b_id: str
    started_at: float
    event_at: float
    duration_seconds: float
    normalized_distance: float

    @property
    def pair(self) -> Pair:
        return (self.person_a_id, self.person_b_id)


@dataclass(slots=True)
class _PairState:
    started_at: float
    last_close_at: float
    last_distance: float
    active: bool = False


class ProximityInteractionEngine:
    """Estimate sustained interactions from identified person observations.

    Frames for a namespace must be submitted in strictly increasing timestamp
    order.  :meth:`process` may receive observations from multiple frames and
    sorts them before applying them.  :meth:`process_frame` also accepts an
    empty frame, which is useful for advancing gap time when nobody is visible.
    """

    def __init__(self, config: InteractionConfig | None = None) -> None:
        self.config = config or InteractionConfig()
        self._states: dict[tuple[str, str, str, str], _PairState] = {}
        self._last_frame_at: dict[tuple[str, str], float] = {}

    def process(
        self,
        observations: Iterable[TrackObservation],
    ) -> list[InteractionEvent]:
        """Process one or more timestamped frames in deterministic order.

        Observations with the same ``(session_id, camera_id, timestamp)`` are
        one frame.  Duplicate observations for an identity in that frame are
        collapsed deterministically, so they can never create a self-pair.
        """

        frames: dict[
            tuple[float, str, str],
            list[TrackObservation],
        ] = {}
        for observation in observations:
            if not isinstance(observation, TrackObservation):
                raise TypeError("observations must contain TrackObservation values")
            frame_key = (
                observation.timestamp,
                observation.session_id,
                observation.camera_id,
            )
            frames.setdefault(frame_key, []).append(observation)

        ordered_frames = sorted(frames.items())
        prospective_last = dict(self._last_frame_at)
        for (timestamp, session_id, camera_id), _ in ordered_frames:
            namespace = (session_id, camera_id)
            previous = prospective_last.get(namespace)
            if previous is not None and timestamp <= previous:
                raise ValueError(
                    "frame timestamps must be strictly increasing within "
                    f"namespace {namespace!r}: {timestamp} <= {previous}"
                )
            prospective_last[namespace] = timestamp

        events: list[InteractionEvent] = []
        for (timestamp, session_id, camera_id), frame in ordered_frames:
            events.extend(
                self._apply_frame(
                    session_id=session_id,
                    camera_id=camera_id,
                    timestamp=timestamp,
                    observations=frame,
                )
            )
        return events

    def process_frame(
        self,
        *,
        session_id: str,
        camera_id: str,
        timestamp: float,
        observations: Iterable[TrackObservation] = (),
    ) -> list[InteractionEvent]:
        """Process exactly one frame, including an explicitly empty frame."""

        self._validate_namespace(session_id, camera_id)
        frame_at = _finite_number(timestamp, "timestamp")
        frame = list(observations)
        for observation in frame:
            if not isinstance(observation, TrackObservation):
                raise TypeError("observations must contain TrackObservation values")
            if observation.pair_namespace != (session_id, camera_id):
                raise ValueError("observation does not match the frame namespace")
            if observation.timestamp != frame_at:
                raise ValueError("observation does not match the frame timestamp")

        previous = self._last_frame_at.get((session_id, camera_id))
        if previous is not None and frame_at <= previous:
            raise ValueError(
                "frame timestamps must be strictly increasing within "
                f"namespace {(session_id, camera_id)!r}: {frame_at} <= {previous}"
            )

        return self._apply_frame(
            session_id=session_id,
            camera_id=camera_id,
            timestamp=frame_at,
            observations=frame,
        )

    def flush(
        self,
        *,
        session_id: str | None = None,
        camera_id: str | None = None,
    ) -> list[InteractionEvent]:
        """End active estimates and discard pending candidates.

        With no arguments all namespaces are flushed.  Supplying both IDs
        flushes only that namespace.  End time is always the last qualifying
        proximity sample.  A second flush is therefore idempotent.
        """

        if (session_id is None) != (camera_id is None):
            raise ValueError(
                "session_id and camera_id must either both be set or both be omitted"
            )
        namespace: tuple[str, str] | None = None
        if session_id is not None and camera_id is not None:
            self._validate_namespace(session_id, camera_id)
            namespace = (session_id, camera_id)

        selected_keys = [
            key
            for key in sorted(self._states)
            if namespace is None or key[:2] == namespace
        ]
        events: list[InteractionEvent] = []
        for key in selected_keys:
            state = self._states.pop(key)
            if state.active:
                events.append(self._event("ENDED", key, state, state.last_close_at))

        if namespace is None:
            self._last_frame_at.clear()
        else:
            self._last_frame_at.pop(namespace, None)
        return events

    @staticmethod
    def _validate_namespace(session_id: str, camera_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise ValueError("camera_id must be a non-empty string")

    def _apply_frame(
        self,
        *,
        session_id: str,
        camera_id: str,
        timestamp: float,
        observations: list[TrackObservation],
    ) -> list[InteractionEvent]:
        namespace = (session_id, camera_id)
        events = self._expire_gaps(namespace, timestamp)

        # A ReID pipeline can briefly produce duplicate boxes for one global
        # identity.  Prefer the largest box, then lexicographically smallest
        # coordinates, to collapse those duplicates independently of input
        # ordering.
        by_person: dict[str, TrackObservation] = {}
        for observation in observations:
            current = by_person.get(observation.person_id)
            if current is None or self._box_preference(observation) < self._box_preference(
                current
            ):
                by_person[observation.person_id] = observation

        for person_a_id, person_b_id in combinations(sorted(by_person), 2):
            first = by_person[person_a_id]
            second = by_person[person_b_id]
            distance = normalized_footpoint_distance(first.box, second.box)
            key = (session_id, camera_id, person_a_id, person_b_id)
            state = self._states.get(key)
            threshold = (
                self.config.enter_distance_threshold
                if state is None
                else self.config.exit_distance_threshold
            )
            if distance > threshold:
                continue

            if state is None:
                state = _PairState(
                    started_at=timestamp,
                    last_close_at=timestamp,
                    last_distance=distance,
                )
                self._states[key] = state
            else:
                state.last_close_at = timestamp
                state.last_distance = distance

            duration = timestamp - state.started_at
            if not state.active and duration >= self.config.minimum_dwell_seconds:
                state.active = True
                events.append(self._event("STARTED", key, state, timestamp))
            elif state.active:
                events.append(self._event("UPDATED", key, state, timestamp))

        self._last_frame_at[namespace] = timestamp
        return events

    def _expire_gaps(
        self,
        namespace: tuple[str, str],
        timestamp: float,
    ) -> list[InteractionEvent]:
        expired_keys = [
            key
            for key, state in sorted(self._states.items())
            if key[:2] == namespace
            and timestamp - state.last_close_at > self.config.gap_grace_seconds
        ]
        events: list[InteractionEvent] = []
        for key in expired_keys:
            state = self._states.pop(key)
            if state.active:
                events.append(self._event("ENDED", key, state, state.last_close_at))
        return events

    @staticmethod
    def _box_preference(observation: TrackObservation) -> tuple[float, Box]:
        x1, y1, x2, y2 = observation.box
        area = (x2 - x1) * (y2 - y1)
        return (-area, observation.box)

    @staticmethod
    def _event(
        event_type: EventType,
        key: tuple[str, str, str, str],
        state: _PairState,
        event_at: float,
    ) -> InteractionEvent:
        session_id, camera_id, person_a_id, person_b_id = key
        return InteractionEvent(
            event_type=event_type,
            session_id=session_id,
            camera_id=camera_id,
            person_a_id=person_a_id,
            person_b_id=person_b_id,
            started_at=state.started_at,
            event_at=event_at,
            duration_seconds=event_at - state.started_at,
            normalized_distance=state.last_distance,
        )


__all__ = [
    "Box",
    "EventType",
    "InteractionConfig",
    "InteractionEvent",
    "ProximityInteractionEngine",
    "TrackObservation",
    "normalized_footpoint_distance",
]
