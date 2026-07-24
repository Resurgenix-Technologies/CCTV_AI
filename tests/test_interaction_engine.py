from __future__ import annotations

import math

import pytest

from src.interaction_engine import (
    InteractionConfig,
    ProximityInteractionEngine,
    TrackObservation,
    normalized_footpoint_distance,
)


def observation(
    person_id: str,
    timestamp: float,
    center_x: float,
    *,
    session_id: str = "session-1",
    camera_id: str = "room-1",
    height: float = 100.0,
    bottom: float = 100.0,
    width: float = 40.0,
) -> TrackObservation:
    return TrackObservation(
        session_id=session_id,
        camera_id=camera_id,
        person_id=person_id,
        timestamp=timestamp,
        box=(
            center_x - width / 2.0,
            bottom - height,
            center_x + width / 2.0,
            bottom,
        ),
    )


def close_pair(
    timestamp: float,
    *,
    first_id: str = "person-b",
    second_id: str = "person-a",
    distance: float = 1.0,
    session_id: str = "session-1",
    camera_id: str = "room-1",
) -> list[TrackObservation]:
    return [
        observation(
            first_id,
            timestamp,
            0.0,
            session_id=session_id,
            camera_id=camera_id,
        ),
        observation(
            second_id,
            timestamp,
            distance * 100.0,
            session_id=session_id,
            camera_id=camera_id,
        ),
    ]


def config(
    *,
    dwell: float = 2.0,
    grace: float = 1.0,
    enter: float = 1.25,
    exit: float = 1.5,
) -> InteractionConfig:
    return InteractionConfig(
        minimum_dwell_seconds=dwell,
        gap_grace_seconds=grace,
        enter_distance_threshold=enter,
        exit_distance_threshold=exit,
    )


def test_normalized_footpoint_distance_is_scale_invariant() -> None:
    first = (0.0, 0.0, 20.0, 100.0)
    second = (100.0, 0.0, 120.0, 100.0)
    scaled_first = tuple(value * 3.0 for value in first)
    scaled_second = tuple(value * 3.0 for value in second)

    assert normalized_footpoint_distance(first, second) == pytest.approx(1.0)
    assert normalized_footpoint_distance(
        scaled_first,
        scaled_second,
    ) == pytest.approx(1.0)


def test_footpoint_distance_includes_vertical_separation() -> None:
    first = (0.0, 0.0, 20.0, 100.0)
    second = (60.0, 80.0, 80.0, 180.0)

    assert normalized_footpoint_distance(first, second) == pytest.approx(1.0)


def test_started_waits_for_dwell_then_updated_uses_canonical_pair() -> None:
    engine = ProximityInteractionEngine(config(dwell=2.0, grace=2.0))

    assert engine.process(close_pair(10.0)) == []
    assert engine.process(close_pair(11.9)) == []

    started = engine.process(close_pair(12.0))
    assert len(started) == 1
    assert started[0].event_type == "STARTED"
    assert started[0].pair == ("person-a", "person-b")
    assert started[0].started_at == 10.0
    assert started[0].event_at == 12.0
    assert started[0].duration_seconds == 2.0
    assert started[0].normalized_distance == pytest.approx(1.0)

    updated = engine.process(close_pair(13.0))
    assert [event.event_type for event in updated] == ["UPDATED"]
    assert updated[0].started_at == 10.0
    assert updated[0].duration_seconds == 3.0


def test_zero_dwell_starts_on_first_qualifying_frame() -> None:
    engine = ProximityInteractionEngine(config(dwell=0.0))

    events = engine.process(close_pair(5.0))

    assert [event.event_type for event in events] == ["STARTED"]
    assert events[0].duration_seconds == 0.0


def test_duplicate_identity_never_forms_a_self_pair() -> None:
    engine = ProximityInteractionEngine(config(dwell=0.0, enter=2.0, exit=2.0))
    observations = [
        observation("person-a", 0.0, 0.0, width=50.0),
        observation("person-a", 0.0, 5.0, width=40.0),
        observation("person-b", 0.0, 100.0),
    ]

    events = engine.process(reversed(observations))

    assert [event.pair for event in events] == [("person-a", "person-b")]
    assert all(event.person_a_id != event.person_b_id for event in events)


def test_event_order_does_not_depend_on_observation_order() -> None:
    observations = [
        observation("charlie", 0.0, 80.0),
        observation("alice", 0.0, 0.0),
        observation("bob", 0.0, 40.0),
    ]
    first_engine = ProximityInteractionEngine(
        config(dwell=0.0, enter=2.0, exit=2.0)
    )
    second_engine = ProximityInteractionEngine(
        config(dwell=0.0, enter=2.0, exit=2.0)
    )

    first_events = first_engine.process(observations)
    second_events = second_engine.process(reversed(observations))

    assert first_events == second_events
    assert [event.pair for event in first_events] == [
        ("alice", "bob"),
        ("alice", "charlie"),
        ("bob", "charlie"),
    ]


def test_gap_within_grace_keeps_one_estimate() -> None:
    engine = ProximityInteractionEngine(config(dwell=0.0, grace=2.0))
    engine.process(close_pair(0.0))

    assert engine.process_frame(
        session_id="session-1",
        camera_id="room-1",
        timestamp=1.0,
    ) == []

    events = engine.process(close_pair(2.0))
    assert [event.event_type for event in events] == ["UPDATED"]
    assert events[0].started_at == 0.0
    assert events[0].duration_seconds == 2.0


def test_gap_expiry_ends_at_last_close_without_inflating_duration() -> None:
    engine = ProximityInteractionEngine(config(dwell=0.0, grace=2.0))
    engine.process(close_pair(10.0))
    engine.process(close_pair(11.0))

    events = engine.process_frame(
        session_id="session-1",
        camera_id="room-1",
        timestamp=13.01,
    )

    assert [event.event_type for event in events] == ["ENDED"]
    assert events[0].started_at == 10.0
    assert events[0].event_at == 11.0
    assert events[0].duration_seconds == 1.0


def test_return_after_grace_ends_old_estimate_before_new_start() -> None:
    engine = ProximityInteractionEngine(config(dwell=0.0, grace=1.0))
    engine.process(close_pair(0.0))

    events = engine.process(close_pair(2.0))

    assert [event.event_type for event in events] == ["ENDED", "STARTED"]
    assert [event.started_at for event in events] == [0.0, 2.0]
    assert [event.event_at for event in events] == [0.0, 2.0]


def test_candidate_that_never_reaches_dwell_ends_silently() -> None:
    engine = ProximityInteractionEngine(config(dwell=5.0, grace=1.0))
    assert engine.process(close_pair(0.0)) == []

    assert engine.process_frame(
        session_id="session-1",
        camera_id="room-1",
        timestamp=2.0,
    ) == []
    assert engine.flush() == []


def test_exit_threshold_provides_distance_hysteresis() -> None:
    engine = ProximityInteractionEngine(
        config(dwell=0.0, grace=10.0, enter=1.0, exit=1.5)
    )
    engine.process(close_pair(0.0, distance=0.9))

    in_hysteresis_band = engine.process(close_pair(1.0, distance=1.3))
    assert [event.event_type for event in in_hysteresis_band] == ["UPDATED"]

    assert engine.process(close_pair(2.0, distance=1.6)) == []

    recovered_in_band = engine.process(close_pair(3.0, distance=1.4))
    assert [event.event_type for event in recovered_in_band] == ["UPDATED"]
    assert recovered_in_band[0].started_at == 0.0


def test_new_candidate_must_cross_enter_threshold() -> None:
    engine = ProximityInteractionEngine(
        config(dwell=0.0, grace=1.0, enter=1.0, exit=1.5)
    )

    assert engine.process(close_pair(0.0, distance=1.2)) == []
    assert engine.process(close_pair(2.0, distance=0.9))[0].event_type == "STARTED"


def test_same_people_are_isolated_by_camera_and_session() -> None:
    engine = ProximityInteractionEngine(config(dwell=0.0))
    observations = (
        close_pair(0.0, session_id="day-2", camera_id="room-1")
        + close_pair(0.0, session_id="day-1", camera_id="room-2")
        + close_pair(0.0, session_id="day-1", camera_id="room-1")
    )

    events = engine.process(reversed(observations))

    assert [(event.session_id, event.camera_id) for event in events] == [
        ("day-1", "room-1"),
        ("day-1", "room-2"),
        ("day-2", "room-1"),
    ]
    assert all(event.pair == ("person-a", "person-b") for event in events)


def test_process_sorts_multiple_frames_before_applying_them() -> None:
    engine = ProximityInteractionEngine(config(dwell=2.0, grace=2.0))
    observations = close_pair(2.0) + close_pair(0.0) + close_pair(1.0)

    events = engine.process(reversed(observations))

    assert [event.event_type for event in events] == ["STARTED"]
    assert events[0].started_at == 0.0
    assert events[0].event_at == 2.0


def test_out_of_order_frame_is_rejected_without_changing_state() -> None:
    engine = ProximityInteractionEngine(config(dwell=0.0, grace=10.0))
    engine.process(close_pair(5.0))

    with pytest.raises(ValueError, match="strictly increasing"):
        engine.process(close_pair(4.0))

    events = engine.process(close_pair(6.0))
    assert [event.event_type for event in events] == ["UPDATED"]
    assert events[0].started_at == 5.0


def test_process_frame_rejects_mismatched_observation_context() -> None:
    engine = ProximityInteractionEngine()
    wrong_camera = observation("person-a", 0.0, 0.0, camera_id="room-2")

    with pytest.raises(ValueError, match="namespace"):
        engine.process_frame(
            session_id="session-1",
            camera_id="room-1",
            timestamp=0.0,
            observations=[wrong_camera],
        )


def test_flush_is_deterministic_ends_active_and_discards_pending() -> None:
    engine = ProximityInteractionEngine(config(dwell=2.0, grace=5.0))
    engine.process(close_pair(0.0, session_id="active", camera_id="cam-b"))
    engine.process(close_pair(2.0, session_id="active", camera_id="cam-b"))
    engine.process(close_pair(0.0, session_id="pending", camera_id="cam-a"))

    events = engine.flush()

    assert [event.event_type for event in events] == ["ENDED"]
    assert (events[0].session_id, events[0].camera_id) == ("active", "cam-b")
    assert events[0].event_at == 2.0
    assert events[0].duration_seconds == 2.0
    assert engine.flush() == []

    # A full flush resets namespace clocks as well as pair state.
    restarted = engine.process(
        close_pair(0.0, session_id="active", camera_id="cam-b")
    )
    assert restarted == []


def test_selective_flush_leaves_other_namespaces_running() -> None:
    engine = ProximityInteractionEngine(config(dwell=0.0, grace=5.0))
    engine.process(close_pair(0.0, session_id="s1", camera_id="cam"))
    engine.process(close_pair(0.0, session_id="s2", camera_id="cam"))

    ended = engine.flush(session_id="s1", camera_id="cam")
    assert [(event.event_type, event.session_id) for event in ended] == [
        ("ENDED", "s1")
    ]

    still_active = engine.process(
        close_pair(1.0, session_id="s2", camera_id="cam")
    )
    assert [(event.event_type, event.session_id) for event in still_active] == [
        ("UPDATED", "s2")
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_dwell_seconds": -1.0},
        {"gap_grace_seconds": -1.0},
        {"enter_distance_threshold": 0.0},
        {"enter_distance_threshold": 2.0, "exit_distance_threshold": 1.0},
        {"exit_distance_threshold": math.inf},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        InteractionConfig(**kwargs)


@pytest.mark.parametrize(
    "box",
    [
        (0.0, 0.0, 0.0, 10.0),
        (0.0, 0.0, 10.0, 0.0),
        (0.0, 0.0, math.nan, 10.0),
    ],
)
def test_invalid_observation_box_is_rejected(box: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        TrackObservation("session", "camera", "person", 0.0, box)  # type: ignore[arg-type]


def test_flush_requires_a_complete_namespace() -> None:
    engine = ProximityInteractionEngine()

    with pytest.raises(ValueError, match="both be set"):
        engine.flush(session_id="session-1")
