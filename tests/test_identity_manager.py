from src.identity_manager import IdentityManager
from src.recognizer import RecognitionResult


def known(
    person_id: str,
    name: str,
    score: float,
) -> RecognitionResult:
    return RecognitionResult(
        True,
        person_id,
        name,
        score,
        1,
    )


def test_identity_voting_confirmation() -> None:
    manager = IdentityManager(
        confirmation_votes=3,
        vote_window=5,
        ttl_frames=60,
    )

    for frame in (10, 20, 30):
        state = manager.observe(
            7,
            known(
                "person-a",
                "Shrayan Sarkar",
                0.8,
            ),
            frame,
        )

    assert state.confirmed
    assert state.person_id == "person-a"

    progress = manager.vote_progress(7)
    assert progress.status == "CONFIRMED"
    assert progress.votes == 3


def test_conflicting_votes_stay_pending() -> None:
    manager = IdentityManager(
        confirmation_votes=3,
        vote_window=5,
        ttl_frames=60,
    )

    manager.observe(
        3,
        known("a", "A", 0.8),
        10,
    )
    manager.observe(
        3,
        known("b", "B", 0.82),
        20,
    )
    manager.observe(
        3,
        known("a", "A", 0.81),
        30,
    )
    manager.observe(
        3,
        known("b", "B", 0.83),
        40,
    )

    state = manager.get(3)
    progress = manager.vote_progress(3)

    assert not state.confirmed
    assert state.full_name == "UNKNOWN"
    assert progress.status == "PENDING"
    assert progress.votes == 2


def test_unknown_result_does_not_confirm() -> None:
    manager = IdentityManager(
        confirmation_votes=3,
        vote_window=5,
        ttl_frames=60,
    )

    unknown = RecognitionResult.unknown(
        0.31
    )

    for frame in (
        10,
        20,
        30,
        40,
        50,
    ):
        manager.observe(
            9,
            unknown,
            frame,
        )

    state = manager.get(9)
    progress = manager.vote_progress(9)

    assert not state.confirmed
    assert state.similarity == 0.31
    assert progress.status == "UNKNOWN"
    assert progress.votes == 0


def test_unknown_results_age_out_old_known_votes() -> None:
    manager = IdentityManager(
        confirmation_votes=3,
        vote_window=5,
        ttl_frames=60,
    )

    manager.observe(
        4,
        known("a", "A", 0.80),
        10,
    )
    manager.observe(
        4,
        known("a", "A", 0.81),
        20,
    )

    unknown = RecognitionResult.unknown(
        0.30
    )

    for frame in (
        30,
        40,
        50,
        60,
        70,
    ):
        manager.observe(
            4,
            unknown,
            frame,
        )

    progress = manager.vote_progress(4)

    assert progress.status == "UNKNOWN"
    assert progress.votes == 0
    assert not manager.get(4).confirmed


def test_confirmed_identity_survives_weak_result() -> None:
    manager = IdentityManager(
        confirmation_votes=3,
        vote_window=5,
        ttl_frames=60,
    )

    for frame, score in (
        (10, 0.70),
        (20, 0.75),
        (30, 0.80),
    ):
        state = manager.observe(
            8,
            known(
                "person-a",
                "A",
                score,
            ),
            frame,
        )

    confirmed_score = state.similarity

    state = manager.observe(
        8,
        RecognitionResult.unknown(0.22),
        40,
    )

    assert state.confirmed
    assert state.person_id == "person-a"
    assert state.similarity == confirmed_score


def test_track_start_and_expiry_events() -> None:
    manager = IdentityManager(
        confirmation_votes=3,
        vote_window=5,
        ttl_frames=5,
    )

    started = manager.mark_seen(
        {11},
        frame_index=1,
    )

    assert [
        state.track_id
        for state in started
    ] == [11]

    assert manager.mark_seen(
        {11},
        frame_index=2,
    ) == []

    expired = manager.expire(
        frame_index=8,
    )

    assert [
        state.track_id
        for state in expired
    ] == [11]


def test_close_all_returns_active_tracks_once() -> None:
    manager = IdentityManager(
        confirmation_votes=2,
        vote_window=3,
        ttl_frames=5,
    )

    manager.mark_seen({9, 2}, frame_index=10)

    assert [
        state.track_id
        for state in manager.close_all()
    ] == [2, 9]
    assert manager.close_all() == []
