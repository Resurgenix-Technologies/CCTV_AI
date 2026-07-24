from __future__ import annotations

import math

from src.global_identity import (
    AssociationConfig,
    CompletedTracklet,
    GlobalIdentityAssociator,
    TrackletId,
    TravelWindow,
)


def unit_vector(angle_radians: float) -> tuple[float, float]:
    return (
        math.cos(angle_radians),
        math.sin(angle_radians),
    )


def tracklet(
    camera_id: str,
    local_track_id: int,
    started_at: float,
    ended_at: float,
    *,
    camera_session_id: str = "session-1",
    embedding: tuple[float, ...] = (1.0, 0.0),
    quality: float = 0.95,
    anchor: str | None = None,
) -> CompletedTracklet:
    return CompletedTracklet(
        tracklet_id=TrackletId(
            camera_session_id=camera_session_id,
            camera_id=camera_id,
            local_track_id=local_track_id,
        ),
        started_at=started_at,
        ended_at=ended_at,
        body_embedding=embedding,
        quality=quality,
        trusted_person_anchor=anchor,
    )


def subject_ids(*values: str):
    return iter(values).__next__


def test_valid_directed_camera_handoff_is_auditable() -> None:
    associator = GlobalIdentityAssociator(
        {("room-a", "room-b"): TravelWindow(3.0, 10.0)},
        config=AssociationConfig(
            similarity_threshold=0.90,
            ambiguity_margin=0.05,
            minimum_quality=0.70,
        ),
        id_factory=subject_ids("subject-1", "subject-2"),
    )
    first = associator.associate(tracklet("room-a", 11, 0.0, 10.0))
    handoff = associator.associate(
        tracklet(
            "room-b",
            22,
            15.0,
            25.0,
            embedding=unit_vector(0.05),
        )
    )

    assert first.outcome == "NEW_SUBJECT"
    assert handoff.outcome == "MATCHED_SUBJECT"
    assert handoff.reason == "CONFIDENT_MATCH"
    assert handoff.subject_id == first.subject_id == "subject-1"
    assert handoff.selected_candidate is not None
    assert handoff.selected_candidate.travel_seconds == 5.0
    assert handoff.selected_candidate.similarity > 0.99
    assert handoff.selected_candidate.rejection_reasons == ()


def test_equal_local_ids_on_different_cameras_remain_namespaced() -> None:
    associator = GlobalIdentityAssociator(
        {("room-a", "room-b"): TravelWindow(0.0, 20.0)},
        id_factory=subject_ids("subject-1", "subject-2"),
    )
    first_evidence = tracklet("room-a", 7, 0.0, 10.0)
    second_evidence = tracklet("room-b", 7, 15.0, 25.0)

    first = associator.associate(first_evidence)
    second = associator.associate(second_evidence)

    assert first_evidence.tracklet_id != second_evidence.tracklet_id
    assert (
        first_evidence.tracklet_id.namespaced_id
        != second_evidence.tracklet_id.namespaced_id
    )
    assert second.outcome == "MATCHED_SUBJECT"
    assert second.subject_id == first.subject_id


def test_impossible_subject_overlap_fails_closed() -> None:
    associator = GlobalIdentityAssociator(
        {("room-a", "room-b"): TravelWindow(0.0, 20.0)},
        id_factory=subject_ids("subject-1", "subject-2"),
    )
    first = associator.associate(tracklet("room-a", 1, 0.0, 10.0))
    overlapping = associator.associate(
        tracklet("room-b", 2, 9.0, 20.0)
    )

    assert overlapping.outcome == "NEW_SUBJECT"
    assert overlapping.subject_id != first.subject_id
    assert overlapping.reason == "NO_ELIGIBLE_CANDIDATE"
    assert "IMPOSSIBLE_OVERLAP" in (
        overlapping.candidate_scores[0].rejection_reasons
    )


def test_unconfigured_directed_transition_is_disallowed() -> None:
    associator = GlobalIdentityAssociator(
        {("room-a", "room-b"): TravelWindow(0.0, 20.0)},
        id_factory=subject_ids("subject-1", "subject-2"),
    )
    first = associator.associate(tracklet("room-a", 1, 0.0, 10.0))
    disallowed = associator.associate(
        tracklet("room-c", 2, 15.0, 25.0)
    )

    assert disallowed.outcome == "NEW_SUBJECT"
    assert disallowed.subject_id != first.subject_id
    assert "TRANSITION_NOT_ALLOWED" in (
        disallowed.candidate_scores[0].rejection_reasons
    )


def test_close_best_and_second_subject_scores_are_ambiguous() -> None:
    associator = GlobalIdentityAssociator(
        {
            ("room-a", "room-b"): TravelWindow(0.0, 20.0),
            ("room-c", "room-b"): TravelWindow(0.0, 20.0),
        },
        config=AssociationConfig(
            similarity_threshold=0.85,
            ambiguity_margin=0.05,
            minimum_quality=0.5,
        ),
        id_factory=subject_ids(
            "subject-1",
            "subject-2",
            "subject-3",
        ),
    )
    first = associator.associate(
        tracklet(
            "room-a",
            1,
            0.0,
            10.0,
            embedding=unit_vector(0.0),
        )
    )
    second = associator.associate(
        tracklet(
            "room-c",
            2,
            0.0,
            10.0,
            embedding=unit_vector(0.04),
        )
    )
    ambiguous = associator.associate(
        tracklet(
            "room-b",
            3,
            15.0,
            25.0,
            embedding=unit_vector(0.02),
        )
    )

    assert first.subject_id == "subject-1"
    assert second.subject_id == "subject-2"
    assert ambiguous.outcome == "NEW_SUBJECT"
    assert ambiguous.subject_id == "subject-3"
    assert ambiguous.reason == "AMBIGUOUS_BEST_MATCH"
    eligible_subjects = {
        score.subject_id
        for score in ambiguous.candidate_scores
        if score.eligible
    }
    assert eligible_subjects == {"subject-1", "subject-2"}
    assert ambiguous.selected_candidate is None


def test_trusted_anchor_conflicts_cannot_be_visual_matches() -> None:
    associator = GlobalIdentityAssociator(
        {
            ("room-a", "room-b"): TravelWindow(0.0, 30.0),
            ("room-a", "room-c"): TravelWindow(0.0, 30.0),
            ("room-b", "room-c"): TravelWindow(0.0, 30.0),
        },
        id_factory=subject_ids(
            "subject-1",
            "subject-2",
            "subject-3",
        ),
    )
    anchored_a = associator.associate(
        tracklet(
            "room-a",
            1,
            0.0,
            10.0,
            anchor="trusted-person-a",
        )
    )
    conflicting = associator.associate(
        tracklet(
            "room-b",
            2,
            15.0,
            25.0,
            anchor="trusted-person-b",
        )
    )

    assert conflicting.outcome == "NEW_SUBJECT"
    assert conflicting.subject_id != anchored_a.subject_id
    assert "ANCHOR_CONFLICT" in (
        conflicting.candidate_scores[0].rejection_reasons
    )

    anchored_a_again = associator.associate(
        tracklet(
            "room-c",
            3,
            30.0,
            40.0,
            anchor="trusted-person-a",
        )
    )
    assert anchored_a_again.outcome == "MATCHED_SUBJECT"
    assert anchored_a_again.subject_id == anchored_a.subject_id
    assert (
        anchored_a_again.subject_trusted_person_anchor
        == "trusted-person-a"
    )
    conflicting_scores = [
        score
        for score in anchored_a_again.candidate_scores
        if score.subject_id == conflicting.subject_id
    ]
    assert conflicting_scores
    assert "ANCHOR_CONFLICT" in conflicting_scores[0].rejection_reasons


def test_trusted_anchor_can_attach_only_after_confident_handoff() -> None:
    associator = GlobalIdentityAssociator(
        {
            ("room-a", "room-b"): TravelWindow(0.0, 20.0),
            ("room-b", "room-c"): TravelWindow(0.0, 20.0),
        },
        id_factory=subject_ids("subject-1", "subject-2"),
    )
    unanchored = associator.associate(tracklet("room-a", 1, 0.0, 10.0))
    anchored = associator.associate(
        tracklet(
            "room-b",
            2,
            15.0,
            25.0,
            anchor="trusted-person-a",
        )
    )

    assert anchored.subject_id == unanchored.subject_id
    assert anchored.subject_trusted_person_anchor == "trusted-person-a"

    conflicting = associator.associate(
        tracklet(
            "room-c",
            3,
            30.0,
            40.0,
            anchor="trusted-person-b",
        )
    )
    assert conflicting.outcome == "NEW_SUBJECT"
    assert conflicting.subject_id == "subject-2"
    assert "ANCHOR_CONFLICT" in (
        conflicting.candidate_scores[0].rejection_reasons
    )
