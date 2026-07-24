from dataclasses import dataclass

import pytest

from src.association import (
    assign_faces_to_tracks,
    box_iou,
    face_matches_person,
    find_matching_track,
)


@dataclass
class Track:
    track_id: int
    box: tuple[int, int, int, int]


def test_face_center_inside_upper_person_box() -> None:
    assert face_matches_person((40, 20, 60, 40), (20, 10, 100, 210))


def test_face_outside_person_box() -> None:
    assert not face_matches_person((140, 20, 160, 40), (20, 10, 100, 210))


def test_two_faces_map_to_separate_people() -> None:
    tracks = [
        Track(1, (0, 0, 100, 220)),
        Track(2, (120, 0, 220, 220)),
    ]
    assert find_matching_track((30, 20, 60, 60), tracks).track_id == 1
    assert find_matching_track((150, 20, 180, 60), tracks).track_id == 2


def test_one_to_one_assignment_never_reuses_a_track() -> None:
    tracks = [
        Track(1, (0, 0, 120, 220)),
        Track(2, (100, 0, 220, 220)),
    ]
    assignments = assign_faces_to_tracks(
        [(35, 20, 70, 60), (145, 20, 180, 60)],
        tracks,
    )

    assert assignments[0].track_id == 1
    assert assignments[1].track_id == 2
    assert len({track.track_id for track in assignments.values()}) == 2


def test_assignment_priority_can_prefer_identity_quality() -> None:
    tracks = [Track(1, (0, 0, 120, 220))]
    assignments = assign_faces_to_tracks(
        [(55, 25, 65, 37), (40, 20, 82, 64)],
        tracks,
        face_priorities=[1.0, 2.0],
    )

    assert set(assignments) == {1}
    assert assignments[1].track_id == 1


def test_assignment_maximizes_cardinality_before_distance() -> None:
    tracks = [
        Track(1, (0, 0, 120, 220)),
        Track(2, (80, 0, 200, 220)),
    ]
    assignments = assign_faces_to_tracks(
        [
            (0, 15, 10, 55),
            (85, 15, 95, 55),
        ],
        tracks,
    )

    assert assignments[0].track_id == 1
    assert assignments[1].track_id == 2


def test_assignment_is_invariant_to_track_input_order() -> None:
    first = Track(1, (0, 0, 120, 220))
    second = Track(2, (0, 0, 120, 220))
    faces = [(40, 20, 80, 60), (40, 20, 80, 60)]

    forward = assign_faces_to_tracks(faces, [first, second])
    reversed_order = assign_faces_to_tracks(faces, [second, first])

    assert {
        face_index: track.track_id
        for face_index, track in forward.items()
    } == {
        face_index: track.track_id
        for face_index, track in reversed_order.items()
    }


def test_box_iou_and_priority_validation() -> None:
    assert box_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(
        25 / 175
    )
    assert box_iou((0, 0, 5, 5), (6, 6, 10, 10)) == 0.0
    with pytest.raises(ValueError, match="one value per face"):
        assign_faces_to_tracks(
            [(0, 0, 10, 10)],
            [Track(1, (0, 0, 20, 40))],
            face_priorities=[],
        )
    with pytest.raises(ValueError, match="integer tiers"):
        assign_faces_to_tracks(
            [(0, 0, 10, 10)],
            [Track(1, (0, 0, 20, 40))],
            face_priorities=[1.5],
        )
