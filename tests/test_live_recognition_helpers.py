import argparse

import pytest

from scripts.live_recognition import (
    parse_positive_integer,
    parse_normalized_roi,
    should_process_source_frame,
    tracks_inside_roi,
    draw_restricted_roi,
    normalized_roi_from_selection,
    track_camera_label,
    track_event_id,
    track_overlay_line,
    validate_frame_stride,
)
from src.tracker import TrackedPerson


def test_standard_track_metadata_remains_backward_compatible() -> None:
    track = TrackedPerson(7, (1, 2, 30, 50), 0.8)

    assert track_camera_label("NDI: Office", track) == "NDI: Office"
    assert track_event_id(track) == 7
    assert track_overlay_line(track) == "Local track: 7"


def test_quadrant_track_uses_camera_local_event_namespace() -> None:
    track = TrackedPerson(
        track_id=19,
        box=(101, 202, 130, 250),
        confidence=0.8,
        local_track_id=3,
        quadrant_index=4,
    )

    assert track_camera_label("NDI: Office", track) == "NDI: Office [Q4]"
    assert track_event_id(track) == 3
    assert track_overlay_line(track) == "Camera Q4 / track 3"


def test_recovered_quadrant_track_has_unique_event_identity() -> None:
    track = TrackedPerson(
        track_id=(2 << 32) | 19,
        box=(101, 202, 130, 250),
        confidence=0.8,
        local_track_id=3,
        quadrant_index=4,
        quadrant_epoch=2,
    )

    assert track_event_id(track) == (2 << 32) | 3
    assert track_overlay_line(track) == "Camera Q4 / track 3 / recovery 2"


def test_offline_stride_schedule_preserves_first_source_frame() -> None:
    selected = [
        frame
        for frame in range(1, 13)
        if should_process_source_frame(frame, 5)
    ]

    assert selected == [1, 6, 11]


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_positive_integer_parser_rejects_invalid_stride(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_positive_integer(value)


def test_frame_stride_is_recording_only() -> None:
    validate_frame_stride(5, is_recording=True)
    validate_frame_stride(1, is_recording=False)

    with pytest.raises(ValueError, match="recorded videos"):
        validate_frame_stride(5, is_recording=False)


def test_restricted_roi_filters_track_centers() -> None:
    tracks = [
        TrackedPerson(1, (0, 0, 20, 20), 0.8),
        TrackedPerson(2, (80, 80, 100, 100), 0.8),
    ]

    kept = tracks_inside_roi(tracks, (100, 100, 3), (0.25, 0.25, 0.75, 0.75))

    assert kept == []
    assert tracks_inside_roi(tracks, (100, 100, 3), None) == tracks


def test_restricted_roi_parser() -> None:
    assert parse_normalized_roi("0.1, 0.2, 0.8, 0.9") == (
        0.1,
        0.2,
        0.8,
        0.9,
    )
    with pytest.raises(argparse.ArgumentTypeError):
        parse_normalized_roi("0,0,1")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_normalized_roi("0.8,0.1,0.2,0.9")


def test_restricted_roi_is_drawn_on_frame() -> None:
    import numpy as np

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_restricted_roi(frame, (0.25, 0.25, 0.75, 0.75))

    assert np.count_nonzero(frame) > 0


def test_pixel_roi_selection_becomes_normalized_roi() -> None:
    assert normalized_roi_from_selection((100, 50, 400, 200), (400, 800, 3)) == (
        0.125,
        0.125,
        0.625,
        0.625,
    )
    assert normalized_roi_from_selection((0, 0, 1, 1), (400, 800, 3)) is None
