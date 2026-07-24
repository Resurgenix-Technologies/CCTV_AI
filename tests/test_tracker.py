from __future__ import annotations

import numpy as np
import pytest

import src.tracker as tracker_module
from src.tracker import (
    has_meaningful_video_content,
    PersonTracker,
    QuadrantPersonTracker,
    TrackedPerson,
    split_frame_2x2,
)


class FakeYOLOModel:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def track(self, **kwargs: object) -> list[object]:
        self.kwargs = kwargs
        return []


def test_tracker_passes_configurable_inference_size(monkeypatch) -> None:
    model = FakeYOLOModel()
    monkeypatch.setattr(
        tracker_module,
        "YOLO",
        lambda model_name: model,
    )
    tracker = PersonTracker(
        "fake.pt",
        0.35,
        "bytetrack.yaml",
        image_size=960,
    )
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    assert tracker.track(frame) == []
    assert model.kwargs is not None
    assert model.kwargs["source"] is frame
    assert model.kwargs["persist"] is True
    assert model.kwargs["classes"] == [0]
    assert model.kwargs["conf"] == 0.35
    assert model.kwargs["tracker"] == "bytetrack.yaml"
    assert model.kwargs["imgsz"] == 960
    assert model.kwargs["verbose"] is False


def test_tracker_rejects_invalid_inference_size(monkeypatch) -> None:
    monkeypatch.setattr(
        tracker_module,
        "YOLO",
        lambda model_name: FakeYOLOModel(),
    )
    with pytest.raises(ValueError, match="at least 32"):
        PersonTracker("fake.pt", 0.35, "bytetrack.yaml", image_size=16)


def test_split_frame_2x2_preserves_odd_frame_pixels() -> None:
    frame = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)

    quadrants = split_frame_2x2(frame)

    assert [item.index for item in quadrants] == [1, 2, 3, 4]
    assert [item.bounds for item in quadrants] == [
        (0, 0, 3, 2),
        (3, 0, 7, 2),
        (0, 2, 3, 5),
        (3, 2, 7, 5),
    ]
    assert [item.image.shape[:2] for item in quadrants] == [
        (2, 3),
        (2, 4),
        (3, 3),
        (3, 4),
    ]
    assert all(item.image.flags.c_contiguous for item in quadrants)

    recomposed = np.empty_like(frame)
    for item in quadrants:
        x1, y1, x2, y2 = item.bounds
        recomposed[y1:y2, x1:x2] = item.image
    np.testing.assert_array_equal(recomposed, frame)


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((1, 4, 3), dtype=np.uint8),
        np.zeros((4, 1, 3), dtype=np.uint8),
        np.zeros((4, 4), dtype=np.uint8),
        np.zeros((4, 4, 4), dtype=np.uint8),
    ],
)
def test_split_frame_2x2_rejects_invalid_frames(frame: np.ndarray) -> None:
    with pytest.raises(ValueError, match="2x2 layout"):
        split_frame_2x2(frame)


def test_video_content_gate_rejects_loss_screen_and_accepts_scene() -> None:
    loss = np.zeros((200, 300, 3), dtype=np.uint8)
    loss[80:100, 140:160] = 255
    flat_blue_loss = np.full((200, 300, 3), (120, 24, 8), dtype=np.uint8)
    flat_blue_loss[90:110, 130:170] = 255
    scene = np.zeros((200, 300, 3), dtype=np.uint8)
    scene[:, :40] = (30, 30, 30)

    assert not has_meaningful_video_content(loss)
    assert not has_meaningful_video_content(flat_blue_loss)
    assert has_meaningful_video_content(scene)


class FakeQuadrantBackend:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def track(self, frame: np.ndarray) -> list[TrackedPerson]:
        self.frames.append(frame.copy())
        height, width = frame.shape[:2]
        return [
            TrackedPerson(
                track_id=1,
                box=(0, 0, width, height),
                confidence=0.8,
            )
        ]


def test_quadrant_tracker_isolates_ids_and_maps_boxes_to_mosaic() -> None:
    backends: list[FakeQuadrantBackend] = []

    def factory(
        model_name: str,
        confidence: float,
        tracker_config: str,
        image_size: int,
    ) -> FakeQuadrantBackend:
        assert (model_name, confidence, tracker_config, image_size) == (
            "fake.pt",
            0.35,
            "bytetrack.yaml",
            640,
        )
        backend = FakeQuadrantBackend()
        backends.append(backend)
        return backend

    tracker = QuadrantPersonTracker(
        "fake.pt",
        0.35,
        "bytetrack.yaml",
        640,
        tracker_factory=factory,
        suppress_startup_inactive=False,
    )
    frame = np.zeros((5, 7, 3), dtype=np.uint8)
    for quadrant in split_frame_2x2(frame):
        x1, y1, x2, y2 = quadrant.bounds
        frame[y1:y2, x1:x2] = quadrant.index

    first = tracker.track(frame)
    second = tracker.track(frame)

    assert len(backends) == 4
    assert all(len(backend.frames) == 3 for backend in backends)
    assert all(
        np.count_nonzero(backend.frames[0]) == 0
        for backend in backends
    )
    assert [int(backend.frames[1][0, 0, 0]) for backend in backends] == [
        1,
        2,
        3,
        4,
    ]
    assert all(
        np.array_equal(backend.frames[1], backend.frames[2])
        for backend in backends
    )
    assert [item.track_id for item in first] == [1, 2, 3, 4]
    assert [item.track_id for item in second] == [1, 2, 3, 4]
    assert [item.local_track_id for item in first] == [1, 1, 1, 1]
    assert [item.quadrant_index for item in first] == [1, 2, 3, 4]
    assert [item.box for item in first] == [
        (0, 0, 3, 2),
        (3, 0, 7, 2),
        (0, 2, 3, 5),
        (3, 2, 7, 5),
    ]

    with pytest.raises(RuntimeError, match="resolution changed"):
        tracker.track(np.zeros((6, 8, 3), dtype=np.uint8))


class SequencedQuadrantBackend:
    def __init__(self, local_ids: list[int | None]) -> None:
        self.local_ids = local_ids
        self.calls = 0

    def track(self, frame: np.ndarray) -> list[TrackedPerson]:
        self.calls += 1
        if self.calls == 1:
            return []  # One-time blank tracker initialization.
        sequence_index = self.calls - 2
        local_id = self.local_ids[sequence_index]
        if local_id is None:
            return []
        return [TrackedPerson(local_id, (0, 0, 2, 2), 0.8)]


def test_quadrant_global_ids_handle_delayed_heterogeneous_local_ids() -> None:
    sequences = [
        [2, 2],
        [1, 1],
        [None, 3],
        [2, 2],
    ]
    factory_index = 0

    def factory(
        model_name: str,
        confidence: float,
        tracker_config: str,
        image_size: int,
    ) -> SequencedQuadrantBackend:
        nonlocal factory_index
        backend = SequencedQuadrantBackend(sequences[factory_index])
        factory_index += 1
        return backend

    tracker = QuadrantPersonTracker(
        "fake.pt",
        0.35,
        "bytetrack.yaml",
        tracker_factory=factory,
        suppress_startup_inactive=False,
    )
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    first = tracker.track(frame)
    second = tracker.track(frame)

    assert [item.track_id for item in first] == [5, 2, 8]
    assert [item.track_id for item in second] == [5, 2, 11, 8]
    assert len({item.track_id for item in second}) == len(second)


def test_quadrant_tracker_suppresses_startup_loss_and_activates_recovery() -> None:
    backends: list[FakeQuadrantBackend] = []

    def factory(
        model_name: str,
        confidence: float,
        tracker_config: str,
        image_size: int,
    ) -> FakeQuadrantBackend:
        backend = FakeQuadrantBackend()
        backends.append(backend)
        return backend

    tracker = QuadrantPersonTracker(
        "fake.pt",
        0.35,
        "bytetrack.yaml",
        tracker_factory=factory,
    )
    loss_frame = np.zeros((200, 200, 3), dtype=np.uint8)

    for _ in range(3):
        assert tracker.track(loss_frame) == []
    assert all(len(backend.frames) == 4 for backend in backends)

    recovered = loss_frame.copy()
    recovered[:100, :50] = 80
    assert tracker.track(recovered) == []
    results = tracker.track(recovered)

    assert len(results) == 1
    assert results[0].quadrant_index == 1
    assert results[0].quadrant_epoch == 1
    assert results[0].track_id == (1 << 32) | 1
    assert len(backends[0].frames) == 5
    assert all(len(backend.frames) == 4 for backend in backends[1:])


def test_quadrant_loss_after_activation_gets_new_identity_epoch() -> None:
    backends: list[FakeQuadrantBackend] = []

    def factory(
        model_name: str,
        confidence: float,
        tracker_config: str,
        image_size: int,
    ) -> FakeQuadrantBackend:
        backend = FakeQuadrantBackend()
        backends.append(backend)
        return backend

    tracker = QuadrantPersonTracker(
        "fake.pt",
        0.35,
        "bytetrack.yaml",
        tracker_factory=factory,
    )
    live = np.zeros((200, 200, 3), dtype=np.uint8)
    live[:, ::2] = (100, 100, 100)
    loss = np.zeros_like(live)

    before_loss = tracker.track(live)
    assert [item.track_id for item in before_loss] == [1, 2, 3, 4]

    for _ in range(3):
        assert tracker.track(loss) == []
    calls_while_offline = [backend.frames.__len__() for backend in backends]
    assert tracker.track(loss) == []
    assert [backend.frames.__len__() for backend in backends] == (
        calls_while_offline
    )

    assert tracker.track(live) == []
    after_recovery = tracker.track(live)

    assert [item.quadrant_epoch for item in after_recovery] == [1, 1, 1, 1]
    assert [item.track_id for item in after_recovery] == [
        (1 << 32) | 1,
        (1 << 32) | 2,
        (1 << 32) | 3,
        (1 << 32) | 4,
    ]
    assert not (
        {item.track_id for item in before_loss}
        & {item.track_id for item in after_recovery}
    )
