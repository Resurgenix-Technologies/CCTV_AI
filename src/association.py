"""Associate detected faces with tracked person boxes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


Box = tuple[int, int, int, int]


class HasPersonBox(Protocol):
    track_id: int
    box: Box


@dataclass
class _FlowEdge:
    """Mutable residual edge used by the small bipartite matcher."""

    to: int
    reverse_index: int
    capacity: int
    cost: float


def _add_flow_edge(
    graph: list[list[_FlowEdge]],
    source: int,
    target: int,
    capacity: int,
    cost: float,
) -> _FlowEdge:
    forward = _FlowEdge(target, len(graph[target]), capacity, cost)
    reverse = _FlowEdge(source, len(graph[source]), 0, -cost)
    graph[source].append(forward)
    graph[target].append(reverse)
    return forward


def face_matches_person(
    face_box: Box,
    person_box: Box,
    upper_body_ratio: float = 0.45,
) -> bool:
    fx1, fy1, fx2, fy2 = face_box
    px1, py1, px2, py2 = person_box

    face_center_x = (fx1 + fx2) / 2.0
    face_center_y = (fy1 + fy2) / 2.0
    upper_body_limit = py1 + (py2 - py1) * upper_body_ratio

    return (
        px1 <= face_center_x <= px2
        and py1 <= face_center_y <= upper_body_limit
    )


def find_matching_track(
    face_box: Box,
    tracks: list[HasPersonBox],
    upper_body_ratio: float = 0.45,
) -> HasPersonBox | None:
    candidates = [
        track
        for track in tracks
        if face_matches_person(
            face_box,
            track.box,
            upper_body_ratio,
        )
    ]
    if not candidates:
        return None

    face_center_x = (face_box[0] + face_box[2]) / 2.0
    face_center_y = (face_box[1] + face_box[3]) / 2.0

    def distance(track: HasPersonBox) -> float:
        x1, y1, x2, y2 = track.box
        center_x = (x1 + x2) / 2.0
        upper_center_y = (
            y1 + (y2 - y1) * upper_body_ratio * 0.5
        )
        return (center_x - face_center_x) ** 2 + (
            upper_center_y - face_center_y
        ) ** 2

    return min(candidates, key=distance)


def box_iou(first: Box, second: Box) -> float:
    """Return intersection-over-union for two frame-coordinate boxes."""

    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection <= 0:
        return 0.0

    first_area = max(0, first[2] - first[0]) * max(
        0, first[3] - first[1]
    )
    second_area = max(0, second[2] - second[0]) * max(
        0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def assign_faces_to_tracks(
    face_boxes: list[Box],
    tracks: list[HasPersonBox],
    upper_body_ratio: float = 0.45,
    face_priorities: list[float] | None = None,
) -> dict[int, HasPersonBox]:
    """
    Make a deterministic one-face/one-track assignment.

    A face and track are eligible only when the face center lies inside the
    configured upper-body region. A min-cost maximum-cardinality bipartite
    match then minimizes normalized head distance without reusing a face or
    track. Optional priorities are discrete quality tiers; they decide which
    faces survive only when there are more faces than assignable tracks.
    """

    if not 0.0 < upper_body_ratio <= 1.0:
        raise ValueError("upper_body_ratio must be between 0 and 1.")
    if (
        face_priorities is not None
        and len(face_priorities) != len(face_boxes)
    ):
        raise ValueError(
            "face_priorities must contain one value per face box."
        )
    if len({track.track_id for track in tracks}) != len(tracks):
        raise ValueError("track IDs must be unique within one frame.")

    try:
        priorities = (
            [float(value) for value in face_priorities]
            if face_priorities is not None
            else [0.0] * len(face_boxes)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "face priorities must be finite non-negative integer tiers."
        ) from exc
    if not all(math.isfinite(value) for value in priorities):
        raise ValueError("face priorities must be finite.")
    if any(value < 0.0 or not value.is_integer() for value in priorities):
        raise ValueError(
            "face priorities must be non-negative integer tiers."
        )

    ordered_tracks = sorted(tracks, key=lambda track: track.track_id)

    eligible_pairs: list[tuple[int, int, float]] = []

    for face_index, face_box in enumerate(face_boxes):
        face_center_x = (face_box[0] + face_box[2]) / 2.0
        face_center_y = (face_box[1] + face_box[3]) / 2.0

        for track_index, track in enumerate(ordered_tracks):
            if not face_matches_person(
                face_box,
                track.box,
                upper_body_ratio,
            ):
                continue

            x1, y1, x2, y2 = track.box
            width = max(1.0, float(x2 - x1))
            height = max(1.0, float(y2 - y1))
            expected_x = (x1 + x2) / 2.0
            expected_y = y1 + height * upper_body_ratio * 0.5
            dx = (face_center_x - expected_x) / width
            dy = (
                (face_center_y - expected_y)
                / (height * upper_body_ratio)
            )
            distance = dx * dx + dy * dy
            eligible_pairs.append((face_index, track_index, distance))

    if not eligible_pairs:
        return {}

    face_count = len(face_boxes)
    track_count = len(ordered_tracks)
    source_node = 0
    face_offset = 1
    track_offset = face_offset + face_count
    sink_node = track_offset + track_count
    graph: list[list[_FlowEdge]] = [
        [] for _ in range(sink_node + 1)
    ]

    for face_index in range(face_count):
        _add_flow_edge(
            graph,
            source_node,
            face_offset + face_index,
            1,
            0.0,
        )
    for track_index in range(track_count):
        _add_flow_edge(
            graph,
            track_offset + track_index,
            sink_node,
            1,
            0.0,
        )

    maximum_priority = max(priorities, default=0.0)
    priority_weight = float(min(face_count, track_count) + 1)
    pair_edges: dict[tuple[int, int], _FlowEdge] = {}
    for face_index, track_index, distance in eligible_pairs:
        priority_penalty = maximum_priority - priorities[face_index]
        deterministic_tie_break = (
            (face_index + 1) * (track_index + 1) * 1e-12
        )
        edge_cost = (
            priority_penalty * priority_weight
            + distance
            + deterministic_tie_break
        )
        if not math.isfinite(edge_cost):
            raise ValueError("face priority tiers are too large.")
        pair_edges[(face_index, track_index)] = _add_flow_edge(
            graph,
            face_offset + face_index,
            track_offset + track_index,
            1,
            edge_cost,
        )

    node_count = len(graph)
    while True:
        distances = [float("inf")] * node_count
        previous: list[tuple[int, int] | None] = [None] * node_count
        distances[source_node] = 0.0

        for _ in range(node_count - 1):
            changed = False
            for node, edges in enumerate(graph):
                if not math.isfinite(distances[node]):
                    continue
                for edge_index, edge in enumerate(edges):
                    if edge.capacity <= 0:
                        continue
                    candidate_distance = distances[node] + edge.cost
                    if candidate_distance < distances[edge.to] - 1e-12:
                        distances[edge.to] = candidate_distance
                        previous[edge.to] = (node, edge_index)
                        changed = True
            if not changed:
                break

        if previous[sink_node] is None:
            break

        node = sink_node
        while node != source_node:
            step = previous[node]
            if step is None:  # Defensive; the sink path is complete here.
                raise RuntimeError("Incomplete face-assignment path.")
            previous_node, edge_index = step
            edge = graph[previous_node][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse_index].capacity += 1
            node = previous_node

    assignments: dict[int, HasPersonBox] = {}
    for (face_index, track_index), edge in pair_edges.items():
        if edge.capacity == 0:
            assignments[face_index] = ordered_tracks[track_index]
    return assignments
