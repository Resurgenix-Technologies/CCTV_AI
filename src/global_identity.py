"""High-precision, model-agnostic cross-camera tracklet association.

The associator consumes *completed* tracklet evidence.  An upstream component
must extract and normalize body embeddings, synchronize camera timestamps,
and decide whether an optional person anchor is trusted.  This module does not
run a ReID model, persist embeddings or decisions, or infer a durable person
identity.  Its ``subject_id`` values are pseudonymous journey hypotheses only.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Callable, Literal, Mapping, Sequence
from urllib.parse import quote


DecisionOutcome = Literal["MATCHED_SUBJECT", "NEW_SUBJECT"]
DecisionReason = Literal[
    "CONFIDENT_MATCH",
    "NO_CANDIDATES",
    "LOW_QUALITY",
    "NO_ELIGIBLE_CANDIDATE",
    "BELOW_SIMILARITY_THRESHOLD",
    "AMBIGUOUS_BEST_MATCH",
]
RejectionReason = Literal[
    "INCOMING_QUALITY_BELOW_MINIMUM",
    "CANDIDATE_QUALITY_BELOW_MINIMUM",
    "ANCHOR_CONFLICT",
    "IMPOSSIBLE_OVERLAP",
    "NOT_A_PREDECESSOR",
    "TRANSITION_NOT_ALLOWED",
    "TRAVEL_TIME_OUTSIDE_WINDOW",
    "BELOW_SIMILARITY_THRESHOLD",
]


def _nonempty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        converted = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not isfinite(converted):
        raise ValueError(f"{field_name} must be a finite number")
    return converted


@dataclass(frozen=True, slots=True)
class TrackletId:
    """A local tracker ID namespaced by camera session and camera."""

    camera_session_id: str
    camera_id: str
    local_track_id: str | int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "camera_session_id",
            _nonempty_text(self.camera_session_id, "camera_session_id"),
        )
        object.__setattr__(
            self,
            "camera_id",
            _nonempty_text(self.camera_id, "camera_id"),
        )
        if isinstance(self.local_track_id, bool):
            raise ValueError("local_track_id must be a string or integer")
        local_id = str(self.local_track_id).strip()
        if not local_id:
            raise ValueError("local_track_id must be a string or integer")
        object.__setattr__(self, "local_track_id", local_id)

    @property
    def namespaced_id(self) -> str:
        """Return an unambiguous, displayable form of the composite ID."""

        parts = (
            self.camera_session_id,
            self.camera_id,
            str(self.local_track_id),
        )
        return "/".join(quote(part, safe="") for part in parts)


@dataclass(frozen=True, slots=True)
class CompletedTracklet:
    """Evidence emitted after a local camera track has completed.

    Times are finite synchronized UTC epoch seconds.  ``body_embedding`` must
    already be L2-normalized by the external ReID model pipeline.
    """

    tracklet_id: TrackletId
    started_at: float
    ended_at: float
    body_embedding: tuple[float, ...]
    quality: float
    trusted_person_anchor: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tracklet_id, TrackletId):
            raise ValueError("tracklet_id must be a TrackletId")

        started = _finite_float(self.started_at, "started_at")
        ended = _finite_float(self.ended_at, "ended_at")
        if ended <= started:
            raise ValueError("ended_at must be later than started_at")
        object.__setattr__(self, "started_at", started)
        object.__setattr__(self, "ended_at", ended)

        quality = _finite_float(self.quality, "quality")
        if not 0.0 <= quality <= 1.0:
            raise ValueError("quality must be between 0 and 1")
        object.__setattr__(self, "quality", quality)

        try:
            embedding = tuple(
                _finite_float(value, "body_embedding value")
                for value in self.body_embedding
            )
        except TypeError as exc:
            raise ValueError("body_embedding must be a finite vector") from exc
        if not embedding:
            raise ValueError("body_embedding cannot be empty")
        norm = sqrt(sum(value * value for value in embedding))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError("body_embedding must be L2-normalized")
        object.__setattr__(self, "body_embedding", embedding)

        if self.trusted_person_anchor is not None:
            object.__setattr__(
                self,
                "trusted_person_anchor",
                _nonempty_text(
                    self.trusted_person_anchor,
                    "trusted_person_anchor",
                ),
            )

    @property
    def camera_id(self) -> str:
        return self.tracklet_id.camera_id


@dataclass(frozen=True, slots=True)
class TravelWindow:
    """Inclusive allowed travel time for one directed camera edge."""

    minimum_seconds: float
    maximum_seconds: float

    def __post_init__(self) -> None:
        minimum = _finite_float(self.minimum_seconds, "minimum_seconds")
        maximum = _finite_float(self.maximum_seconds, "maximum_seconds")
        if minimum < 0.0:
            raise ValueError("minimum_seconds cannot be negative")
        if maximum < minimum:
            raise ValueError(
                "maximum_seconds must be at least minimum_seconds"
            )
        object.__setattr__(self, "minimum_seconds", minimum)
        object.__setattr__(self, "maximum_seconds", maximum)


@dataclass(frozen=True, slots=True)
class AssociationConfig:
    """Conservative matching thresholds for the association engine."""

    similarity_threshold: float = 0.85
    ambiguity_margin: float = 0.05
    minimum_quality: float = 0.5

    def __post_init__(self) -> None:
        threshold = _finite_float(
            self.similarity_threshold,
            "similarity_threshold",
        )
        margin = _finite_float(self.ambiguity_margin, "ambiguity_margin")
        quality = _finite_float(self.minimum_quality, "minimum_quality")
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1 and 1")
        if not 0.0 <= margin <= 2.0:
            raise ValueError("ambiguity_margin must be between 0 and 2")
        if not 0.0 <= quality <= 1.0:
            raise ValueError("minimum_quality must be between 0 and 1")
        object.__setattr__(self, "similarity_threshold", threshold)
        object.__setattr__(self, "ambiguity_margin", margin)
        object.__setattr__(self, "minimum_quality", quality)


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """Auditable score for one previous tracklet candidate."""

    subject_id: str
    candidate_tracklet_id: TrackletId
    similarity: float
    travel_seconds: float | None
    candidate_quality: float
    subject_trusted_person_anchor: str | None
    hard_constraints_passed: bool
    above_similarity_threshold: bool
    rejection_reasons: tuple[RejectionReason, ...]

    @property
    def eligible(self) -> bool:
        return (
            self.hard_constraints_passed
            and self.above_similarity_threshold
        )


@dataclass(frozen=True, slots=True)
class AssociationDecision:
    """One replayable, auditable association result."""

    tracklet_id: TrackletId
    subject_id: str
    outcome: DecisionOutcome
    reason: DecisionReason
    selected_candidate: CandidateScore | None
    candidate_scores: tuple[CandidateScore, ...]
    subject_trusted_person_anchor: str | None
    similarity_threshold: float
    ambiguity_margin: float


@dataclass(slots=True)
class _SubjectState:
    subject_id: str
    trusted_person_anchor: str | None
    tracklets: list[CompletedTracklet]


def cosine_similarity(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    """Return cosine similarity, validating finite, equal-sized vectors."""

    if len(first) != len(second) or not first:
        raise ValueError("embeddings must be non-empty and equal-sized")
    first_values = tuple(
        _finite_float(value, "embedding value") for value in first
    )
    second_values = tuple(
        _finite_float(value, "embedding value") for value in second
    )
    first_norm = sqrt(sum(value * value for value in first_values))
    second_norm = sqrt(sum(value * value for value in second_values))
    if first_norm == 0.0 or second_norm == 0.0:
        raise ValueError("embeddings cannot have zero length")
    similarity = sum(
        left * right for left, right in zip(first_values, second_values)
    ) / (first_norm * second_norm)
    return max(-1.0, min(1.0, similarity))


class GlobalIdentityAssociator:
    """Associate completed tracklets into pseudonymous subject journeys.

    Quality is used as a hard evidence gate; the match score itself remains
    cosine similarity so every decision is straightforward to audit.
    """

    def __init__(
        self,
        topology: Mapping[tuple[str, str], TravelWindow],
        *,
        config: AssociationConfig | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.config = config or AssociationConfig()
        self._id_factory = id_factory or (
            lambda: f"subject-{uuid.uuid4()}"
        )
        self._topology: dict[tuple[str, str], TravelWindow] = {}
        for edge, window in topology.items():
            if not isinstance(edge, tuple) or len(edge) != 2:
                raise ValueError(
                    "topology keys must be (source_camera, target_camera)"
                )
            source = _nonempty_text(edge[0], "source_camera")
            target = _nonempty_text(edge[1], "target_camera")
            if not isinstance(window, TravelWindow):
                raise ValueError("topology values must be TravelWindow objects")
            self._topology[(source, target)] = window

        self._subjects: dict[str, _SubjectState] = {}
        self._evidence_by_id: dict[TrackletId, CompletedTracklet] = {}
        self._decisions_by_id: dict[TrackletId, AssociationDecision] = {}
        self._embedding_dimension: int | None = None

    def associate(self, evidence: CompletedTracklet) -> AssociationDecision:
        """Associate one completed tracklet, failing closed when uncertain."""

        if not isinstance(evidence, CompletedTracklet):
            raise ValueError("evidence must be a CompletedTracklet")

        previous_evidence = self._evidence_by_id.get(evidence.tracklet_id)
        if previous_evidence is not None:
            if previous_evidence == evidence:
                return self._decisions_by_id[evidence.tracklet_id]
            raise ValueError(
                "Conflicting replay for tracklet "
                f"{evidence.tracklet_id.namespaced_id}"
            )

        dimension = len(evidence.body_embedding)
        if self._embedding_dimension is None:
            self._embedding_dimension = dimension
        elif dimension != self._embedding_dimension:
            raise ValueError(
                "body_embedding dimension differs from previous tracklets"
            )

        scores = self._score_candidates(evidence)
        selected, reason = self._select_candidate(evidence, scores)

        if selected is None:
            subject_id = self._new_subject_id()
            subject = _SubjectState(
                subject_id=subject_id,
                trusted_person_anchor=evidence.trusted_person_anchor,
                tracklets=[evidence],
            )
            self._subjects[subject_id] = subject
            outcome: DecisionOutcome = "NEW_SUBJECT"
        else:
            subject = self._subjects[selected.subject_id]
            if (
                subject.trusted_person_anchor is None
                and evidence.trusted_person_anchor is not None
            ):
                subject.trusted_person_anchor = evidence.trusted_person_anchor
            subject.tracklets.append(evidence)
            subject_id = subject.subject_id
            outcome = "MATCHED_SUBJECT"

        decision = AssociationDecision(
            tracklet_id=evidence.tracklet_id,
            subject_id=subject_id,
            outcome=outcome,
            reason=reason,
            selected_candidate=selected,
            candidate_scores=scores,
            subject_trusted_person_anchor=subject.trusted_person_anchor,
            similarity_threshold=self.config.similarity_threshold,
            ambiguity_margin=self.config.ambiguity_margin,
        )
        self._evidence_by_id[evidence.tracklet_id] = evidence
        self._decisions_by_id[evidence.tracklet_id] = decision
        return decision

    def _score_candidates(
        self,
        incoming: CompletedTracklet,
    ) -> tuple[CandidateScore, ...]:
        scores: list[CandidateScore] = []
        for subject in self._subjects.values():
            subject_overlaps = any(
                self._overlaps(previous, incoming)
                for previous in subject.tracklets
            )
            anchor_conflict = (
                incoming.trusted_person_anchor is not None
                and subject.trusted_person_anchor is not None
                and incoming.trusted_person_anchor
                != subject.trusted_person_anchor
            )

            for previous in subject.tracklets:
                reasons: list[RejectionReason] = []
                if incoming.quality < self.config.minimum_quality:
                    reasons.append("INCOMING_QUALITY_BELOW_MINIMUM")
                if previous.quality < self.config.minimum_quality:
                    reasons.append("CANDIDATE_QUALITY_BELOW_MINIMUM")
                if anchor_conflict:
                    reasons.append("ANCHOR_CONFLICT")
                if subject_overlaps:
                    reasons.append("IMPOSSIBLE_OVERLAP")

                travel_seconds: float | None = None
                if previous.ended_at > incoming.started_at:
                    if not self._overlaps(previous, incoming):
                        reasons.append("NOT_A_PREDECESSOR")
                else:
                    travel_seconds = incoming.started_at - previous.ended_at
                    window = self._topology.get(
                        (previous.camera_id, incoming.camera_id)
                    )
                    if window is None:
                        reasons.append("TRANSITION_NOT_ALLOWED")
                    elif not (
                        window.minimum_seconds
                        <= travel_seconds
                        <= window.maximum_seconds
                    ):
                        reasons.append("TRAVEL_TIME_OUTSIDE_WINDOW")

                similarity = cosine_similarity(
                    previous.body_embedding,
                    incoming.body_embedding,
                )
                hard_constraints_passed = not reasons
                above_threshold = (
                    similarity >= self.config.similarity_threshold
                )
                if not above_threshold:
                    reasons.append("BELOW_SIMILARITY_THRESHOLD")

                scores.append(
                    CandidateScore(
                        subject_id=subject.subject_id,
                        candidate_tracklet_id=previous.tracklet_id,
                        similarity=similarity,
                        travel_seconds=travel_seconds,
                        candidate_quality=previous.quality,
                        subject_trusted_person_anchor=(
                            subject.trusted_person_anchor
                        ),
                        hard_constraints_passed=hard_constraints_passed,
                        above_similarity_threshold=above_threshold,
                        rejection_reasons=tuple(reasons),
                    )
                )

        scores.sort(
            key=lambda score: (
                -score.similarity,
                score.subject_id,
                score.candidate_tracklet_id.namespaced_id,
            )
        )
        return tuple(scores)

    def _select_candidate(
        self,
        incoming: CompletedTracklet,
        scores: tuple[CandidateScore, ...],
    ) -> tuple[CandidateScore | None, DecisionReason]:
        if incoming.quality < self.config.minimum_quality:
            return None, "LOW_QUALITY"
        if not scores:
            return None, "NO_CANDIDATES"

        # Compare subjects, not individual tracklets.  Multiple fragments from
        # the same subject must not create a false ambiguity against itself.
        best_by_subject: dict[str, CandidateScore] = {}
        for score in scores:
            if not score.hard_constraints_passed:
                continue
            current = best_by_subject.get(score.subject_id)
            if current is None or score.similarity > current.similarity:
                best_by_subject[score.subject_id] = score

        ranked = sorted(
            best_by_subject.values(),
            key=lambda score: (-score.similarity, score.subject_id),
        )
        if not ranked:
            return None, "NO_ELIGIBLE_CANDIDATE"

        best = ranked[0]
        if best.similarity < self.config.similarity_threshold:
            return None, "BELOW_SIMILARITY_THRESHOLD"

        if len(ranked) > 1:
            margin = best.similarity - ranked[1].similarity
            if margin < self.config.ambiguity_margin:
                return None, "AMBIGUOUS_BEST_MATCH"

        return best, "CONFIDENT_MATCH"

    def _new_subject_id(self) -> str:
        subject_id = _nonempty_text(self._id_factory(), "subject_id")
        if subject_id in self._subjects:
            raise ValueError(f"id_factory returned duplicate ID: {subject_id}")
        return subject_id

    @staticmethod
    def _overlaps(
        first: CompletedTracklet,
        second: CompletedTracklet,
    ) -> bool:
        return (
            first.started_at < second.ended_at
            and second.started_at < first.ended_at
        )
