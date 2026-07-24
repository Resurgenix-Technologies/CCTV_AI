"""Track-level identity voting, confirmation and retention."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Literal

from src.recognizer import RecognitionResult


IdentityStatus = Literal["UNKNOWN", "PENDING", "CONFIRMED"]


@dataclass(frozen=True)
class Vote:
    """One recognition result in a track's rolling vote window."""

    person_id: str | None
    full_name: str
    similarity: float
    matched: bool


@dataclass(frozen=True)
class VoteProgress:
    """Current leading candidate and confirmation progress."""

    status: IdentityStatus
    candidate_person_id: str | None
    candidate_name: str
    votes: int
    required_votes: int
    window_size: int


@dataclass
class TrackIdentity:
    """Identity state retained for one temporary ByteTrack ID."""

    track_id: int
    confirmed: bool = False
    person_id: str | None = None
    full_name: str = "UNKNOWN"
    similarity: float = 0.0
    last_face_frame: int | None = None
    last_seen_frame: int = 0
    votes: deque[Vote] = field(default_factory=deque)


class IdentityManager:
    """
    Require repeated face matches before confirming a track.

    UNKNOWN results are included in the rolling result window. This
    prevents weak known matches from accumulating indefinitely across
    many unrelated recognition attempts.
    """

    def __init__(
        self,
        confirmation_votes: int,
        vote_window: int,
        ttl_frames: int,
    ) -> None:
        if confirmation_votes < 1:
            raise ValueError("confirmation_votes must be positive.")
        if vote_window < 1:
            raise ValueError("vote_window must be positive.")
        if confirmation_votes > vote_window:
            raise ValueError(
                "confirmation_votes cannot exceed vote_window."
            )
        if ttl_frames < 1:
            raise ValueError("ttl_frames must be positive.")

        self.confirmation_votes = confirmation_votes
        self.vote_window = vote_window
        self.ttl_frames = ttl_frames
        self._tracks: dict[int, TrackIdentity] = {}

    def _new_state(
        self,
        track_id: int,
        frame_index: int = 0,
    ) -> TrackIdentity:
        return TrackIdentity(
            track_id=track_id,
            last_seen_frame=frame_index,
            votes=deque(maxlen=self.vote_window),
        )

    def mark_seen(
        self,
        track_ids: set[int],
        frame_index: int,
    ) -> list[TrackIdentity]:
        """
        Mark active tracker IDs and return states created this frame.
        """

        started: list[TrackIdentity] = []

        for track_id in track_ids:
            state = self._tracks.get(track_id)

            if state is None:
                state = self._new_state(track_id, frame_index)
                self._tracks[track_id] = state
                started.append(state)

            state.last_seen_frame = frame_index

        return started

    def observe(
        self,
        track_id: int,
        result: RecognitionResult,
        frame_index: int,
    ) -> TrackIdentity:
        """
        Add one face-recognition result to a track's vote window.
        """

        state = self._tracks.get(track_id)

        if state is None:
            state = self._new_state(track_id, frame_index)
            self._tracks[track_id] = state

        state.last_seen_frame = frame_index
        state.last_face_frame = frame_index

        # Once confirmed, retain the global identity. A fresh matching
        # result can update its displayed similarity; a weak/conflicting
        # result must not erase the confirmed identity or score.
        if state.confirmed:
            if (
                result.matched
                and result.person_id == state.person_id
            ):
                state.similarity = result.similarity
            return state

        vote = Vote(
            person_id=result.person_id if result.matched else None,
            full_name=(
                result.full_name
                if result.matched
                else "UNKNOWN"
            ),
            similarity=result.similarity,
            matched=result.matched,
        )
        state.votes.append(vote)
        state.similarity = result.similarity

        known_votes = [
            item
            for item in state.votes
            if item.matched and item.person_id
        ]

        if not known_votes:
            return state

        counts = Counter(
            item.person_id
            for item in known_votes
            if item.person_id is not None
        )

        person_id, count = counts.most_common(1)[0]

        if count >= self.confirmation_votes:
            matching_votes = [
                item
                for item in known_votes
                if item.person_id == person_id
            ]

            state.confirmed = True
            state.person_id = person_id
            state.full_name = matching_votes[-1].full_name
            state.similarity = sum(
                item.similarity
                for item in matching_votes
            ) / len(matching_votes)

        return state

    def get(self, track_id: int) -> TrackIdentity:
        """Return an existing state or create an unseen one."""

        state = self._tracks.get(track_id)

        if state is None:
            state = self._new_state(track_id)
            self._tracks[track_id] = state

        return state

    def vote_progress(
        self,
        track_id: int,
    ) -> VoteProgress:
        """Return the current display state for a track."""

        state = self.get(track_id)

        if state.confirmed:
            return VoteProgress(
                status="CONFIRMED",
                candidate_person_id=state.person_id,
                candidate_name=state.full_name,
                votes=self.confirmation_votes,
                required_votes=self.confirmation_votes,
                window_size=self.vote_window,
            )

        known_votes = [
            vote
            for vote in state.votes
            if vote.matched and vote.person_id
        ]

        if not known_votes:
            return VoteProgress(
                status="UNKNOWN",
                candidate_person_id=None,
                candidate_name="UNKNOWN",
                votes=0,
                required_votes=self.confirmation_votes,
                window_size=self.vote_window,
            )

        counts = Counter(
            vote.person_id
            for vote in known_votes
            if vote.person_id is not None
        )
        candidate_person_id, count = counts.most_common(1)[0]

        candidate_votes = [
            vote
            for vote in known_votes
            if vote.person_id == candidate_person_id
        ]

        return VoteProgress(
            status="PENDING",
            candidate_person_id=candidate_person_id,
            candidate_name=candidate_votes[-1].full_name,
            votes=count,
            required_votes=self.confirmation_votes,
            window_size=self.vote_window,
        )

    def expire(
        self,
        frame_index: int,
    ) -> list[TrackIdentity]:
        """
        Remove stale track states and return the expired records.
        """

        stale_ids = [
            track_id
            for track_id, state in self._tracks.items()
            if (
                frame_index - state.last_seen_frame
                > self.ttl_frames
            )
        ]

        expired = [
            self._tracks.pop(track_id)
            for track_id in stale_ids
        ]

        return expired

    def close_all(self) -> list[TrackIdentity]:
        """
        Remove and return every active track.

        Camera workers must call this when a stream ends or the process is
        shutting down.  Waiting for TTL expiry cannot close tracks that are
        still visible in the final frame.
        """

        active = [
            self._tracks[track_id]
            for track_id in sorted(self._tracks)
        ]
        self._tracks.clear()
        return active
