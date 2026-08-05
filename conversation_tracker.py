from collections import deque
import time
import threading
import cv2

# Thresholds & Parameters
DISTANCE_THRESHOLD_RATIO = 1.4
MIN_DISTANCE_THRESHOLD = 100
MAX_DISTANCE_THRESHOLD = 450

TALKING_DURATION = 5
GRACE_PERIOD_SECONDS = 4

MOVEMENT_HISTORY_LEN = 6
STATIONARY_MAX_AVG_MOVEMENT = 18


def get_display_name(cam_name: str, identity_manager, track_id: int) -> str:
    """Returns candidate name if identity confirmed, else fallback identifier."""
    progress = identity_manager.vote_progress(track_id)
    if progress.status == "CONFIRMED":
        return progress.candidate_name
    return f"Unknown-{cam_name}-{track_id}"


def stable_identity_key(cam_name: str, track_id: int, identity_manager):
    """Returns a stable tuple key representing the confirmed person name or camera track ID."""
    progress = identity_manager.vote_progress(track_id)
    if progress.status == "CONFIRMED":
        if progress.candidate_person_id is not None:
            return ("NAME", progress.candidate_person_id)
        return ("NAME", progress.candidate_name)
    return ("TRACK", cam_name, track_id)


def resolve_display_name(key: tuple, identity_managers: dict) -> str:
    """Resolves stable identity key into human-readable display string."""
    if key[0] == "NAME":
        return key[1]
    _, cam_name, track_id = key
    return get_display_name(cam_name, identity_managers[cam_name], track_id)


class ConversationTracker:
    """Tracks movement, pair proximity, interaction signals (mouth & gesture), conversation logging, and UI overlays."""

    def __init__(self):
        self.lock = threading.Lock()
        self.pair_proximity_start: dict[tuple, float] = {}
        self.pair_last_close_time: dict[tuple, float] = {}
        self.talking_pairs: set[tuple] = set()
        self.track_movement_history: dict[tuple[str, int], deque] = {}

    def update_movement_and_check_stationary(self, cam_name: str, track_id: int, cx: int, cy: int) -> bool:
        """Updates movement history for a track and returns True if average displacement is within stationary threshold."""
        key = (cam_name, track_id)
        history = self.track_movement_history.get(key)
        if history is None:
            history = deque(maxlen=MOVEMENT_HISTORY_LEN)
            self.track_movement_history[key] = history
        history.append((cx, cy))

        if len(history) < 2:
            return True

        total_movement = 0.0
        for i in range(1, len(history)):
            px, py = history[i - 1]
            qx, qy = history[i]
            total_movement += ((qx - px) ** 2 + (qy - py) ** 2) ** 0.5

        avg_movement = total_movement / (len(history) - 1)
        return avg_movement <= STATIONARY_MAX_AVG_MOVEMENT

    def forget_track(self, cam_name: str, track_id: int):
        """Cleans up movement history for an expired track."""
        self.track_movement_history.pop((cam_name, track_id), None)

    def evaluate_pair_proximity(
        self,
        cam_name: str,
        cached_boxes: list,
        identity_manager,
        mouth_tracker,
        gesture_tracker,
        current_time: float,
    ):
        """Evaluates pairwise proximity & multi-modal engagement for all tracked individuals on a camera feed."""
        for i in range(len(cached_boxes)):
            for j in range(i + 1, len(cached_boxes)):
                (x1i, y1i, x2i, y2i, id1, cx1, cy1, h1, stationary1) = cached_boxes[i]
                (x1j, y1j, x2j, y2j, id2, cx2, cy2, h2, stationary2) = cached_boxes[j]

                if id1 == -1 or id2 == -1:
                    continue

                distance = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                avg_box_height = (h1 + h2) / 2
                distance_threshold = min(
                    MAX_DISTANCE_THRESHOLD,
                    max(MIN_DISTANCE_THRESHOLD, avg_box_height * DISTANCE_THRESHOLD_RATIO),
                )

                if distance >= distance_threshold:
                    continue

                # Wait for both identities to be fully resolved (confirmed
                # AI team member, or confirmed-unknown visitor) before
                # tracking this pair at all. stable_identity_key() switches
                # from a temporary TRACK-based key to a permanent NAME-based
                # key the moment a track gets confirmed, and that key is
                # recomputed every frame -- so starting the timer before
                # resolution risks the key changing mid-conversation and
                # silently resetting/orphaning the proximity timer, which
                # is why conversations were never reaching TALKING_DURATION.
                progress1 = identity_manager.vote_progress(id1)
                progress2 = identity_manager.vote_progress(id2)
                if progress1.status != "CONFIRMED" or progress2.status != "CONFIRMED":
                    continue

                key1 = stable_identity_key(cam_name, id1, identity_manager)
                key2 = stable_identity_key(cam_name, id2, identity_manager)
                pair_key = tuple(sorted([key1, key2]))

                with self.lock:
                    is_new_pair = pair_key not in self.pair_proximity_start

                    mouth_ready = (
                        mouth_tracker.has_enough_data((cam_name, id1))
                        or mouth_tracker.has_enough_data((cam_name, id2))
                    )
                    gesture_ready = (
                        gesture_tracker.has_enough_data((cam_name, id1))
                        or gesture_tracker.has_enough_data((cam_name, id2))
                    )

                    if mouth_ready or gesture_ready:
                        mouth_active = (
                            mouth_tracker.is_actively_talking((cam_name, id1))
                            or mouth_tracker.is_actively_talking((cam_name, id2))
                        )
                        gesture_active = (
                            gesture_tracker.is_actively_gesturing((cam_name, id1))
                            or gesture_tracker.is_actively_gesturing((cam_name, id2))
                        )
                        engagement_ok = mouth_active or gesture_active
                    else:
                        engagement_ok = True

                    if is_new_pair and not (stationary1 and stationary2 and engagement_ok):
                        continue

                    if is_new_pair:
                        self.pair_proximity_start[pair_key] = current_time
                    self.pair_last_close_time[pair_key] = current_time

    def finalize_global_conversations(self, conversation_logger, identity_managers: dict):
        """Finalizes conversation sessions and logs completed dialogues to CSV."""
        current_time = time.time()

        with self.lock:
            for pair_key, start_time in self.pair_proximity_start.items():
                if pair_key in self.talking_pairs:
                    continue
                last_close = self.pair_last_close_time.get(pair_key, current_time)
                if (last_close - start_time) >= TALKING_DURATION:
                    self.talking_pairs.add(pair_key)

            pairs_to_end = []
            for pair_key, last_close in self.pair_last_close_time.items():
                if (current_time - last_close) > GRACE_PERIOD_SECONDS:
                    pairs_to_end.append(pair_key)

            for pair_key in pairs_to_end:
                if pair_key in self.talking_pairs:
                    key_a, key_b = pair_key
                    name_a = resolve_display_name(key_a, identity_managers)
                    name_b = resolve_display_name(key_b, identity_managers)
                    start_time = self.pair_proximity_start.get(pair_key, self.pair_last_close_time[pair_key])
                    end_time = self.pair_last_close_time[pair_key]
                    conversation_logger.log_conversation(
                        camera="ROOM",
                        person_a=name_a,
                        person_b=name_b,
                        start_time=start_time,
                        end_time=end_time,
                    )
                self.pair_proximity_start.pop(pair_key, None)
                self.pair_last_close_time.pop(pair_key, None)
                self.talking_pairs.discard(pair_key)

    def draw_talking_overlays(self, cam_name: str, frame: cv2.Mat, cached_boxes: list, identity_manager):
        """Draws visual 'Talking' indicators and red bounding lines between active conversation participants."""
        lookup = {}
        for entry in cached_boxes:
            track_id = entry[4]
            if track_id == -1:
                continue
            key = stable_identity_key(cam_name, track_id, identity_manager)
            lookup[key] = (entry[5], entry[6])

        for pair_key in self.talking_pairs:
            key_a, key_b = pair_key
            if key_a in lookup and key_b in lookup:
                cx1, cy1 = lookup[key_a]
                cx2, cy2 = lookup[key_b]
                cv2.line(frame, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
                mx, my = (cx1 + cx2) // 2, (cy1 + cy2) // 2
                cv2.putText(
                    frame,
                    "Talking",
                    (mx, my),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

    def shutdown_active_conversations(self, conversation_logger, identity_managers: dict):
        """Logs any remaining active conversations upon application shutdown."""
        shutdown_time = time.time()
        for pair_key in list(self.talking_pairs):
            key_a, key_b = pair_key
            name_a = resolve_display_name(key_a, identity_managers)
            name_b = resolve_display_name(key_b, identity_managers)
            start_time = self.pair_proximity_start.get(pair_key, shutdown_time)
            end_time = self.pair_last_close_time.get(pair_key, shutdown_time)
            conversation_logger.log_conversation(
                camera="ROOM",
                person_a=name_a,
                person_b=name_b,
                start_time=start_time,
                end_time=end_time,
            )