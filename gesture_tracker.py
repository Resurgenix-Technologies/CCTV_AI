from collections import deque
import numpy as np


class GestureActivityTracker:
    """Tracks wrist movement relative to shoulders to detect gesturing activity and handshake interactions."""

    def __init__(self, history_len: int = 8, movement_std_threshold: float = 6.0):
        self.history_len = history_len
        self.movement_std_threshold = movement_std_threshold
        self._min_samples = max(4, history_len // 2)
        self._history: dict[tuple, deque] = {}
        self._last_keypoints: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def update(self, key: tuple, relative_wrist_offset: float | None) -> None:
        if relative_wrist_offset is None:
            return
        history = self._history.get(key)
        if history is None:
            history = deque(maxlen=self.history_len)
            self._history[key] = history
        history.append(relative_wrist_offset)

    def update_keypoints(self, key: tuple, keypoints_xy: np.ndarray, keypoints_conf: np.ndarray) -> None:
        """Stores recent pose keypoints for handshake and spatial interaction analysis."""
        self._last_keypoints[key] = (keypoints_xy, keypoints_conf)

    def has_enough_data(self, key: tuple) -> bool:
        history = self._history.get(key)
        return history is not None and len(history) >= self._min_samples

    def is_actively_gesturing(self, key: tuple) -> bool:
        history = self._history.get(key)
        if history is None or len(history) < self._min_samples:
            return False
        return float(np.std(history)) >= self.movement_std_threshold

    def check_handshake(self, key_a: tuple, key_b: tuple, avg_box_height: float) -> bool:
        """Evaluates wrist keypoints proximity between two individuals to detect handshakes."""
        kp_a = self._last_keypoints.get(key_a)
        kp_b = self._last_keypoints.get(key_b)
        if kp_a is None or kp_b is None:
            return False
        return detect_handshake_between_keypoints(kp_a[0], kp_a[1], kp_b[0], kp_b[1], avg_box_height)

    def forget(self, key: tuple) -> None:
        self._history.pop(key, None)
        self._last_keypoints.pop(key, None)


def extract_gesture_signal(keypoints_xy, keypoints_conf) -> float | None:
    """Extracts relative wrist distance from shoulders as a gesture indicator from pose keypoints."""
    try:
        if keypoints_conf[5] < 0.3 or keypoints_conf[6] < 0.3:
            return None

        shoulder_mid = (keypoints_xy[5] + keypoints_xy[6]) / 2.0
        offsets = []
        if keypoints_conf[9] >= 0.3:
            offsets.append(float(np.linalg.norm(keypoints_xy[9] - shoulder_mid)))
        if keypoints_conf[10] >= 0.3:
            offsets.append(float(np.linalg.norm(keypoints_xy[10] - shoulder_mid)))

        if not offsets:
            return None
        return float(np.mean(offsets))
    except (IndexError, TypeError):
        return None


def extract_wrist_keypoints(keypoints_xy: np.ndarray, keypoints_conf: np.ndarray) -> list[np.ndarray]:
    """Returns valid (x, y) wrist coordinates (keypoints 9 and 10) for a person."""
    wrists = []
    try:
        for idx in (9, 10):
            if keypoints_conf[idx] >= 0.30:
                wrists.append(np.asarray(keypoints_xy[idx], dtype=np.float32))
    except (IndexError, TypeError):
        pass
    return wrists


def detect_handshake_between_keypoints(
    keypoints_a_xy: np.ndarray,
    keypoints_a_conf: np.ndarray,
    keypoints_b_xy: np.ndarray,
    keypoints_b_conf: np.ndarray,
    avg_box_height: float,
) -> bool:
    """
    Evaluates Euclidean distance between wrist keypoints of Person A and Person B.
    Returns True if wrists are in close proximity indicating a handshake interaction.
    """
    wrists_a = extract_wrist_keypoints(keypoints_a_xy, keypoints_a_conf)
    wrists_b = extract_wrist_keypoints(keypoints_b_xy, keypoints_b_conf)

    if not wrists_a or not wrists_b:
        return False

    min_dist = float("inf")
    for wa in wrists_a:
        for wb in wrists_b:
            dist = float(np.linalg.norm(wa - wb))
            if dist < min_dist:
                min_dist = dist

    # Dynamic wrist distance threshold scaled by body height
    threshold = min(65.0, max(25.0, avg_box_height * 0.22))
    return min_dist <= threshold
