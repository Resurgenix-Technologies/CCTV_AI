from collections import deque
import numpy as np


class GestureActivityTracker:
    """Tracks wrist movement relative to shoulders to detect gesturing activity."""

    def __init__(self, history_len: int = 8, movement_std_threshold: float = 6.0):
        self.history_len = history_len
        self.movement_std_threshold = movement_std_threshold
        self._min_samples = max(4, history_len // 2)
        self._history: dict[tuple, deque] = {}

    def update(self, key: tuple, relative_wrist_offset: float | None) -> None:
        if relative_wrist_offset is None:
            return
        history = self._history.get(key)
        if history is None:
            history = deque(maxlen=self.history_len)
            self._history[key] = history
        history.append(relative_wrist_offset)

    def has_enough_data(self, key: tuple) -> bool:
        history = self._history.get(key)
        return history is not None and len(history) >= self._min_samples

    def is_actively_gesturing(self, key: tuple) -> bool:
        history = self._history.get(key)
        if history is None or len(history) < self._min_samples:
            return False
        return float(np.std(history)) >= self.movement_std_threshold

    def forget(self, key: tuple) -> None:
        self._history.pop(key, None)


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
