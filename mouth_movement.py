"""Mouth-movement heuristic: is this face actively moving its mouth (a
lightweight proxy for "is talking"), using MediaPipe FaceMesh mouth
landmarks. No training needed - just geometry on ready-made landmarks.

Degrades gracefully: if mediapipe isn't installed, or a face crop is too
small/blurry for FaceMesh to find landmarks, callers should treat the
mouth signal as "not available" rather than "not talking" - see
MouthMovementTracker.has_enough_data().
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

try:
    import mediapipe as mp
    _face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
    )
    MEDIAPIPE_AVAILABLE = True
except Exception as exc:
    _face_mesh = None
    MEDIAPIPE_AVAILABLE = False
    print(
        f"mediapipe's FaceMesh API isn't available ({type(exc).__name__}: {exc}). "
        "Mouth-movement talking signal disabled - falling back to "
        "proximity+stationary+gesture only. This is usually a mediapipe/"
        "Python version mismatch; try 'pip install mediapipe==0.10.9' "
        "(a known-stable version) if you want this signal enabled."
    )

# MediaPipe FaceMesh landmark indices for the mouth (standard 468-point mesh).
_UPPER_LIP = 13
_LOWER_LIP = 14
_MOUTH_LEFT = 61
_MOUTH_RIGHT = 291


def measure_mouth_aspect_ratio(bgr_face_crop: np.ndarray) -> float | None:
    """
    Returns the Mouth Aspect Ratio (vertical lip opening / horizontal
    mouth width) for the face in this crop, or None if no face/landmarks
    could be found (mediapipe missing, face too small/blurry for it, etc).
    """
    if not MEDIAPIPE_AVAILABLE or bgr_face_crop is None or bgr_face_crop.size == 0:
        return None

    rgb = cv2.cvtColor(bgr_face_crop, cv2.COLOR_BGR2RGB)
    result = _face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark
    h, w = bgr_face_crop.shape[:2]

    def point(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    upper = point(_UPPER_LIP)
    lower = point(_LOWER_LIP)
    left = point(_MOUTH_LEFT)
    right = point(_MOUTH_RIGHT)

    mouth_height = float(np.linalg.norm(upper - lower))
    mouth_width = float(np.linalg.norm(left - right))
    if mouth_width < 1e-3:
        return None

    return mouth_height / mouth_width


class MouthMovementTracker:
    """
    Keeps a short rolling history of MAR values per (camera, track_id) and
    reports whether the mouth has been actively moving recently.

    Variance-based: a mouth that's just sitting open OR sitting closed
    has low variance; a mouth that's actually talking opens and closes
    repeatedly, producing noticeably higher variance in the MAR signal.
    """

    def __init__(self, history_len: int = 10, movement_std_threshold: float = 0.018):
        self.history_len = history_len
        self.movement_std_threshold = movement_std_threshold
        self._min_samples = max(4, history_len // 2)
        self._history: dict[tuple, deque] = {}

    def update(self, key: tuple, mar: float | None) -> None:
        if mar is None:
            return
        history = self._history.get(key)
        if history is None:
            history = deque(maxlen=self.history_len)
            self._history[key] = history
        history.append(mar)

    def has_enough_data(self, key: tuple) -> bool:
        history = self._history.get(key)
        return history is not None and len(history) >= self._min_samples

    def is_actively_talking(self, key: tuple) -> bool:
        history = self._history.get(key)
        if history is None or len(history) < self._min_samples:
            return False
        return float(np.std(history)) >= self.movement_std_threshold

    def forget(self, key: tuple) -> None:
        self._history.pop(key, None)