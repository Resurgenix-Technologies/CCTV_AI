"""Body Appearance Re-ID module: extracts spatial color and structural appearance
vectors from body crops to maintain track identity when faces are obscured or turned away."""

from __future__ import annotations

from collections import deque
import cv2
import numpy as np


def extract_body_appearance_vector(bgr_crop: np.ndarray) -> np.ndarray | None:
    """
    Extracts a normalized 128-dimensional spatial appearance vector combining LAB
    and HSV color histograms across upper and lower body zones.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return None

    h, w = bgr_crop.shape[:2]
    if h < 20 or w < 10:
        return None

    # Resize to standard aspect ratio for histogram consistency
    standardized = cv2.resize(bgr_crop, (64, 128), interpolation=cv2.INTER_AREA)

    # Split into 3 spatial zones: Upper body (torso), Middle body, Lower body (legs)
    h_zone = 128 // 3
    zone_upper = standardized[0:h_zone, :]
    zone_middle = standardized[h_zone : 2 * h_zone, :]
    zone_lower = standardized[2 * h_zone :, :]

    histograms = []
    for zone in (zone_upper, zone_middle, zone_lower):
        # LAB space histograms (Lighting-invariant color)
        lab = cv2.cvtColor(zone, cv2.COLOR_BGR2LAB)
        hist_l = cv2.calcHist([lab], [0], None, [16], [0, 256])
        hist_a = cv2.calcHist([lab], [1], None, [16], [0, 256])
        hist_b = cv2.calcHist([lab], [2], None, [16], [0, 256])

        # HSV space histograms (Hue and Saturation)
        hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])

        histograms.extend([hist_l, hist_a, hist_b, hist_h])

    vector = np.concatenate([h.flatten() for h in histograms], axis=0).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None

    return (vector / norm).astype(np.float32)


class BodyAppearanceReIDTracker:
    """
    Tracks and matches body appearance vectors for each temporary track ID to
    preserve identity continuity when a person turns their face away.
    """

    def __init__(self, history_len: int = 6, similarity_threshold: float = 0.72):
        self.history_len = history_len
        self.similarity_threshold = similarity_threshold
        self._history: dict[tuple[str, int], deque] = {}
        self._confirmed_signatures: dict[str, np.ndarray] = {}

    def update(self, key: tuple[str, int], bgr_crop: np.ndarray, confirmed_name: str | None = None) -> np.ndarray | None:
        vector = extract_body_appearance_vector(bgr_crop)
        if vector is None:
            return None

        history = self._history.get(key)
        if history is None:
            history = deque(maxlen=self.history_len)
            self._history[key] = history
        history.append(vector)

        # Average recent vectors for running track signature
        stacked = np.stack(history, axis=0)
        mean_vec = np.mean(stacked, axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm <= 1e-12:
            return None
        fused = (mean_vec / norm).astype(np.float32)

        if confirmed_name is not None and confirmed_name != "Unknown":
            self._confirmed_signatures[confirmed_name] = fused

        return fused

    def match_appearance(self, key: tuple[str, int]) -> tuple[str | None, float]:
        """Matches a track's body appearance against stored confirmed person signatures."""
        history = self._history.get(key)
        if not history or not self._confirmed_signatures:
            return None, 0.0

        stacked = np.stack(history, axis=0)
        track_vec = np.mean(stacked, axis=0)
        norm = float(np.linalg.norm(track_vec))
        if norm <= 1e-12:
            return None, 0.0
        track_vec = track_vec / norm

        best_name = None
        best_sim = 0.0

        for name, sig in self._confirmed_signatures.items():
            sim = float(np.dot(track_vec, sig))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim >= self.similarity_threshold:
            return best_name, best_sim
        return None, best_sim

    def forget(self, key: tuple[str, int]) -> None:
        self._history.pop(key, None)
