from __future__ import annotations

import csv
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class BrochureEntry:
    camera: str
    person_name: str
    timestamp_raw: float

    @property
    def timestamp_clock(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp_raw))


class BrochureLogger:
    """Thread-safe logger for recording brochure pickup events to CSV and in-memory store."""

    def __init__(self, csv_path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._entries: list[BrochureEntry] = []
        self._csv_path = csv_path

        if self._csv_path is not None:
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not self._csv_path.exists():
                with self._csv_path.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(["camera", "person_name", "timestamp"])

    def log_pickup(self, camera: str, person_name: str, timestamp_raw: float | None = None) -> BrochureEntry:
        if timestamp_raw is None:
            timestamp_raw = time.time()

        entry = BrochureEntry(
            camera=camera,
            person_name=person_name,
            timestamp_raw=timestamp_raw,
        )

        with self._lock:
            self._entries.append(entry)
            if self._csv_path is not None:
                with self._csv_path.open("a", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow([entry.camera, entry.person_name, entry.timestamp_clock])

        print(f"[BROCHURE PICKUP LOGGED] Camera: {camera} | Person: {person_name} | Time: {entry.timestamp_clock}")

        return entry

    def get_all(self) -> list[BrochureEntry]:
        with self._lock:
            return list(self._entries)

    def total_count(self) -> int:
        with self._lock:
            return len(self._entries)


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


class BrochureTracker:
    """Tracks wrist keypoint interactions inside defined brochure ROI boxes per camera, enforcing debouncing and per-person cooldowns."""

    def __init__(
        self,
        rois: dict[str, tuple[int, int, int, int] | None] | None = None,
        debounce_frames: int = 3,
        cooldown_seconds: float = 15.0,
    ):
        self.rois = rois if rois is not None else {}
        self.debounce_frames = debounce_frames
        self.cooldown_seconds = cooldown_seconds

        self._wrist_in_roi_counts: dict[tuple[str, int], int] = {}
        self._person_cooldowns: dict[str, float] = {}
        self._recent_pickups: list[tuple[str, str, float]] = []  # (cam_name, person_label, display_expiry_time)
        self._lock = threading.Lock()

    def update_roi(self, cam_name: str, roi: tuple[int, int, int, int] | None):
        self.rois[cam_name] = roi

    def check_wrist_in_roi(
        self,
        cam_name: str,
        track_id: int,
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray,
        person_name: str,
        current_time: float,
    ) -> bool:
        """
        Evaluates wrist keypoints against camera's brochure ROI box.
        Returns True if a valid new brochure pickup event is triggered.
        """
        roi = self.rois.get(cam_name)
        if roi is None or track_id == -1:
            return False

        rx1, ry1, rx2, ry2 = roi
        wrists = extract_wrist_keypoints(keypoints_xy, keypoints_conf)

        wrist_inside = False
        for wx, wy in wrists:
            if rx1 <= wx <= rx2 and ry1 <= wy <= ry2:
                wrist_inside = True
                break

        key = (cam_name, track_id)
        if wrist_inside:
            count = self._wrist_in_roi_counts.get(key, 0) + 1
            self._wrist_in_roi_counts[key] = count

            if count >= self.debounce_frames:
                # Check per-person cooldown
                last_pickup_time = self._person_cooldowns.get(person_name, 0.0)
                if (current_time - last_pickup_time) >= self.cooldown_seconds:
                    self._person_cooldowns[person_name] = current_time
                    self._wrist_in_roi_counts[key] = 0

                    with self._lock:
                        self._recent_pickups.append((cam_name, person_name, current_time + 3.0))

                    return True
        else:
            self._wrist_in_roi_counts[key] = max(0, self._wrist_in_roi_counts.get(key, 0) - 1)

        return False

    def forget_track(self, cam_name: str, track_id: int):
        self._wrist_in_roi_counts.pop((cam_name, track_id), None)

    def draw_overlays(self, cam_name: str, frame: np.ndarray, current_time: float):
        """Renders brochure ROI box and active pickup notification alerts on the display frame."""
        roi = self.rois.get(cam_name)
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 191, 0), 2)
            cv2.putText(
                frame,
                "Brochure Area",
                (rx1, max(20, ry1 - 8)),
                cv2.FONT_HERSHEY_COMPLEX,
                0.55,
                (255, 191, 0),
                2,
            )

        with self._lock:
            # Clean expired pickup alerts
            self._recent_pickups = [p for p in self._recent_pickups if current_time < p[2]]
            active_alerts = [p for p in self._recent_pickups if p[0] == cam_name]

        y_offset = 60
        for alert in active_alerts:
            cam_name_alert, name_alert, _ = alert
            cv2.putText(
                frame,
                f"Brochure Picked by: {name_alert}",
                (10, y_offset),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            y_offset += 25


def calibrate_brochure_rois(quadrant_names: list[str], get_camera_frame_fn) -> dict[str, tuple[int, int, int, int] | None]:
    """Interactive GUI loop for drawing rectangular brochure pickup region (ROI) per camera feed using mouse clicks."""
    brochure_rois = {}
    current_points = []
    drawing_rect = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_points, drawing_rect
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(current_points) == 0:
                current_points = [(x, y)]
                drawing_rect = True
            elif len(current_points) == 1:
                current_points.append((x, y))
                drawing_rect = False
        elif event == cv2.EVENT_MOUSEMOVE and drawing_rect and len(current_points) == 1:
            # Interactive hover helper
            pass

    win_name = "Set Brochure Box ROI"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_callback)

    for cam_name in quadrant_names:
        current_points = []
        drawing_rect = False
        print(f"\n=== {cam_name} ke liye Brochure Box ROI set karo ===")
        print("Click Top-Left point, then Click Bottom-Right point. Press 'c' to confirm, 'r' to reset, 's' to skip.")

        while True:
            cam_frame = get_camera_frame_fn(cam_name)
            if cam_frame is None:
                time.sleep(0.01)
                continue

            quad_frame = cv2.resize(cam_frame, (960, 540))
            display_frame = quad_frame.copy()

            if len(current_points) == 1:
                pt1 = current_points[0]
                cv2.circle(display_frame, pt1, 5, (0, 255, 255), -1)
                cv2.putText(
                    display_frame,
                    f"{cam_name}: Click corner 2/2 to define box",
                    (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.6,
                    (0, 255, 255),
                    1,
                )
            elif len(current_points) == 2:
                x1 = min(current_points[0][0], current_points[1][0])
                y1 = min(current_points[0][1], current_points[1][1])
                x2 = max(current_points[0][0], current_points[1][0])
                y2 = max(current_points[0][1], current_points[1][1])

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 191, 0), 2)
                cv2.putText(
                    display_frame,
                    "Brochure Area",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.55,
                    (255, 191, 0),
                    2,
                )
                cv2.putText(
                    display_frame,
                    "Press 'c' to confirm ROI, 'r' to reset",
                    (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.6,
                    (0, 255, 0),
                    1,
                )
            else:
                cv2.putText(
                    display_frame,
                    f"{cam_name}: Click corner 1/2 for Brochure Box",
                    (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.6,
                    (0, 255, 255),
                    1,
                )

            cv2.putText(
                display_frame,
                "Press 's' to skip brochure ROI for this camera",
                (10, 510),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            cv2.imshow(win_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                current_points = []
                drawing_rect = False
            elif key == ord('c') and len(current_points) == 2:
                x1 = min(current_points[0][0], current_points[1][0])
                y1 = min(current_points[0][1], current_points[1][1])
                x2 = max(current_points[0][0], current_points[1][0])
                y2 = max(current_points[0][1], current_points[1][1])
                brochure_rois[cam_name] = (x1, y1, x2, y2)
                print(f"{cam_name} brochure ROI confirmed: {brochure_rois[cam_name]}")
                break
            elif key == ord('s'):
                brochure_rois[cam_name] = None
                print(f"{cam_name} brochure ROI skipped.")
                break
            elif key == ord('q'):
                cv2.destroyWindow(win_name)
                sys.exit(0)

    cv2.destroyWindow(win_name)
    print("\nBrochure ROI setup complete.\n")
    return brochure_rois
