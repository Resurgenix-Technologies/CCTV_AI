import sys
import time
import threading
import cv2
import numpy as np
import NDIlib as ndi


class NDIReceiver:
    """Threaded NDI Video Receiver that continually captures and converts video frames."""

    def __init__(self, ndi_recv):
        self.ndi_recv = ndi_recv
        self.latest_frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            t, v, a, m = ndi.recv_capture_v2(self.ndi_recv, 1000)
            if t != ndi.FRAME_TYPE_VIDEO:
                continue

            frame_raw = np.copy(v.data)
            fourcc = v.FourCC

            if fourcc in (ndi.FOURCC_VIDEO_TYPE_BGRA, ndi.FOURCC_VIDEO_TYPE_BGRX):
                frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGRA2BGR)
            elif fourcc in (ndi.FOURCC_VIDEO_TYPE_RGBA, ndi.FOURCC_VIDEO_TYPE_RGBX):
                frame = cv2.cvtColor(frame_raw, cv2.COLOR_RGBA2BGR)
            elif fourcc == ndi.FOURCC_VIDEO_TYPE_UYVY:
                frame = cv2.cvtColor(frame_raw, cv2.COLOR_YUV2BGR_UYVY)
            else:
                ndi.recv_free_video_v2(self.ndi_recv, v)
                continue

            with self.lock:
                self.latest_frame = frame

            ndi.recv_free_video_v2(self.ndi_recv, v)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        self.running = False


class NDIManager:
    """Manages NDI stream creation, camera frame retrieval, and lifecycle cleanup."""

    def __init__(
        self,
        quadrant_names: list[str],
        camera_source_names: dict[str, str],
        ndi_mode: str = "independent",
        search_timeout: float = 8.0,
    ):
        self.quadrant_names = quadrant_names
        self.camera_source_names = camera_source_names
        self.ndi_mode = ndi_mode
        self.search_timeout = search_timeout

        self.ndi_find = None
        self.ndi_receivers = {}
        self.ndi_recv_handles = {}
        self.shared_ndi_receiver = None
        self.shared_ndi_recv_handle = None
        self.half_width = None
        self.half_height = None
        self.full_width = None
        self.full_height = None

        self._initialize_ndi()

    def _initialize_ndi(self):
        if not ndi.initialize():
            print("NDI could not be initialized")
            sys.exit(1)

        self.ndi_find = ndi.find_create_v2()
        sources = []
        print("Searching for NDI sources...")
        required_count = 4 if self.ndi_mode == "independent" else 1
        search_start = time.time()

        while len(sources) < required_count:
            ndi.find_wait_for_sources(self.ndi_find, 1000)
            sources = ndi.find_get_current_sources(self.ndi_find)
            print(f"  ...found {len(sources)} source(s) so far")

            if self.ndi_mode == "independent" and (time.time() - search_start) > self.search_timeout:
                if len(sources) >= 1:
                    print(
                        f"Only {len(sources)} NDI source(s) found after "
                        f"{self.search_timeout}s (need 4 for independent mode). "
                        "Falling back to quad_split mode."
                    )
                    self.ndi_mode = "quad_split"
                    required_count = 1
                break

        print("Available NDI sources:")
        for i, s in enumerate(sources):
            print(i, s.ndi_name)

        if self.ndi_mode == "independent":
            used_source_indices = set()
            for cam_name in self.quadrant_names:
                wanted_name = self.camera_source_names.get(cam_name, "")
                chosen_index = None

                if wanted_name:
                    for idx, s in enumerate(sources):
                        if idx not in used_source_indices and wanted_name.lower() in s.ndi_name.lower():
                            chosen_index = idx
                            break
                    if chosen_index is None:
                        print(
                            f"WARNING: no NDI source matched CAMERA_SOURCE_NAMES['{cam_name}']="
                            f"'{wanted_name}'. Falling back to order-based assignment."
                        )

                if chosen_index is None:
                    for idx in range(len(sources)):
                        if idx not in used_source_indices:
                            chosen_index = idx
                            break

                if chosen_index is None:
                    print(
                        f"ERROR: not enough NDI sources found for {cam_name}. "
                        f"Found {len(sources)}, need {len(self.quadrant_names)}."
                    )
                    sys.exit(1)

                used_source_indices.add(chosen_index)
                recv_handle = ndi.recv_create_v3()
                ndi.recv_connect(recv_handle, sources[chosen_index])
                print(f"{cam_name} -> connected to: {sources[chosen_index].ndi_name}")

                self.ndi_recv_handles[cam_name] = recv_handle
                self.ndi_receivers[cam_name] = NDIReceiver(recv_handle)

            print("Waiting for first frame on all 4 cameras...")
            while any(self.ndi_receivers[name].get_frame() is None for name in self.quadrant_names):
                time.sleep(0.1)
            print("First frame received on all cameras.")

        else:
            source_index = 0
            self.shared_ndi_recv_handle = ndi.recv_create_v3()
            ndi.recv_connect(self.shared_ndi_recv_handle, sources[source_index])
            print(f"Connected to: {sources[source_index].ndi_name}")

            self.shared_ndi_receiver = NDIReceiver(self.shared_ndi_recv_handle)
            print("Waiting for first frame...")
            while self.shared_ndi_receiver.get_frame() is None:
                time.sleep(0.1)
            print("First frame received.")

            sample_frame = self.shared_ndi_receiver.get_frame()
            self.full_height, self.full_width = sample_frame.shape[:2]
            self.half_width = self.full_width // 2
            self.half_height = self.full_height // 2

    def get_quadrants(self, full_frame: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "CAM1": full_frame[0:self.half_height, 0:self.half_width],
            "CAM2": full_frame[0:self.half_height, self.half_width:self.full_width],
            "CAM3": full_frame[self.half_height:self.full_height, 0:self.half_width],
            "CAM4": full_frame[self.half_height:self.full_height, self.half_width:self.full_width],
        }

    def get_camera_frame(self, cam_name: str) -> np.ndarray | None:
        if self.ndi_mode == "independent":
            return self.ndi_receivers[cam_name].get_frame()
        else:
            full_frame = self.shared_ndi_receiver.get_frame()
            if full_frame is None:
                return None
            return self.get_quadrants(full_frame)[cam_name]

    def cleanup(self):
        """Stops receivers and releases all NDI resources."""
        if self.ndi_mode == "independent":
            for cam_name in self.quadrant_names:
                if cam_name in self.ndi_receivers:
                    self.ndi_receivers[cam_name].stop()
                if cam_name in self.ndi_recv_handles:
                    ndi.recv_destroy(self.ndi_recv_handles[cam_name])
        else:
            if self.shared_ndi_receiver:
                self.shared_ndi_receiver.stop()
            if self.shared_ndi_recv_handle:
                ndi.recv_destroy(self.shared_ndi_recv_handle)

        if self.ndi_find:
            ndi.find_destroy(self.ndi_find)
        ndi.destroy()
