"""Direct NDI receiver compatible with the ndi-python NDIlib module."""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass

import cv2
import numpy as np


class NDIUnavailableError(RuntimeError):
    """Raised when NDI bindings or runtime are unavailable."""


class NDISourceNotFoundError(RuntimeError):
    """Raised when no matching NDI source is discovered."""


def _load_ndi():
    try:
        import NDIlib as ndi
    except ImportError as exc:
        raise NDIUnavailableError(
            "NDIlib is unavailable. Install ndi-python and the NDI runtime."
        ) from exc
    return ndi


@dataclass(frozen=True)
class NDISourceInfo:
    index: int
    name: str


def _source_name(source: object) -> str:
    return str(getattr(source, "ndi_name", getattr(source, "p_ndi_name", source)))


def discover_sources(timeout_ms: int = 10000) -> list[NDISourceInfo]:
    ndi = _load_ndi()
    if not ndi.initialize():
        raise NDIUnavailableError("NDI initialization failed.")

    finder = ndi.find_create_v2()
    if finder is None:
        ndi.destroy()
        raise NDIUnavailableError("NDI finder creation failed.")

    try:
        deadline = time.monotonic() + timeout_ms / 1000.0
        sources = []
        while time.monotonic() < deadline:
            ndi.find_wait_for_sources(finder, 500)
            sources = list(ndi.find_get_current_sources(finder))
            if sources:
                break
        return [
            NDISourceInfo(index=index, name=_source_name(source))
            for index, source in enumerate(sources)
        ]
    finally:
        ndi.find_destroy(finder)
        ndi.destroy()


class NDIFrameSource:
    """OpenCV-like direct NDI video source."""

    def __init__(
        self,
        source_selector: str = "",
        *,
        discovery_timeout_ms: int = 10000,
        capture_timeout_ms: int = 1000,
    ) -> None:
        self.ndi = _load_ndi()
        self.source_selector = source_selector.strip()
        self.discovery_timeout_ms = discovery_timeout_ms
        self.capture_timeout_ms = capture_timeout_ms
        self.finder = None
        self.receiver = None
        self.connected_name = ""
        self._open()

    def _open(self) -> None:
        if not self.ndi.initialize():
            raise NDIUnavailableError("NDI initialization failed.")

        self.finder = self.ndi.find_create_v2()
        if self.finder is None:
            self.ndi.destroy()
            raise NDIUnavailableError("NDI finder creation failed.")

        deadline = time.monotonic() + self.discovery_timeout_ms / 1000.0
        sources = []
        while time.monotonic() < deadline:
            self.ndi.find_wait_for_sources(self.finder, 500)
            sources = list(self.ndi.find_get_current_sources(self.finder))
            if sources:
                break

        if not sources:
            self.release()
            raise NDISourceNotFoundError(
                "No NDI sources were found before the discovery timeout."
            )

        selected = self._select_source(sources)

        settings = self.ndi.RecvCreateV3()
        settings.color_format = self.ndi.RECV_COLOR_FORMAT_BGRX_BGRA

        self.receiver = self.ndi.recv_create_v3(settings)
        if self.receiver is None:
            self.release()
            raise NDIUnavailableError("NDI receiver creation failed.")

        self.ndi.recv_connect(self.receiver, selected)
        self.connected_name = _source_name(selected)

        self.ndi.find_destroy(self.finder)
        self.finder = None

    def _select_source(self, sources: list[object]) -> object:
        if not self.source_selector:
            return sources[0]

        if self.source_selector.isdigit():
            index = int(self.source_selector)
            if 0 <= index < len(sources):
                return sources[index]
            raise NDISourceNotFoundError(
                f"NDI source index {index} is outside 0..{len(sources)-1}."
            )

        selector = self.source_selector.casefold()
        for source in sources:
            if selector in _source_name(source).casefold():
                return source

        available = ", ".join(_source_name(source) for source in sources)
        raise NDISourceNotFoundError(
            f"No NDI source matched {self.source_selector!r}. "
            f"Available: {available}"
        )

    def is_opened(self) -> bool:
        return self.receiver is not None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.receiver is None:
            return False, None

        frame_type, video_frame, _, _ = self.ndi.recv_capture_v2(
            self.receiver,
            self.capture_timeout_ms,
            want_audio=False,
            want_metadata=False,
        )

        if frame_type == self.ndi.FRAME_TYPE_VIDEO:
            try:
                frame = np.copy(video_frame.data)
            finally:
                self.ndi.recv_free_video_v2(self.receiver, video_frame)

            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif frame.ndim != 3 or frame.shape[2] != 3:
                raise NDIUnavailableError(
                    f"Unsupported decoded NDI frame shape: {frame.shape}"
                )
            return True, np.ascontiguousarray(frame)

        if frame_type == self.ndi.FRAME_TYPE_ERROR:
            raise NDIUnavailableError("NDI receiver reported a capture error.")

        return False, None

    def release(self) -> None:
        if self.receiver is not None:
            self.ndi.recv_destroy(self.receiver)
            self.receiver = None
        if self.finder is not None:
            self.ndi.find_destroy(self.finder)
            self.finder = None
        try:
            self.ndi.destroy()
        except Exception:
            pass

    def __enter__(self) -> "NDIFrameSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class ThreadedNDIFrameSource:
    """Low-latency NDI reader that always exposes the newest video frame."""

    def __init__(self, *args, **kwargs) -> None:
        self._source = NDIFrameSource(*args, **kwargs)
        self.connected_name = self._source.connected_name
        self.is_recording = False
        self.fps = 30.0
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._timestamp = 0.0
        self._sequence = 0
        self._delivered_sequence = 0
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while self._running:
            try:
                ok, frame = self._source.read()
            except Exception:
                if not self._running:
                    break
                continue
            if ok and frame is not None:
                with self._lock:
                    self._latest = frame
                    self._timestamp = time.time()
                    self._sequence += 1

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._latest is None or self._sequence == self._delivered_sequence:
                return False, None
            self._delivered_sequence = self._sequence
            return True, self._latest.copy()

    def capture_timestamp(self) -> float:
        with self._lock:
            return self._timestamp or time.time()

    def release(self) -> None:
        self._running = False
        self._source.release()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.release()
