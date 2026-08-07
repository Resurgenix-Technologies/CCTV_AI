"""Thread-safe conversation log store shared between the camera worker
loop and the Tkinter GUI."""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConversationEntry:
    camera: str
    person_a: str
    person_b: str
    start_time: float
    end_time: float

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def start_clock(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.start_time))

    @property
    def end_clock(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.end_time))


class ConversationLogger:
    """
    Collects finished conversation sessions and makes them available to
    any number of readers (e.g. a Tkinter GUI polling on a timer).

    Thread-safe: the camera-processing thread(s) call log_conversation(),
    the GUI thread calls get_new_since()/get_all() on its own timer.
    """

    def __init__(self, csv_path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._entries: list[ConversationEntry] = []
        self._csv_path = csv_path

        if self._csv_path is not None:
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not self._csv_path.exists():
                with self._csv_path.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(
                        ["camera", "person_a", "person_b", "start", "end", "duration_seconds"]
                    )

    def log_conversation(
        self,
        camera: str,
        person_a: str,
        person_b: str,
        start_time: float,
        end_time: float,
    ) -> ConversationEntry:
        entry = ConversationEntry(
            camera=camera,
            person_a=person_a,
            person_b=person_b,
            start_time=start_time,
            end_time=end_time,
        )

        with self._lock:
            self._entries.append(entry)
            if self._csv_path is not None:
                with self._csv_path.open("a", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(
                        [
                            entry.camera,
                            entry.person_a,
                            entry.person_b,
                            entry.start_clock,
                            entry.end_clock,
                            f"{entry.duration_seconds:.1f}",
                        ]
                    )

        print(
            f"Logged conversation: {camera} | {person_a} <-> {person_b} "
            f"({entry.start_clock} - {entry.end_clock}, {entry.duration_seconds:.1f}s)"
        )
        return entry

    def get_all(self) -> list[ConversationEntry]:
        with self._lock:
            return list(self._entries)

    def get_new_since(self, count: int) -> list[ConversationEntry]:
        with self._lock:
            return list(self._entries[count:])

    def total_count(self) -> int:
        with self._lock:
            return len(self._entries)


@dataclass(frozen=True)
class HandshakeEntry:
    camera: str
    person_a: str
    person_b: str
    start_time: float
    end_time: float

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def timestamp_clock(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time))


class HandshakeLogger:
    """Thread-safe logger for recording detected handshake interactions to CSV."""

    def __init__(self, csv_path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._entries: list[HandshakeEntry] = []
        self._csv_path = csv_path

        if self._csv_path is not None:
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not self._csv_path.exists():
                with self._csv_path.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(
                        ["camera", "person_a", "person_b", "timestamp", "duration_seconds"]
                    )

    def log_handshake(
        self,
        camera: str,
        person_a: str,
        person_b: str,
        start_time: float,
        end_time: float,
    ) -> HandshakeEntry:
        entry = HandshakeEntry(
            camera=camera,
            person_a=person_a,
            person_b=person_b,
            start_time=start_time,
            end_time=end_time,
        )

        with self._lock:
            self._entries.append(entry)
            if self._csv_path is not None:
                with self._csv_path.open("a", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow([
                        entry.camera,
                        entry.person_a,
                        entry.person_b,
                        entry.timestamp_clock,
                        f"{entry.duration_seconds:.2f}",
                    ])

        print(
            f"🤝 [HANDSHAKE LOGGED] {camera} | {person_a} <-> {person_b} "
            f"| Time: {entry.timestamp_clock} | Duration: {entry.duration_seconds:.1f}s"
        )
        return entry