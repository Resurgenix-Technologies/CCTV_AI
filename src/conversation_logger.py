"""Human-readable and CSV conversation-estimate logging."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


class ConversationLogger:
    """Write one clear conversation-estimate stream per recognition run."""

    def __init__(self, directory: Path, session_id: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prefix = f"conversation_{stamp}_{session_id[:8]}"
        self.log_path = directory / f"{prefix}.log"
        self.csv_path = directory / f"{prefix}.csv"
        self._log: TextIO = self.log_path.open("w", encoding="utf-8")
        self._csv_file: TextIO = self.csv_path.open(
            "w", encoding="utf-8", newline=""
        )
        self._csv = csv.DictWriter(
            self._csv_file,
            fieldnames=(
                "event",
                "camera",
                "person_a",
                "person_b",
                "person_a_name",
                "person_b_name",
                "started_at",
                "event_at",
                "duration_seconds",
                "normalized_distance",
                "method",
            ),
        )
        self._csv.writeheader()
        self._csv_file.flush()
        self._log.write("Conversation estimates (sustained proximity)\n")
        self._log.write(f"Session: {session_id}\n")
        self._log.write("Speech is not inferred from these records.\n\n")
        self._log.flush()

    def write(self, event: object, person_names: dict[str, str] | None = None) -> None:
        person_names = person_names or {}
        person_a_name = person_names.get(event.person_a_id, event.person_a_id)
        person_b_name = person_names.get(event.person_b_id, event.person_b_id)
        row = {
            "event": event.event_type,
            "camera": event.camera_id,
            "person_a": event.person_a_id,
            "person_b": event.person_b_id,
            "person_a_name": person_a_name,
            "person_b_name": person_b_name,
            "started_at": datetime.fromtimestamp(
                event.started_at, tz=timezone.utc
            ).isoformat(timespec="milliseconds"),
            "event_at": datetime.fromtimestamp(
                event.event_at, tz=timezone.utc
            ).isoformat(timespec="milliseconds"),
            "duration_seconds": f"{event.duration_seconds:.3f}",
            "normalized_distance": f"{event.normalized_distance:.4f}",
            "method": "sustained_proximity_estimate",
        }
        self._csv.writerow(row)
        self._csv_file.flush()
        self._log.write(
            f"[{row['event']}] camera={row['camera']} | "
            f"person_a={row['person_a_name']} ({row['person_a']}) | "
            f"person_b={row['person_b_name']} ({row['person_b']}) | "
            f"started={row['started_at']} | event={row['event_at']} | "
            f"duration={row['duration_seconds']}s | "
            f"distance={row['normalized_distance']}\n"
        )
        self._log.flush()

    def close(self) -> None:
        self._log.close()
        self._csv_file.close()
