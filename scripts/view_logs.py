"""Display or export recent structured recognition events."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402
from src.event_repository import (  # noqa: E402
    EventRepository,
    VALID_EVENT_TYPES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View recognition events stored in SQLite."
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--event-type",
        choices=sorted(VALID_EVENT_TYPES),
    )
    parser.add_argument(
        "--session-id",
        help="Filter by a full or partial recognition session ID.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV output path.",
    )
    return parser.parse_args()


def fetch_rows(
    database_path: Path,
    *,
    limit: int,
    event_type: str | None,
    session_id: str | None,
) -> list[sqlite3.Row]:
    if limit < 1:
        raise ValueError("--limit must be positive.")

    # Also creates the event table if this is the first run.
    EventRepository(database_path)

    conditions: list[str] = []
    parameters: list[object] = []

    if event_type:
        conditions.append("event_type = ?")
        parameters.append(event_type)

    if session_id:
        conditions.append("session_id LIKE ?")
        parameters.append(f"{session_id}%")

    query = """
        SELECT
            event_id,
            occurred_at,
            session_id,
            event_type,
            camera_source,
            local_track_id,
            full_name,
            person_id,
            similarity,
            detection_confidence,
            frame_index
        FROM recognition_events
    """

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY event_id DESC LIMIT ?"
    parameters.append(limit)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(query, parameters).fetchall()
    finally:
        connection.close()


def print_rows(rows: list[sqlite3.Row]) -> None:
    if not rows:
        print("No recognition events found.")
        return

    for row in rows:
        person_id = (
            str(row["person_id"])[:8]
            if row["person_id"]
            else "-"
        )
        similarity = (
            f"{row['similarity']:.3f}"
            if row["similarity"] is not None
            else "-"
        )
        confidence = (
            f"{row['detection_confidence']:.3f}"
            if row["detection_confidence"] is not None
            else "-"
        )
        frame_index = (
            str(row["frame_index"])
            if row["frame_index"] is not None
            else "-"
        )

        print(
            f"{row['occurred_at']} | "
            f"session={str(row['session_id'])[:8]} | "
            f"{row['event_type']:20} | "
            f"source={row['camera_source']} | "
            f"track={row['local_track_id']} | "
            f"person={row['full_name'] or 'UNKNOWN'} | "
            f"id={person_id} | "
            f"similarity={similarity} | "
            f"detection={confidence} | "
            f"frame={frame_index}"
        )


def export_csv(rows: list[sqlite3.Row], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "event_id",
        "occurred_at",
        "session_id",
        "event_type",
        "camera_source",
        "local_track_id",
        "full_name",
        "person_id",
        "similarity",
        "detection_confidence",
        "frame_index",
    ]

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict(row) for row in rows)

    print(f"Exported {len(rows)} events to: {output}")


def main() -> int:
    args = parse_args()
    settings = Settings.from_env()

    try:
        rows = fetch_rows(
            settings.database_path,
            limit=args.limit,
            event_type=args.event_type,
            session_id=args.session_id,
        )
    except (sqlite3.Error, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_rows(rows)

    if args.csv:
        export_csv(rows, args.csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#python scripts\view_logs.py `
#  --limit 1000 `
#  --csv "outputs\recognition_events.csv"

#python -m compileall src scripts tests