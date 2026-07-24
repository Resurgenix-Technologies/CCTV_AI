"""Apply recognition event schema migrations and show the result."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402
from src.database import database_connection  # noqa: E402
from src.event_repository import (  # noqa: E402
    migrate_recognition_events_schema,
)


def main() -> int:
    settings = Settings.from_env()

    try:
        with database_connection(settings.database_path) as connection:
            migrate_recognition_events_schema(connection)

            columns = [
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(recognition_events);"
                ).fetchall()
            ]

            count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM recognition_events;
                    """
                ).fetchone()[0]
            )

    except sqlite3.Error as exc:
        print(f"Migration failed: {exc}")
        return 1

    print(
        "Recognition event migration completed."
    )
    print(
        f"Database: {settings.database_path}"
    )
    print(
        f"Existing event rows preserved: {count}"
    )
    print(
        "Columns: " + ", ".join(columns)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
