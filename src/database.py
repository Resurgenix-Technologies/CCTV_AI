"""SQLite connection helpers for the face identity application."""
from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

class DatabaseUnavailableError(RuntimeError):
    """Raised when the SQLite database cannot be opened."""

@contextmanager
def database_connection(database_path: Path) -> Iterator[sqlite3.Connection]:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        connection = sqlite3.connect(database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON;")
        connection.execute("PRAGMA journal_mode = WAL;")
    except sqlite3.Error as exc:
        raise DatabaseUnavailableError(f"Could not open SQLite database at {database_path}.") from exc
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
