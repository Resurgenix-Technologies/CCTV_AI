from __future__ import annotations

import datetime
import logging
import queue
import threading
import time
from pathlib import Path
import psycopg2


DEFAULT_POSTGRES_URI = (
    "postgresql://neondb_owner:npg_EGhtRg3jr8MB@ep-green-tooth-au043d93.c-10.us-east-1.aws.neon.tech/neondb?sslmode=require"
)

logger = logging.getLogger("PostgresLogger")


class PostgresLogger:
    """
    Asynchronous, thread-safe PostgreSQL logger for recording handshakes,
    talking sessions, and brochure pickup events.
    Uses a background worker queue so DB network latency never blocks camera streams.
    """

    def __init__(self, uri: str = DEFAULT_POSTGRES_URI):
        self.uri = uri
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._init_schema()
        self._worker_thread.start()

    def _get_connection(self):
        return psycopg2.connect(self.uri)

    def _init_schema(self):
        """Creates required PostgreSQL database tables if they do not exist."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS handshakes (
            id SERIAL PRIMARY KEY,
            camera VARCHAR(50) NOT NULL,
            person_a VARCHAR(100) NOT NULL,
            person_b VARCHAR(100) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            duration_seconds DOUBLE PRECISION NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS talking_events (
            id SERIAL PRIMARY KEY,
            camera VARCHAR(50) NOT NULL,
            person_a VARCHAR(100) NOT NULL,
            person_b VARCHAR(100) NOT NULL,
            start_time TIMESTAMPTZ NOT NULL,
            end_time TIMESTAMPTZ NOT NULL,
            duration_seconds DOUBLE PRECISION NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS brochure_pickups (
            id SERIAL PRIMARY KEY,
            camera VARCHAR(50) NOT NULL,
            person_name VARCHAR(100) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(schema_sql)
                conn.commit()
            logger.info("PostgreSQL schema initialized successfully.")
            print("[POSTGRES DB] Schema initialized successfully on Neon PostgreSQL.")
        except Exception as exc:
            logger.error(f"Failed to initialize PostgreSQL schema: {exc}")
            print(f"[POSTGRES DB ERROR] Initializing schema failed: {exc}")


    def log_handshake_async(
        self,
        camera: str,
        person_a: str,
        person_b: str,
        timestamp_raw: float,
        duration_seconds: float,
    ):
        """Enqueues a handshake event for DB insertion."""
        dt = datetime.datetime.fromtimestamp(timestamp_raw, tz=datetime.timezone.utc)
        self._queue.put(
            (
                "HANDSHAKE",
                {
                    "camera": camera,
                    "person_a": person_a,
                    "person_b": person_b,
                    "timestamp": dt,
                    "duration_seconds": duration_seconds,
                },
            )
        )

    def log_talking_async(
        self,
        camera: str,
        person_a: str,
        person_b: str,
        start_time_raw: float,
        end_time_raw: float,
        duration_seconds: float,
    ):
        """Enqueues a talking event for DB insertion."""
        start_dt = datetime.datetime.fromtimestamp(start_time_raw, tz=datetime.timezone.utc)
        end_dt = datetime.datetime.fromtimestamp(end_time_raw, tz=datetime.timezone.utc)
        self._queue.put(
            (
                "TALKING",
                {
                    "camera": camera,
                    "person_a": person_a,
                    "person_b": person_b,
                    "start_time": start_dt,
                    "end_time": end_dt,
                    "duration_seconds": duration_seconds,
                },
            )
        )

    def log_brochure_pickup_async(
        self,
        camera: str,
        person_name: str,
        timestamp_raw: float,
    ):
        """Enqueues a brochure pickup event for DB insertion."""
        dt = datetime.datetime.fromtimestamp(timestamp_raw, tz=datetime.timezone.utc)
        self._queue.put(
            (
                "BROCHURE",
                {
                    "camera": camera,
                    "person_name": person_name,
                    "timestamp": dt,
                },
            )
        )

    def _worker_loop(self):
        conn = None
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            event_type, data = item
            success = False
            for attempt in range(3):
                try:
                    if conn is None or conn.closed != 0:
                        conn = self._get_connection()

                    with conn.cursor() as cur:
                        if event_type == "HANDSHAKE":
                            cur.execute(
                                """
                                INSERT INTO handshakes (camera, person_a, person_b, timestamp, duration_seconds)
                                VALUES (%s, %s, %s, %s, %s)
                                """,
                                (
                                    data["camera"],
                                    data["person_a"],
                                    data["person_b"],
                                    data["timestamp"],
                                    data["duration_seconds"],
                                ),
                            )
                        elif event_type == "TALKING":
                            cur.execute(
                                """
                                INSERT INTO talking_events (camera, person_a, person_b, start_time, end_time, duration_seconds)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                """,
                                (
                                    data["camera"],
                                    data["person_a"],
                                    data["person_b"],
                                    data["start_time"],
                                    data["end_time"],
                                    data["duration_seconds"],
                                ),
                            )
                        elif event_type == "BROCHURE":
                            cur.execute(
                                """
                                INSERT INTO brochure_pickups (camera, person_name, timestamp)
                                VALUES (%s, %s, %s)
                                """,
                                (
                                    data["camera"],
                                    data["person_name"],
                                    data["timestamp"],
                                ),
                            )
                    conn.commit()
                    success = True
                    print(f"[POSTGRES DB INSERTED] Event Type: {event_type} | Camera: {data['camera']}")
                    break

                except Exception as exc:
                    logger.error(f"PostgreSQL insert failed (attempt {attempt+1}): {exc}")
                    if conn:
                        try:
                            conn.close()
                        except Exception:
                            pass
                    conn = None
                    time.sleep(0.5)

            self._queue.task_done()

        if conn and conn.closed == 0:
            try:
                conn.close()
            except Exception:
                pass

    def stop(self):
        """Flushes remaining queued items and stops the background worker thread."""
        self._stop_event.set()
        self._worker_thread.join(timeout=5.0)
