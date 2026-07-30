"""Run the dependency-free personal activity demo portal."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.activity_repository import (  # noqa: E402
    ActivityRepository,
    ActivityRepositoryError,
)
from src.portal_server import create_portal_server  # noqa: E402


def _positive_minutes(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "session minutes must be an integer"
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "session minutes must be positive"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serve the local QR activity portal. The server deliberately "
            "does not access-log request paths because QR URLs contain secrets."
        )
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="TCP port (default: 8080).",
    )
    parser.add_argument(
        "--database",
        default=os.getenv("DATABASE_PATH", "data/face_identity.db"),
        help=(
            "Initialized SQLite database path (default: DATABASE_PATH or "
            "data/face_identity.db)."
        ),
    )
    parser.add_argument(
        "--photo-root",
        default=str(ROOT),
        help=(
            "Base directory for relative configured photo paths "
            "(default: project root)."
        ),
    )
    parser.add_argument(
        "--session-minutes",
        type=_positive_minutes,
        default=15,
        help="Fixed in-memory session lifetime (default: 15 minutes).",
    )
    parser.add_argument(
        "--secure-cookie",
        action="store_true",
        help=(
            "Add the Secure cookie attribute. Enable this when users reach "
            "the server through an HTTPS reverse proxy."
        ),
    )
    return parser.parse_args()


def _project_relative(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    args = parse_args()
    if not 0 <= args.port <= 65535:
        print("Portal startup failed: port must be between 0 and 65535.")
        return 2

    database_path = _project_relative(args.database).resolve()
    photo_root = _project_relative(args.photo_root).resolve()

    try:
        repository = ActivityRepository(database_path)
        server = create_portal_server(
            (args.host, args.port),
            repository,
            session_ttl_seconds=args.session_minutes * 60,
            secure_cookie=args.secure_cookie,
            photo_root=photo_root,
        )
    except (ActivityRepositoryError, OSError, ValueError) as exc:
        # These startup errors contain configuration, not request credentials.
        print(f"Portal startup failed: {exc}")
        return 1

    host, port = server.server_address[:2]
    scheme = "https" if args.secure_cookie else "http"
    print(f"Portal listening on {scheme}://{host}:{port}")
    if not args.secure_cookie:
        print(
            "Local HTTP demo mode: use an HTTPS reverse proxy and "
            "--secure-cookie for non-local access."
        )
    print("Request access logging is disabled to protect QR/session secrets.")

    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("Portal stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
