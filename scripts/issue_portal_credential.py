"""Issue an expiring QR credential for one enrolled person."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.activity_repository import (  # noqa: E402
    ActivityRepository,
    ActivityRepositoryError,
)
from src.config import Settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Issue a random, expiring portal credential and QR image. "
            "The database stores only a SHA-256 digest of the token."
        )
    )
    parser.add_argument(
        "person_id",
        help="Enrolled person UUID that will own the credential.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080",
        help="Public portal origin encoded into the QR image.",
    )
    parser.add_argument(
        "--ttl-hours",
        type=float,
        default=24.0,
        help="Credential lifetime in hours (default: 24).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "PNG destination. Defaults to outputs/portal_<credential-id>.png."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ttl_hours <= 0:
        print("--ttl-hours must be positive.", file=sys.stderr)
        return 2

    parsed_base = urlsplit(args.base_url)
    if (
        parsed_base.scheme not in {"http", "https"}
        or not parsed_base.netloc
        or parsed_base.query
        or parsed_base.fragment
    ):
        print(
            "--base-url must be an http(s) origin or base path without "
            "a query or fragment.",
            file=sys.stderr,
        )
        return 2

    try:
        import qrcode
    except ImportError:
        print(
            "The qrcode package is required. Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    settings = Settings.from_env()
    repository = ActivityRepository(settings.database_path)
    issued_at = datetime.now(timezone.utc)

    try:
        credential = repository.issue_portal_credential(
            args.person_id,
            issued_at + timedelta(hours=args.ttl_hours),
            issued_at=issued_at,
        )
    except (ActivityRepositoryError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    portal_url = (
        args.base_url.rstrip("/")
        + "/q/"
        + credential.token
    )
    output = (
        args.output
        or settings.paths.outputs
        / f"portal_{credential.credential_id}.png"
    ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        image = qrcode.make(portal_url)
        image.save(output)
    except OSError as exc:
        try:
            repository.revoke_portal_credential(
                credential.credential_id
            )
        except ActivityRepositoryError:
            pass
        print(f"Could not save QR image: {exc}", file=sys.stderr)
        return 1

    print(f"Credential ID: {credential.credential_id}")
    print(f"Expires at: {credential.expires_at}")
    print(f"QR image: {output}")
    print(f"Portal URL (contains a secret; share only with its owner): {portal_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
