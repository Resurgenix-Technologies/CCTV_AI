"""Delete locally captured enrolment photos without touching test_images."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENROLLMENT = ROOT / "data" / "enrollment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion without an interactive prompt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ENROLLMENT.mkdir(parents=True, exist_ok=True)

    if not args.yes:
        answer = input(
            f"Delete local captures inside {ENROLLMENT}? [y/N]: "
        ).strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled.")
            return 0

    for child in ENROLLMENT.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    print("Local captured enrolment images deleted.")
    print("Repository test_images were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
