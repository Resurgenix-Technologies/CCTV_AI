"""List discoverable NDI sources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ndi_source import (  # noqa: E402
    NDIUnavailableError,
    discover_sources,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-ms", type=int, default=10000)
    args = parser.parse_args()

    try:
        sources = discover_sources(args.timeout_ms)
    except NDIUnavailableError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not sources:
        print("No NDI sources found.")
        return 1

    for source in sources:
        print(f"[{source.index}] {source.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
