"""Download and verify the official OpenCV YuNet and SFace ONNX files."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://huggingface.co/opencv/face_detection_yunet/"
        "resolve/main/face_detection_yunet_2023mar.onnx?download=true",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://huggingface.co/opencv/face_recognition_sface/"
        "resolve/main/face_recognition_sface_2021dec.onnx?download=true",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for filename, (url, expected_hash) in MODELS.items():
        destination = model_dir / filename
        if destination.is_file() and sha256(destination) == expected_hash:
            print(f"OK       {filename}")
            continue

        print(f"DOWNLOAD {filename}")
        urllib.request.urlretrieve(url, destination)

        actual_hash = sha256(destination)
        if actual_hash != expected_hash:
            destination.unlink(missing_ok=True)
            print(f"Hash verification failed for {filename}", file=sys.stderr)
            return 1

        print(f"VERIFIED {filename}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
