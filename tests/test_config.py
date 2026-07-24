from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings


DISTANT_ENV_NAMES = {
    "FACE_MATCH_MARGIN",
    "LIVE_FACE_DETECTION_MIN_SIZE",
    "LIVE_FACE_IDENTITY_MIN_SIZE",
    "LIVE_FACE_ROI_UPSCALE",
    "LIVE_FACE_ROI_RATIO",
    "LIVE_FACE_ROI_SCORE_THRESHOLD",
    "LIVE_FACE_ROI_MAX_DIMENSION",
    "FACE_DIAGNOSTIC_LOG_INTERVAL",
    "YOLO_IMAGE_SIZE",
    "NDI_LAYOUT",
}


def empty_env(tmp_path: Path) -> Path:
    path = tmp_path / "empty.env"
    path.write_text("", encoding="utf-8")
    return path


def clear_distant_env(monkeypatch) -> None:
    for name in DISTANT_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_distant_camera_defaults_are_safe(monkeypatch, tmp_path: Path) -> None:
    clear_distant_env(monkeypatch)

    settings = Settings.from_env(empty_env(tmp_path))

    assert settings.face_match_margin == 0.05
    assert settings.live_face_detection_min_size == 12
    assert settings.live_face_identity_min_size == 32
    assert settings.live_face_roi_upscale == 2.0
    assert settings.live_face_roi_ratio == 0.55
    assert settings.live_face_roi_score_threshold == 0.60
    assert settings.live_face_roi_max_dimension == 640
    assert settings.face_diagnostic_log_interval == 300
    assert settings.yolo_image_size == 640
    assert settings.ndi_layout == "single"
    assert settings.resolve_input_layout(is_ndi=True) == "single"
    assert settings.resolve_input_layout(is_ndi=False) == "single"
    assert (
        settings.resolve_input_layout(
            is_ndi=False,
            override="2x2",
        )
        == "2x2"
    )


def test_identity_floor_cannot_be_below_detection_floor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    clear_distant_env(monkeypatch)
    monkeypatch.setenv("LIVE_FACE_DETECTION_MIN_SIZE", "40")
    monkeypatch.setenv("LIVE_FACE_IDENTITY_MIN_SIZE", "32")

    with pytest.raises(ValueError, match="cannot be below"):
        Settings.from_env(empty_env(tmp_path))


def test_ndi_quadrant_layout_only_applies_to_ndi_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    clear_distant_env(monkeypatch)
    monkeypatch.setenv("NDI_LAYOUT", "2x2")
    settings = Settings.from_env(empty_env(tmp_path))

    assert settings.resolve_input_layout(is_ndi=True) == "2x2"
    assert settings.resolve_input_layout(is_ndi=False) == "single"
    assert (
        settings.resolve_input_layout(
            is_ndi=True,
            override="single",
        )
        == "single"
    )


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("LIVE_FACE_ROI_UPSCALE", "5", "between"),
        ("LIVE_FACE_ROI_RATIO", "0.1", "between"),
        ("LIVE_FACE_ROI_SCORE_THRESHOLD", "1.1", "between"),
        ("LIVE_FACE_ROI_MAX_DIMENSION", "16", "at least"),
        ("YOLO_IMAGE_SIZE", "16", "at least"),
        ("NDI_LAYOUT", "4x1", "one of"),
    ],
)
def test_invalid_distant_camera_settings_are_rejected(
    monkeypatch,
    tmp_path: Path,
    name: str,
    value: str,
    message: str,
) -> None:
    clear_distant_env(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        Settings.from_env(empty_env(tmp_path))
