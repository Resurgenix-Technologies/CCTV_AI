from __future__ import annotations

import pytest

from scripts.evaluate_face_archive import (
    ArchiveValidationError,
    member_identity,
)


def test_member_identity_accepts_expected_dataset_layout() -> None:
    assert member_identity(
        "test_images/sarisht/frame.png",
        {"sarisht"},
    ) == ("sarisht", "frame.png")


@pytest.mark.parametrize(
    "member",
    [
        "../outside.png",
        "test_images/sarisht/../../outside.png",
        "/test_images/sarisht/frame.png",
        "unexpected/sarisht/frame.png",
        "test_images/unknown/frame.png",
    ],
)
def test_member_identity_rejects_unsafe_or_unknown_paths(
    member: str,
) -> None:
    with pytest.raises(ArchiveValidationError):
        member_identity(member, {"sarisht"})


def test_member_identity_skips_directories_and_non_images() -> None:
    assert member_identity("test_images/sarisht/", {"sarisht"}) is None
    assert member_identity(
        "test_images/sarisht/labels.txt",
        {"sarisht"},
    ) is None
