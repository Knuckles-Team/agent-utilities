"""Focused coverage for the runtime live-source mount check."""

from __future__ import annotations

import builtins
import io
from pathlib import Path

from agent_utilities.core import live_mount_guard


def _mountinfo(*mount_points: str) -> str:
    return "".join(
        f"36 25 0:32 / {mount_point} rw,relatime - ext4 /dev/root rw\n"
        for mount_point in mount_points
    )


def _mock_mountinfo(monkeypatch, *mount_points: str) -> None:
    monkeypatch.setattr(
        builtins,
        "open",
        lambda *_args, **_kwargs: io.StringIO(_mountinfo(*mount_points)),
    )


def test_accepts_package_directory_when_it_is_an_active_mount(monkeypatch) -> None:
    package_dir = Path("/au/agent_utilities")
    _mock_mountinfo(monkeypatch, "/", str(package_dir))

    assert live_mount_guard._is_bind_mounted(package_dir) is True


def test_accepts_package_below_an_active_source_mount(monkeypatch) -> None:
    package_dir = Path("/au/agent_utilities")
    _mock_mountinfo(monkeypatch, "/", "/au")

    assert live_mount_guard._is_bind_mounted(package_dir) is True


def test_rejects_package_when_only_root_is_mounted(monkeypatch) -> None:
    _mock_mountinfo(monkeypatch, "/")

    assert live_mount_guard._is_bind_mounted(Path("/au/agent_utilities")) is False
