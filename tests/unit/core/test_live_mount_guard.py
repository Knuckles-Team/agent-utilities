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

    assert live_mount_guard._has_active_source_mount(package_dir) is True


def test_accepts_package_below_an_active_source_mount(monkeypatch) -> None:
    package_dir = Path("/au/agent_utilities")
    _mock_mountinfo(monkeypatch, "/", "/au")

    assert (
        live_mount_guard._has_active_source_mount(
            package_dir,
            source_roots=(Path("/au"),),
        )
        is True
    )


def test_rejects_unrelated_mounted_ancestor(monkeypatch) -> None:
    package_dir = Path("/usr/local/lib/python3.14/site-packages/agent_utilities")
    _mock_mountinfo(monkeypatch, "/", "/usr", "/usr/local")

    assert (
        live_mount_guard._has_active_source_mount(
            package_dir,
            source_roots=(Path("/au"),),
        )
        is False
    )


def test_rejects_deeper_mount_that_is_not_a_declared_source_root(monkeypatch) -> None:
    package_dir = Path("/au/vendor/agent_utilities")
    _mock_mountinfo(monkeypatch, "/", "/au", "/au/vendor")

    assert (
        live_mount_guard._has_active_source_mount(
            package_dir,
            source_roots=(Path("/au"),),
        )
        is False
    )


def test_rejects_package_when_only_root_is_mounted(monkeypatch) -> None:
    _mock_mountinfo(monkeypatch, "/")

    assert (
        live_mount_guard._has_active_source_mount(
            Path("/au/agent_utilities"),
            source_roots=(Path("/"),),
        )
        is False
    )


def test_check_canonicalizes_injected_package_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    package_dir = source_root / "agent_utilities"
    package_dir.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(source_root, target_is_directory=True)
    injected = alias / "agent_utilities" / ".." / "agent_utilities"
    _mock_mountinfo(monkeypatch, "/", str(source_root))

    values = {
        live_mount_guard._SKIP_ENV_VAR: False,
        live_mount_guard._IN_POD_ENV_VAR: "cluster",
        "PYTHONPATH": str(source_root),
    }
    monkeypatch.setattr(
        live_mount_guard,
        "setting",
        lambda name, default, **_kwargs: values.get(name, default),
    )

    assert live_mount_guard.check_live_mount(package_dir=injected) is True


def test_decodes_mountinfo_path_escapes(monkeypatch) -> None:
    source_root = Path("/source tree/tab\tline\nslash\\root")
    package_dir = source_root / "agent_utilities"
    escaped_root = r"/source\040tree/tab\011line\012slash\134root"
    _mock_mountinfo(monkeypatch, "/", escaped_root)

    assert (
        live_mount_guard._has_active_source_mount(
            package_dir,
            source_roots=(source_root,),
        )
        is True
    )
