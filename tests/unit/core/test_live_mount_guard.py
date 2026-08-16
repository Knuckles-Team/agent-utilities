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


# ── U-26/BUG-172: immutable-image provenance must not read as drift ────────


def _in_pod_setting(monkeypatch, *, extra: dict | None = None) -> None:
    values = {
        live_mount_guard._SKIP_ENV_VAR: False,
        live_mount_guard._IN_POD_ENV_VAR: "cluster",
        "PYTHONPATH": "",
    }
    values.update(extra or {})
    monkeypatch.setattr(
        live_mount_guard,
        "setting",
        lambda name, default, **_kwargs: values.get(name, default),
    )


def test_no_mount_no_marker_is_drift_and_check_live_mount_is_false(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    """The historical D-EGK-1 case: no active mount, no provenance marker
    at all -- this is the genuinely ambiguous case the guard must still fail
    loud on. A regression that started treating EVERY unmounted package as
    healthy would silently reintroduce D-EGK-1's stale-code blind spot."""
    package_dir = tmp_path / "agent_utilities"
    package_dir.mkdir()
    _mock_mountinfo(monkeypatch, "/")  # no mount at all
    _in_pod_setting(monkeypatch)

    with caplog.at_level("CRITICAL", logger="agent_utilities.live_mount_guard"):
        status, detail = live_mount_guard.check_live_mount_status(
            package_dir=package_dir
        )

    assert status is live_mount_guard.LiveMountStatus.DRIFT
    assert detail["active_source_mount"] is False
    assert any(rec.levelname == "CRITICAL" for rec in caplog.records)
    assert live_mount_guard.check_live_mount(package_dir=package_dir) is False


def test_no_mount_with_provenance_marker_is_immutable_verified_not_drift(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    """U-26's exact scenario: an immutable image (docker/graphos-unified.
    Dockerfile) installs agent_utilities editable at build time with no
    intended live mount, and writes a `.source-revision` marker to prove it.
    This must be healthy -- never the CRITICAL stale-code warning -- and
    must be distinguishable (IMMUTABLE_VERIFIED) from a real live mount
    (ACTIVE_MOUNT), even though both are "healthy"."""
    package_dir = tmp_path / "agent_utilities"
    package_dir.mkdir()
    (package_dir / live_mount_guard.SOURCE_REVISION_MARKER).write_text(
        "abc1234\n", encoding="utf-8"
    )
    _mock_mountinfo(monkeypatch, "/")  # no mount at all
    _in_pod_setting(monkeypatch)

    with caplog.at_level("CRITICAL", logger="agent_utilities.live_mount_guard"):
        status, detail = live_mount_guard.check_live_mount_status(
            package_dir=package_dir
        )

    assert status is live_mount_guard.LiveMountStatus.IMMUTABLE_VERIFIED
    assert detail["source_revision"] == "abc1234"
    assert not any(rec.levelname == "CRITICAL" for rec in caplog.records)
    # Backward-compatible boolean view: healthy, same as an active mount.
    assert live_mount_guard.check_live_mount(package_dir=package_dir) is True


def test_active_mount_is_active_mount_regardless_of_marker(
    monkeypatch, tmp_path: Path
) -> None:
    """A real live mount always wins -- presence of a stale/leftover marker
    file (e.g. a dev bind-mount over an image that also carries one) must
    never downgrade an active mount to IMMUTABLE_VERIFIED; both are healthy,
    but the status must still reflect what is actually true right now."""
    package_dir = tmp_path / "agent_utilities"
    package_dir.mkdir()
    (package_dir / live_mount_guard.SOURCE_REVISION_MARKER).write_text(
        "unknown", encoding="utf-8"
    )
    _mock_mountinfo(monkeypatch, "/", str(package_dir))
    _in_pod_setting(monkeypatch)

    status, detail = live_mount_guard.check_live_mount_status(package_dir=package_dir)

    assert status is live_mount_guard.LiveMountStatus.ACTIVE_MOUNT
    assert detail["active_source_mount"] is True
    assert live_mount_guard.check_live_mount(package_dir=package_dir) is True


def test_outside_pod_is_not_applicable_never_drift_or_healthy(monkeypatch) -> None:
    monkeypatch.setattr(
        live_mount_guard,
        "setting",
        lambda name, default, **_kwargs: default,
    )

    status, _detail = live_mount_guard.check_live_mount_status()

    assert status is live_mount_guard.LiveMountStatus.NOT_APPLICABLE
    assert live_mount_guard.check_live_mount() is None


def test_installed_source_revision_missing_marker_returns_none(tmp_path: Path) -> None:
    assert live_mount_guard._installed_source_revision(tmp_path) is None


def test_installed_source_revision_strips_whitespace(tmp_path: Path) -> None:
    (tmp_path / live_mount_guard.SOURCE_REVISION_MARKER).write_text(
        "  deadbeef  \n", encoding="utf-8"
    )
    assert live_mount_guard._installed_source_revision(tmp_path) == "deadbeef"
