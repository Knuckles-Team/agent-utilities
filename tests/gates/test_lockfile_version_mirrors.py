from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).parents[2]

_PYPROJECT = textwrap.dedent(
    """\
    [project]
    name = "widget-agent"
    version = "3.3.0"
    dependencies = ["agent-utilities[mcp]>=2.0.0,<3.0.0"]
    """
)

_UV_LOCK_OK = textwrap.dedent(
    """\
    version = 1
    [[package]]
    name = "widget-agent"
    version = "3.3.0"
    source = { editable = "." }
    """
)

_UV_LOCK_STALE = textwrap.dedent(
    """\
    version = 1
    [[package]]
    name = "widget-agent"
    version = "2.0.1"
    source = { editable = "." }
    """
)

_REQUIREMENTS_OK = "agent-utilities[mcp]>=2.0.0,<3.0.0\n"
_REQUIREMENTS_STALE_PIN = (
    "agent-utilities==1.0.0\n    # via widget-agent (pyproject.toml)\n"
)
_REQUIREMENTS_STALE_RANGE = "agent-utilities[mcp]>=1.0.0,<2.0.0\n"


def _gate_module() -> ModuleType:
    source = ROOT / "scripts" / "check_lockfile_version_mirrors.py"
    spec = importlib.util.spec_from_file_location(
        "check_lockfile_version_mirrors", source
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(
    root: Path, pyproject: str, uv_lock: str | None, requirements: str | None
) -> None:
    (root / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    if uv_lock is not None:
        (root / "uv.lock").write_text(uv_lock, encoding="utf-8")
    if requirements is not None:
        (root / "requirements.txt").write_text(requirements, encoding="utf-8")


def test_matching_mirrors_pass(tmp_path: Path) -> None:
    gate = _gate_module()
    _write(tmp_path, _PYPROJECT, _UV_LOCK_OK, _REQUIREMENTS_OK)

    assert gate.validate(tmp_path) == []


def test_absent_lock_artifacts_do_not_false_positive(tmp_path: Path) -> None:
    gate = _gate_module()
    _write(tmp_path, _PYPROJECT, uv_lock=None, requirements=None)

    assert gate.validate(tmp_path) == []


def test_stale_uv_lock_self_version_fails(tmp_path: Path) -> None:
    """B6: the self-package's own `uv.lock` entry falls behind pyproject.toml."""
    gate = _gate_module()
    _write(tmp_path, _PYPROJECT, _UV_LOCK_STALE, requirements=None)

    findings = gate.validate(tmp_path)

    assert len(findings) == 1
    assert "uv.lock" in findings[0]
    assert "2.0.1" in findings[0]
    assert "3.3.0" in findings[0]


def test_stale_requirements_pin_fails(tmp_path: Path) -> None:
    """C2: requirements.txt pins a dependency below pyproject.toml's own floor."""
    gate = _gate_module()
    _write(tmp_path, _PYPROJECT, uv_lock=None, requirements=_REQUIREMENTS_STALE_PIN)

    findings = gate.validate(tmp_path)

    assert len(findings) == 1
    assert "requirements.txt" in findings[0]
    assert "1.0.0" in findings[0]


def test_stale_requirements_range_fails(tmp_path: Path) -> None:
    """C2: requirements.txt's own range drifted from pyproject.toml's range."""
    gate = _gate_module()
    _write(tmp_path, _PYPROJECT, uv_lock=None, requirements=_REQUIREMENTS_STALE_RANGE)

    findings = gate.validate(tmp_path)

    assert len(findings) == 1
    assert "requirements.txt" in findings[0]


def test_both_artifacts_stale_reports_both(tmp_path: Path) -> None:
    gate = _gate_module()
    _write(tmp_path, _PYPROJECT, _UV_LOCK_STALE, _REQUIREMENTS_STALE_PIN)

    findings = gate.validate(tmp_path)

    assert len(findings) == 2
