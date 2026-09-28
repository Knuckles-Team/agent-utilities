"""The duality gate scans tracked working-tree Python sources only."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/security/check_httpx_duality.py"


def _gate():
    spec = importlib.util.spec_from_file_location("check_httpx_duality", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tracked_working_tree_is_scanned_without_ignored_copies(tmp_path: Path) -> None:
    gate = _gate()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "src" / "transport.py"
    tracked.parent.mkdir()
    tracked.write_text("pass\n", encoding="utf-8")
    subprocess.run(["git", "add", "src/transport.py"], cwd=tmp_path, check=True)

    bad = "def connect():\n    return SSETransport(auth=child_auth({}))\n"
    tracked.write_text(bad, encoding="utf-8")
    ignored = tmp_path / "build" / "transport.py"
    ignored.parent.mkdir()
    ignored.write_text(bad, encoding="utf-8")

    assert gate._candidate_files(tmp_path) == [tracked]
    assert [(v.path, v.producer) for v in gate.scan(tmp_path)] == [
        ("src/transport.py", "child_auth")
    ]


def test_git_search_failure_is_not_a_clean_scan(tmp_path: Path) -> None:
    gate = _gate()
    with pytest.raises(gate.HttpxDualityGateError, match="git grep"):
        gate._candidate_files(tmp_path)
