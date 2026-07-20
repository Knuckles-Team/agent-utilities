"""Plan 10 meta-tests: prove each guardrail actually trips on a broken fixture
and passes on a clean one. A gate that can't fail is not a gate.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _run(script: str, arg: str) -> int:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), arg],
        capture_output=True,
        text=True,
    ).returncode


# ---- sprawl gate -------------------------------------------------------------


def test_sprawl_gate_trips_on_versioned_clone(tmp_path):
    (tmp_path / "router_v2.py").write_text("x = 1\n")
    assert _run("check_sprawl.py", str(tmp_path)) == 1


def test_sprawl_gate_trips_on_merge_marker(tmp_path):
    (tmp_path / "m.py").write_text("# --- Merged from other.py\n")
    assert _run("check_sprawl.py", str(tmp_path)) == 1


def test_sprawl_gate_trips_on_reject_artifact(tmp_path):
    (tmp_path / "patch.orig").write_text("junk\n")
    assert _run("check_sprawl.py", str(tmp_path)) == 1


def test_sprawl_gate_passes_clean(tmp_path):
    (tmp_path / "router.py").write_text("def f():\n    return 1\n")
    assert _run("check_sprawl.py", str(tmp_path)) == 0


# ---- no_stub gate (check_no_stub.py) ----------------------------------------


def test_no_stub_gate_trips_on_mock(tmp_path):
    (tmp_path / "m.py").write_text('def f():\n    return "[Mock] nope"\n')
    assert _run("check_no_stub.py", str(tmp_path)) == 1


def test_no_stub_gate_trips_on_notimplemented(tmp_path):
    (tmp_path / "m.py").write_text("def f():\n    raise NotImplementedError\n")
    assert _run("check_no_stub.py", str(tmp_path)) == 1


def test_no_stub_gate_allows_abstract_ok(tmp_path):
    (tmp_path / "m.py").write_text(
        "def f():\n    raise NotImplementedError  # ABSTRACT-OK\n"
    )
    assert _run("check_no_stub.py", str(tmp_path)) == 0


def test_no_stub_gate_passes_clean(tmp_path):
    (tmp_path / "m.py").write_text("def f():\n    return 42\n")
    assert _run("check_no_stub.py", str(tmp_path)) == 0
