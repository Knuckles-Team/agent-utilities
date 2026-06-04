"""Meta-test for the designation-eval gate (Plan 04 Step 6 / Plan 10).

A gate that can't fail is not a gate. This proves the designation-eval gate
passes on the real frozen corpus (capability filtering genuinely beats
embedding-only) AND trips in ``--degrade`` mode (capabilities stripped so
filtering can't help). Mirrors the style of ``test_gates_meta_extra.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
    )


def test_designation_eval_gate_passes_on_real_corpus():
    res = _run([str(SCRIPTS / "check_designation_eval.py")])
    assert res.returncode == 0, res.stdout + res.stderr
    # Both scores must be printed.
    assert "embedding-only" in res.stdout
    assert "capability-filtered" in res.stdout
    # Filtering strictly helps on the frozen corpus.
    assert "capability filtering beats embedding-only" in res.stdout


def test_designation_eval_gate_trips_on_degraded_corpus():
    res = _run([str(SCRIPTS / "check_designation_eval.py"), "--degrade"])
    assert res.returncode == 1, res.stdout + res.stderr
    assert "did not beat" in res.stderr
