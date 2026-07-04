"""Meta-test: the reliability-corpus gate passes clean and trips on degrade.

A gate that can't fail is not a gate. CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _run(arg: str | None) -> int:
    cmd = [sys.executable, str(SCRIPTS / "check_reliability_corpus.py")]
    if arg:
        cmd.append(arg)
    return subprocess.run(cmd, capture_output=True, text=True).returncode


def test_reliability_corpus_gate_passes_clean():
    assert _run(None) == 0


def test_reliability_corpus_gate_trips_on_degrade():
    assert _run("--degrade") == 1
