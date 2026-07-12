"""Meta-test: the reliability-corpus gate passes clean and trips on degrade.

A gate that can't fail is not a gate. CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _numeric_kernel_available() -> bool:
    try:
        import agent_utilities.numeric  # noqa: F401
    except ImportError:
        return False
    return True


# The reliability-corpus gate builds/scores a corpus through the harness, which
# needs the kernel-backed agent_utilities.numeric (not pip-installable in lean CI);
# it skips cleanly there, so these pass/trip assertions only hold with the kernel.
_needs_numeric_kernel = pytest.mark.skipif(
    not _numeric_kernel_available(),
    reason="reliability-corpus gate requires the epistemic-graph[numeric] kernel "
    "(it skips cleanly without it, so these trip/pass assertions do not apply)",
)


def _run(arg: str | None) -> int:
    cmd = [sys.executable, str(SCRIPTS / "check_reliability_corpus.py")]
    if arg:
        cmd.append(arg)
    return subprocess.run(cmd, capture_output=True, text=True).returncode


@_needs_numeric_kernel
def test_reliability_corpus_gate_passes_clean():
    assert _run(None) == 0


@_needs_numeric_kernel
def test_reliability_corpus_gate_trips_on_degrade():
    assert _run("--degrade") == 1
