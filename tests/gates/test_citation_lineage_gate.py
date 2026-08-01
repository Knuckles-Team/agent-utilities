"""Meta-test: the citation-lineage gate passes clean and trips on degrade.

A gate that can't fail is not a gate. CONCEPT:AU-KG.retrieval.mandatory-evidence-citation

Unlike the reliability/eval-corpus/retrieval-quality gates, the
citation-resolution half of this gate needs no epistemic-graph[full] kernel
(fragment_markdown/chunk_text/_fragment_ids_for_span are pure Python) — only
the embedding-version-mismatch half does, and that half degrades to a
cleanly-reported SKIP without the kernel rather than failing the whole gate.
So these pass/trip assertions hold in every environment.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _run(arg: str | None) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(SCRIPTS / "check_citation_lineage.py")]
    if arg:
        cmd.append(arg)
    return subprocess.run(cmd, capture_output=True, text=True)


def test_citation_lineage_gate_passes_clean():
    result = _run(None)
    assert result.returncode == 0, result.stdout + result.stderr


def test_citation_lineage_gate_trips_on_degrade():
    result = _run("--degrade")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "not citable" in result.stdout
