"""Plan 10 meta-tests (extra gates): coupling + retrieval-quality.

A gate that can't fail is not a gate. Each test proves the new gate trips on a
deliberately-broken fixture AND passes on a clean one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _numeric_kernel_available() -> bool:
    """True when the epistemic-graph[numeric] kernel is importable.

    The retrieval-quality gate ranks vectors through the kernel-backed
    ``agent_utilities.numeric`` namespace and SKIPS (exit 0) when it is absent
    (lean/headless CI without the ``[numeric]``/``[graphos]`` extra). Its
    pass/trip assertions below only hold when the gate can actually run, so they
    skip in lockstep with the gate — the same subprocess env the test drives.
    """
    try:
        import agent_utilities.numeric  # noqa: F401
    except ImportError:
        return False
    return True


_needs_numeric_kernel = pytest.mark.skipif(
    not _numeric_kernel_available(),
    reason="retrieval-quality gate requires the epistemic-graph[numeric] kernel "
    "(it skips cleanly without it, so these trip/pass assertions do not apply)",
)


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
    )


# ---- coupling gate -----------------------------------------------------------


def _make_geniusbot_tree(root: Path) -> Path:
    """Create a minimal geniusbot-shaped package tree under ``root``."""
    pkg = root / "geniusbot"
    (pkg / "services").mkdir(parents=True)
    (pkg / "qt").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "services" / "__init__.py").write_text("")
    (pkg / "qt" / "__init__.py").write_text("")
    return pkg


def test_coupling_gate_trips_on_non_adapter_import(tmp_path):
    pkg = _make_geniusbot_tree(tmp_path)
    # Adapter import is fine (allowlisted)...
    (pkg / "services" / "backend_adapter.py").write_text(
        "from agent_utilities.graph import run_graph_query\n"
    )
    # ...but a UI module importing agent_utilities directly is a violation.
    (pkg / "qt" / "widget.py").write_text(
        "from agent_utilities.gateway.aggregator import Aggregator\n"
    )
    res = _run([str(SCRIPTS / "check_coupling.py"), str(pkg)])
    assert res.returncode == 1, res.stderr
    assert "widget.py" in res.stderr


def test_coupling_gate_passes_when_only_adapter_imports(tmp_path):
    pkg = _make_geniusbot_tree(tmp_path)
    (pkg / "services" / "backend_adapter.py").write_text(
        "from agent_utilities.graph import run_graph_query\n"
    )
    (pkg / "services" / "gateway_client.py").write_text("import agent_utilities\n")
    # A UI module that goes through the adapter, not agent_utilities directly.
    (pkg / "qt" / "widget.py").write_text(
        "from geniusbot.services.backend_adapter import run_graph_query\n"
    )
    res = _run([str(SCRIPTS / "check_coupling.py"), str(pkg)])
    assert res.returncode == 0, res.stderr


def test_coupling_gate_passes_with_no_imports(tmp_path):
    pkg = _make_geniusbot_tree(tmp_path)
    (pkg / "qt" / "widget.py").write_text("X = 1\n")
    res = _run([str(SCRIPTS / "check_coupling.py"), str(pkg)])
    assert res.returncode == 0, res.stderr


# ---- retrieval-quality gate --------------------------------------------------


@_needs_numeric_kernel
def test_retrieval_gate_passes_on_clean_corpus():
    res = _run([str(SCRIPTS / "check_retrieval_quality.py")])
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Recall@5=1.000" in res.stdout


@_needs_numeric_kernel
def test_retrieval_gate_trips_on_degraded_corpus():
    res = _run([str(SCRIPTS / "check_retrieval_quality.py"), "--degrade"])
    assert res.returncode == 1, res.stdout + res.stderr
    assert "regression detected" in res.stderr
