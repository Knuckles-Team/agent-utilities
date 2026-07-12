"""Meta-test: the CPD drift gate passes clean and trips on a broken fixture.

A gate that can't fail is not a gate. CONCEPT:AU-KG.retrieval.capability-power-descriptor
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
MD_PATH = ROOT / "docs" / "capabilities-power.md"
JSON_PATH = ROOT / "docs" / "capabilities-power.json"

# The CPD gate builds the FULL MCP tool registry (`kg_server`) to prove every
# tool has exactly one CPD — that pulls the serving stack (starlette/fastmcp).
# CI's guardrails job installs only the package core (deliberately lean, to catch
# extra-leakage on the OTHER gates), so the registry can't be built there and the
# coverage gate would be meaningless against a partial registry anyway. It runs in
# the FULL-env pre-commit (the `guardrail-cpd-drift` hook) instead. Skip in lean.
_needs_server_stack = pytest.mark.skipif(
    importlib.util.find_spec("starlette") is None,
    reason="CPD gate needs the full MCP tool registry (serving stack: "
    "starlette/fastmcp); runs in the full-env pre-commit, not the lean CI job",
)


def _run_check_cpd() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "check_cpd.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


@_needs_server_stack
def test_check_cpd_passes_on_the_checked_in_set():
    """The committed docs/capabilities-power.{md,json} must be in sync right now."""
    result = _run_check_cpd()
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def _restore_md():
    original = MD_PATH.read_text(encoding="utf-8")
    try:
        yield
    finally:
        MD_PATH.write_text(original, encoding="utf-8")


@_needs_server_stack
def test_check_cpd_trips_when_the_checked_in_doc_is_stale(_restore_md):
    """Appending content the live generator would never produce must fail the gate."""
    with MD_PATH.open("a", encoding="utf-8") as fh:
        fh.write("\n<!-- hand-edited, never regenerated -->\n")
    result = _run_check_cpd()
    assert result.returncode == 1
    assert "DRIFT" in result.stdout or "stale" in result.stdout


@pytest.fixture
def _restore_json():
    original = JSON_PATH.read_text(encoding="utf-8")
    try:
        yield
    finally:
        JSON_PATH.write_text(original, encoding="utf-8")


@_needs_server_stack
def test_check_cpd_trips_when_a_cpd_is_deleted_from_the_json(_restore_json):
    """Removing one capability from the checked-in JSON must fail coverage/drift."""
    import json

    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    assert data["capabilities"], "fixture precondition: at least one CPD present"
    data["capabilities"].pop()
    data["count"] = len(data["capabilities"])
    JSON_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    result = _run_check_cpd()
    assert result.returncode == 1
