"""AU-SEMANTIC-R011.9: ``knowledge_graph/ingest_worker.py`` is deleted.

The local ingest-worker entry point is removed in favor of the EG-served
durable-job path. This is a forbidden-module census: it fails loud if the
module (or an importable reference to it) ever reappears.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "agent_utilities"
REPO_ROOT = PACKAGE_ROOT.parent


@pytest.mark.spec("AU-SEMANTIC-R011.9")
def test_ingest_worker_module_is_absent() -> None:
    """The forbidden module no longer exists under ``agent_utilities/``."""
    target = PACKAGE_ROOT / "knowledge_graph" / "ingest_worker.py"
    assert not target.exists(), (
        "AU-SEMANTIC-R011.9 requires 'knowledge_graph/ingest_worker.py' to "
        "be deleted; it still exists."
    )


@pytest.mark.spec("AU-SEMANTIC-R011.9")
def test_no_importer_of_ingest_worker_remains() -> None:
    """No module under ``agent_utilities/`` or ``tests/`` imports the
    deleted module. Comment/docstring prose mentions are not imports and are
    intentionally excluded by matching only ``import``/``from`` statements.
    """
    pattern = (
        r"^\s*(from agent_utilities\.knowledge_graph\.ingest_worker import"
        r"|import agent_utilities\.knowledge_graph\.ingest_worker\b)"
    )
    result = subprocess.run(
        [
            "git",
            "grep",
            "-nE",
            pattern,
            "--",
            "agent_utilities",
            "tests",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # git grep exit code 1 == no matches found (the desired outcome).
    assert result.returncode == 1, (
        "Found live import(s) of the deleted 'knowledge_graph.ingest_worker' "
        f"module:\n{result.stdout}"
    )


@pytest.mark.spec("AU-SEMANTIC-R011.9")
def test_no_console_script_names_ingest_worker() -> None:
    """``pyproject.toml`` carries no console-script/entry-point pointing at
    the deleted module (e.g. the former ``kg-ingest-worker`` script)."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "knowledge_graph.ingest_worker" not in pyproject
