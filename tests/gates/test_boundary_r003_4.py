"""Import-boundary regression for AU-BOUNDARY-R003.4.

``agent_utilities/sdd/watcher.py`` used to reach directly into
``agent_utilities.mcp.kg_server``'s private ``_ingest_capabilities`` helper
to re-ingest ``mcp_config.json`` changes -- a cross-module private-symbol
boundary violation (a leading-underscore name is not a public contract).
That call (and its containing ``_reingest_mcp_config`` wrapper) is now gone;
``process_kg_ingest_location`` routes every watched KG location, including
``mcp_config.json``, through the same ``engine.submit_task()`` path.

This module proves both halves of the row: the function/import is gone from
watcher.py, and its one caller (``process_kg_ingest_location``) still works
-- now via the generic engine.submit_task() re-ingestion path -- without it.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

WATCHER_PATH = (
    Path(__file__).resolve().parents[2] / "agent_utilities" / "sdd" / "watcher.py"
)


@pytest.mark.spec("AU-BOUNDARY-R003.4")
def test_sdd_watcher_ingest_capabilities_removed():
    """No import of, or call to, ``_ingest_capabilities`` remains in
    watcher.py, and the ``_reingest_mcp_config`` wrapper that held its only
    call site is gone too."""
    source = WATCHER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(WATCHER_PATH))

    imported_names: set[str] = set()
    called_names: set[str] = set()
    defined_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called_names.add(node.func.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined_names.add(node.name)

    assert "_ingest_capabilities" not in imported_names
    assert "_ingest_capabilities" not in called_names
    assert "_reingest_mcp_config" not in defined_names
    assert "_reingest_mcp_config" not in called_names


@pytest.mark.spec("AU-BOUNDARY-R003.4")
def test_mcp_config_reingest_uses_generic_engine_submit_task(tmp_path):
    """The former caller, ``process_kg_ingest_location``, still re-ingests an
    ``mcp_config.json`` change -- now via the same ``engine.submit_task()``
    path every other watched KG location uses, with no import of kg_server
    at all."""
    from agent_utilities.sdd.watcher import process_kg_ingest_location

    mock_engine = MagicMock()
    mock_engine.submit_task = MagicMock()

    mcp_config = tmp_path / "mcp_config.json"
    mcp_config.write_text("{}")

    process_kg_ingest_location(mock_engine, mcp_config)

    mock_engine.submit_task.assert_called_once_with(
        target_path=str(mcp_config.resolve()),
        is_codebase=False,
        task_type="document",
        provenance={"source": "watcher_kg_ingest"},
    )
