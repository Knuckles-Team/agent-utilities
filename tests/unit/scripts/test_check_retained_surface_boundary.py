"""Regression tests for the AU-BOUNDARY-R043 retained-surface boundary gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _gate() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "check_retained_surface_boundary.py"
    )
    spec = importlib.util.spec_from_file_location("retained_surface_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_rejects_a_serving_import() -> None:
    gate = _gate()
    failures = gate._source_violations(
        "agent_utilities/agent/example.py",
        "from agent_utilities.server.routers import interop\n",
    )
    assert any("forbidden serving import" in failure for failure in failures)


def test_gate_rejects_a_storage_import() -> None:
    gate = _gate()
    failures = gate._source_violations(
        "agent_utilities/harness/example.py",
        "import sqlite3\nsqlite3.connect(':memory:')\n",
    )
    assert any("forbidden storage import" in failure for failure in failures)


def test_gate_honors_the_reviewed_allowlist() -> None:
    gate = _gate()
    failures = gate._source_violations(
        "agent_utilities/harness/memorydata/adapter.py",
        "from agent_utilities.server.routers.benchmark import judge_binary\n",
    )
    assert failures == []


def test_gate_does_not_flag_an_unrelated_import() -> None:
    gate = _gate()
    failures = gate._source_violations(
        "agent_utilities/agent/example.py",
        "from agent_utilities.core.model_factory import build_model\nimport json\n",
    )
    assert failures == []


def test_repository_retained_surface_has_no_serving_or_storage_violation() -> None:
    assert _gate().violations() == []
