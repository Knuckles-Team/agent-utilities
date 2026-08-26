"""LANE H-kg: KG-query / MCP-tool-catalog validation-harness stage module.

Proves ``mcp_tool_catalog`` and ``kg_general`` (``scripts/_harness_kg_queries.py``)
fail closed when a required count is zero and pass when the graph is populated,
against a fake backend/engine — no live KG required. Also regression-guards the
``labels(n)`` trap documented in the module: the engine's Cypher executor does not
implement ``labels(n)`` (it raises ``CypherEngineError`` live), so this module must
never emit that call and must use labelled ``MATCH`` patterns / the SQL
``node_type`` column instead.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_MODULE_PATH = Path(__file__).parents[3] / "scripts" / "_harness_kg_queries.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_harness_kg_queries", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _non_docstring_string_literals(source: str) -> list[str]:
    """Every plain/f-string literal the module's CODE builds — deliberately
    excluding module/class/function docstrings, which legitimately discuss (in
    prose) the very query shapes the code must avoid. Used to regression-guard
    against the module ever actually *emitting* one of those shapes, as opposed to
    merely mentioning it in commentary.
    """
    tree = ast.parse(source)
    docstring_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_ids.add(id(body[0].value))

    literals: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstring_ids:
                literals.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            parts = [
                piece.value
                for piece in node.values
                if isinstance(piece, ast.Constant) and isinstance(piece.value, str)
            ]
            literals.append("".join(parts))
    return literals


class _FakeBackend:
    """Fake ``engine.backend`` — records every Cypher string it is asked to run and
    answers node-count / edge-count / PROVIDES-reachability queries from canned
    tables, so tests control exactly what the graph "contains" without a live KG.
    """

    def __init__(
        self,
        counts: dict[str, int] | None = None,
        total_nodes: int = 0,
        used_tool_edge_count: int = 0,
        provides_ids: list[str] | None = None,
    ) -> None:
        self.counts = counts or {}
        self.total_nodes = total_nodes
        self.used_tool_edge_count = used_tool_edge_count
        self.provides_ids = provides_ids if provides_ids is not None else []
        self.calls: list[str] = []

    def execute(self, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        self.calls.append(cypher)
        # Regression guard, exercised on every single call this fake ever answers:
        # the module must never emit the unsupported labels(n) form.
        assert "labels(" not in cypher, f"emitted unsupported labels(n) call: {cypher}"

        if "PROVIDES" in cypher:
            return [{"id": i} for i in self.provides_ids]
        if "USED_TOOL" in cypher:
            return [{"c": self.used_tool_edge_count}]
        if cypher.strip() == "MATCH (n) RETURN count(n) AS c":
            return [{"c": self.total_nodes}]
        match = re.search(r"MATCH \(n:(\w+)\)", cypher)
        if match:
            return [{"c": self.counts.get(match.group(1), 0)}]
        raise AssertionError(f"unexpected cypher shape in test fake: {cypher!r}")


class _FakeEngine:
    """Fake engine handle: exposes ``.backend.execute`` (Cypher) and ``.sql`` (the
    DataFusion catalog surface), matching the two entrypoints
    ``scripts/_harness_kg_queries.py`` calls on whatever engine it is handed.
    """

    def __init__(
        self,
        backend: _FakeBackend,
        label_sweep_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.backend = backend
        self._label_sweep_rows = (
            label_sweep_rows if label_sweep_rows is not None else []
        )
        self.sql_calls: list[str] = []

    def sql(self, query: str) -> list[dict[str, Any]]:
        self.sql_calls.append(query)
        # Regression guard: must group on node_type, never the nonexistent `type`
        # column (confirmed live: `SELECT type, COUNT(*) FROM nodes` raises
        # "Schema error: No field named type").
        assert "GROUP BY node_type" in query
        assert "SELECT type" not in query
        return self._label_sweep_rows


_POPULATED_COUNTS = {
    "Tool": 2941,
    "CallableResource": 361,
    "RunTrace": 72,
    "ToolCall": 500,
    "Concept": 4000,
    "MCPServer": 90,
    "OutcomeEvaluation": 40,
}
_POPULATED_LABEL_SWEEP = [
    {"node_type": "Concept", "c": 4000},
    {"node_type": "Tool", "c": 2941},
]
# The module transforms each raw SQL row {"node_type": ..., "c": ...} (the shape
# `engine.sql()` actually returns) into {"label": ..., "count": ...} in its result.
_POPULATED_LABEL_SWEEP_TRANSFORMED = [
    {"label": "Concept", "count": 4000},
    {"label": "Tool", "count": 2941},
]


def _populated_engine(**overrides: Any) -> _FakeEngine:
    counts = dict(_POPULATED_COUNTS)
    counts.update(overrides.pop("counts", {}))
    backend = _FakeBackend(
        counts=counts,
        total_nodes=overrides.pop("total_nodes", 56_853),
        used_tool_edge_count=overrides.pop("used_tool_edge_count", 72),
        provides_ids=overrides.pop("provides_ids", ["a", "b", "c"]),
    )
    return _FakeEngine(
        backend,
        label_sweep_rows=overrides.pop("label_sweep_rows", _POPULATED_LABEL_SWEEP),
    )


# ---------------------------------------------------------------------------------
# mcp_tool_catalog
# ---------------------------------------------------------------------------------
def test_mcp_tool_catalog_passes_when_populated() -> None:
    mod = _module()
    engine = _populated_engine()

    result = asyncio.run(mod.mcp_tool_catalog(engine))

    assert result["tool_count"] == 2941
    assert result["callable_resource_count"] == 361
    assert result["provides_reachable_count"] == 3


def test_mcp_tool_catalog_fails_when_tool_count_zero() -> None:
    mod = _module()
    engine = _populated_engine(counts={"Tool": 0})

    with pytest.raises(RuntimeError, match=r":Tool count is 0"):
        asyncio.run(mod.mcp_tool_catalog(engine))


def test_mcp_tool_catalog_fails_when_callable_resource_count_zero() -> None:
    mod = _module()
    engine = _populated_engine(counts={"CallableResource": 0})

    with pytest.raises(RuntimeError, match=r":CallableResource count is 0"):
        asyncio.run(mod.mcp_tool_catalog(engine))


def test_mcp_tool_catalog_fails_when_callable_resource_orphaned_from_server() -> None:
    """CallableResource nodes exist but none are reachable via
    (:Server)-[:PROVIDES]-> — the exact orphaning failure shape the task calls out.
    """
    mod = _module()
    engine = _populated_engine(provides_ids=[])

    with pytest.raises(RuntimeError, match="orphaned"):
        asyncio.run(mod.mcp_tool_catalog(engine))


def test_mcp_tool_catalog_dedupes_provides_ids_client_side() -> None:
    """Cypher DISTINCT is documented unreliable for this query shape, so the module
    must dedupe the returned ids itself rather than trust engine-side DISTINCT.
    """
    mod = _module()
    engine = _populated_engine(provides_ids=["dup", "dup", "dup"])

    result = asyncio.run(mod.mcp_tool_catalog(engine))

    assert result["provides_reachable_count"] == 1


# ---------------------------------------------------------------------------------
# kg_general
# ---------------------------------------------------------------------------------
def test_kg_general_passes_when_populated() -> None:
    mod = _module()
    engine = _populated_engine()

    result = asyncio.run(mod.kg_general(engine))

    assert result["total_nodes"] == 56_853
    assert result["run_trace_count"] == 72
    assert result["tool_call_count"] == 500
    assert result["concept_count"] == 4000
    assert result["mcp_server_count"] == 90
    assert result["outcome_count"] == 40
    assert result["used_tool_edge_count"] == 72
    assert result["label_sweep"] == _POPULATED_LABEL_SWEEP_TRANSFORMED


@pytest.mark.parametrize(
    ("overrides", "expected_match"),
    [
        ({"total_nodes": 0}, "total node count is 0"),
        ({"label_sweep_rows": []}, "label-cardinality sweep returned 0 labels"),
        ({"counts": {"RunTrace": 0}}, r":RunTrace count is 0"),
        ({"counts": {"ToolCall": 0}}, r":ToolCall count is 0"),
        ({"counts": {"Concept": 0}}, r":Concept count is 0"),
        ({"counts": {"MCPServer": 0}}, r":MCPServer count is 0"),
        ({"counts": {"OutcomeEvaluation": 0}}, r":OutcomeEvaluation count is 0"),
        ({"used_tool_edge_count": 0}, "no .*USED_TOOL.* edge found"),
    ],
)
def test_kg_general_fails_when_a_required_count_is_zero(
    overrides: dict[str, Any], expected_match: str
) -> None:
    mod = _module()
    engine = _populated_engine(**overrides)

    with pytest.raises(RuntimeError, match=expected_match):
        asyncio.run(mod.kg_general(engine))


def test_kg_general_reports_every_failure_at_once() -> None:
    """A harness stage that stops at the first failing check hides the rest of the
    picture; kg_general must report all failing sub-checks in one raise.
    """
    mod = _module()
    engine = _populated_engine(counts={"RunTrace": 0, "ToolCall": 0}, total_nodes=0)

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(mod.kg_general(engine))

    message = str(excinfo.value)
    assert "total node count is 0" in message
    assert ":RunTrace count is 0" in message
    assert ":ToolCall count is 0" in message


# ---------------------------------------------------------------------------------
# labels(n) trap regression coverage
# ---------------------------------------------------------------------------------
def test_module_source_never_emits_labels_n() -> None:
    """Static regression guard: the engine's Cypher executor does not implement
    ``labels(n)`` (confirmed live to raise CypherEngineError, not return null or an
    empty result), so a query built around it would either error loudly (this
    module must never trigger that) or, in older reports, silently misreport. Only
    the module's actual code strings are scanned (docstrings are excluded — they
    legitimately discuss this exact trap in prose). Belt and suspenders alongside
    the runtime guard in ``_FakeBackend.execute`` above, which would fail any test
    that actually issued such a call.
    """
    literals = _non_docstring_string_literals(_MODULE_PATH.read_text())
    assert not any("labels(" in literal for literal in literals)


def test_module_never_filters_on_bare_type_or_label_sql_columns() -> None:
    """Static regression guard for the silent-zero trap: a bare `type`/`label`
    column filter in the SQL surface returns 0 rows instead of erroring. The
    verified-working replacement is `node_type`. Only the module's actual code
    strings are scanned — docstrings legitimately discuss the rejected shape.
    """
    literals = _non_docstring_string_literals(_MODULE_PATH.read_text())
    assert not any("SELECT type" in literal for literal in literals)
    assert not any("GROUP BY type" in literal for literal in literals)


def test_label_sweep_uses_the_sql_catalog_with_node_type() -> None:
    mod = _module()
    engine = _populated_engine()

    asyncio.run(mod.kg_general(engine))

    assert len(engine.sql_calls) == 1
    assert "node_type" in engine.sql_calls[0]
    assert "labels(" not in engine.sql_calls[0]


# ---------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------
def test_safe_label_rejects_unsafe_interpolation() -> None:
    mod = _module()

    assert mod._safe_label("Tool") == "Tool"
    with pytest.raises(ValueError, match="unsafe"):
        mod._safe_label("Tool) DETACH DELETE (n")


def test_emit_matches_delegation_probe_convention(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _module()

    mod._emit("mcp_tool_catalog", True, "tool=1 callable_resource=1", 0.42)
    mod._emit("kg_general", False, "total node count is 0", None)

    out = capsys.readouterr().out
    assert "PASS" in out and "mcp_tool_catalog" in out and "0.42" in out
    assert "FAIL" in out and "kg_general" in out
