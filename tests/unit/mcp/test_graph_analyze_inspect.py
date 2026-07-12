"""Tests for ``graph_analyze(action="inspect")`` (exhaustive kg-* validation):
``elif action == "inspect": return engine.inspect(target)`` called a method
that does not exist on ``IntelligenceGraphEngine`` — a wire-first gap
(``'IntelligenceGraphEngine' object has no attribute 'inspect'``). The action
is documented in ``agent_utilities/skills/kg-analyze/SKILL.md`` (structural/
subgraph inspection) and has a REST twin (``GET /graph/analyze/inspect``,
``agent_utilities/mcp/kg_server.py::graph_analyze_inspect_endpoint``) that
already sends ``target``.

The fix builds the structural snapshot from REAL, already-wired read
primitives instead of inventing an engine method: ``engine.query_cypher``
(parameterized — CONCEPT:AU-KG.ingest.never-scan-whole-graph, never an
f-string of the target into Cypher) for node properties, falling back to
``bounded_read.get_node_data`` for a local/test graph, plus
``engine.graph_compute.neighbors``/``.degree`` for the O(1) structural
metrics.
"""

from __future__ import annotations

import asyncio
import json

from agent_utilities.mcp import kg_server


class _FakeGraphCompute:
    """Mirrors the subset of ``GraphComputeEngine`` the inspect action uses."""

    def __init__(
        self, neighbors: dict[str, list[str]], props: dict[str, dict] | None = None
    ):
        self._neighbors = neighbors
        self._props = props or {}

    def neighbors(self, node_id):
        return list(self._neighbors.get(node_id, []))

    def degree(self, node_id):
        return len(self._neighbors.get(node_id, []))

    def _get_node_properties(self, node_id):
        return dict(self._props.get(node_id, {}))


class _FakeEngineWithCypher:
    """``backend``-backed engine: ``query_cypher`` is the parameterized read path."""

    def __init__(self, node_id: str, props: dict, neighbors: list[str]):
        self.backend = object()  # truthy — a backend is configured
        self._node_id = node_id
        self._props = props
        self.graph_compute = _FakeGraphCompute({node_id: neighbors})
        self.cypher_calls: list[tuple[str, dict]] = []

    def query_cypher(self, query, params=None):
        self.cypher_calls.append((query, params or {}))
        if (params or {}).get("ident") == self._node_id:
            return [
                {"node": dict(self._props), "labels": [self._props.get("type", "Node")]}
            ]
        return []


class _FakeEngineNoBackend:
    """No backend (local/test graph) — must fall back to ``bounded_read``."""

    def __init__(self, node_id: str, props: dict, neighbors: list[str]):
        self.backend = None
        self.graph_compute = _FakeGraphCompute(
            {node_id: neighbors}, props={node_id: props}
        )

    def query_cypher(self, *a, **k):  # pragma: no cover - must not be reached usefully
        raise AssertionError("query_cypher should not be relied on without a backend")


def _get_tool():
    kg_server.ensure_tools_registered()
    return kg_server.REGISTERED_TOOLS["graph_analyze"]


def test_inspect_no_longer_raises_missing_method(monkeypatch):
    """The exact regression: ``action='inspect'`` must not surface
    ``'IntelligenceGraphEngine' object has no attribute 'inspect'``."""
    engine = _FakeEngineWithCypher(
        "svc:checkout",
        {"type": "Service", "name": "checkout"},
        ["svc:cart", "svc:auth"],
    )
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    tool = _get_tool()
    out = asyncio.run(tool(action="inspect", target="svc:checkout"))

    assert "no attribute 'inspect'" not in out
    assert not out.startswith("Analysis error"), out


def test_inspect_returns_structural_snapshot_via_query_cypher(monkeypatch):
    engine = _FakeEngineWithCypher(
        "svc:checkout",
        {"type": "Service", "name": "checkout"},
        ["svc:cart", "svc:auth"],
    )
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    tool = _get_tool()
    out = asyncio.run(tool(action="inspect", target="svc:checkout"))
    payload = json.loads(out)

    assert payload["id"] == "svc:checkout"
    assert payload["properties"]["name"] == "checkout"
    assert payload["degree"] == 2
    assert set(payload["neighbors"]) == {"svc:cart", "svc:auth"}
    assert payload["neighbor_count"] == 2

    # parameterized — the target must travel as a bound param, never be
    # f-string-spliced into the Cypher text (injection safety).
    query, params = engine.cypher_calls[0]
    assert params == {"ident": "svc:checkout"}
    assert "svc:checkout" not in query


def test_inspect_falls_back_to_bounded_read_without_backend(monkeypatch):
    engine = _FakeEngineNoBackend(
        "svc:billing", {"type": "Service", "name": "billing"}, ["svc:ledger"]
    )
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    tool = _get_tool()
    out = asyncio.run(tool(action="inspect", target="svc:billing"))
    payload = json.loads(out)

    assert payload["id"] == "svc:billing"
    assert payload["properties"]["name"] == "billing"
    assert payload["degree"] == 1
    assert payload["neighbors"] == ["svc:ledger"]


def test_inspect_missing_node_reports_not_found(monkeypatch):
    engine = _FakeEngineWithCypher("real-node", {"type": "Service"}, [])
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    tool = _get_tool()
    out = asyncio.run(tool(action="inspect", target="does-not-exist"))

    assert "No node found" in out


def test_inspect_requires_target(monkeypatch):
    engine = _FakeEngineWithCypher("n1", {"type": "Service"}, [])
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    tool = _get_tool()
    # Pass every ident-source param explicitly (empty) — the tool is invoked
    # directly here (bypassing FastMCP's own default-resolution machinery, per
    # this suite's convention, e.g. test_engine_tools_scope_policy.py), so an
    # omitted Field-defaulted param would arrive as the raw ``FieldInfo``
    # sentinel rather than its resolved default.
    out = asyncio.run(tool(action="inspect", target="", query="", node_id=""))

    assert "target" in out.lower()
    assert not engine.cypher_calls  # rejected before any read
