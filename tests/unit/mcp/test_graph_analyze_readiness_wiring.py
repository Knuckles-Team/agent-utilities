"""Wiring proof: ``graph_analyze(action="readiness")`` reaches the REAL
``collect_readiness_snapshot`` collector from the live MCP entrypoint (GOC-02).

This does not call ``readiness.collect_readiness_snapshot`` directly (that is
what ``tests/unit/knowledge_graph/test_readiness.py`` already proves in
isolation) — it drives the REGISTERED coroutine ``kg_server.REGISTERED_TOOLS
["graph_analyze"]`` exactly like a real MCP/REST call would, the same seam
``tests/unit/mcp/test_graph_analyze_inspect.py`` uses for ``action="inspect"``.
Only the engine boundary is faked; nothing here mocks the readiness collector,
``build_code_context``, or the dispatch chain in between.

Proves the same known-bad / known-good pair as the unit-level test, but through
the actual live entrypoint: readiness must be UNAVAILABLE when the engine is
degraded during the synthetic-query canary, and READY only when a genuine,
grounded answer comes back — never green from liveness alone.
"""

from __future__ import annotations

import asyncio
import json

from agent_utilities.mcp import kg_server


class _FakeEngine:
    """Mirrors ``tests/unit/knowledge_graph/test_readiness.py``'s fake engine —
    a ``query_cypher``-shaped double that drives the REAL ``build_code_context``/
    ``resolve_anchors`` call sites."""

    def __init__(self, *, anchor_rows=None, raise_exc: Exception | None = None):
        self._anchor_rows = anchor_rows or []
        self._raise_exc = raise_exc
        self.backend = object()

    def query_cypher(self, cypher: str, params: dict | None = None):
        params = params or {}
        is_anchor_query = "c.id = $id" in cypher or "c.name = $tok" in cypher
        if is_anchor_query and self._raise_exc is not None:
            raise self._raise_exc
        if is_anchor_query:
            return list(self._anchor_rows)
        return []


_ANCHOR_ROW = {
    "id": "code:graph_analyze",
    "name": "graph_analyze",
    "file_path": "agent_utilities/mcp/tools/analysis_tools.py",
    "line": 1,
    "language": "python",
    "kind": "function",
    "instance": "",
    "source_system": "agent-utilities",
}


def _get_tool():
    kg_server.ensure_tools_registered()
    return kg_server.REGISTERED_TOOLS["graph_analyze"]


def test_readiness_action_is_registered_on_graph_analyze():
    tool = _get_tool()
    assert tool is not None


def test_readiness_wiring_reports_unavailable_when_engine_degraded(monkeypatch):
    """KNOWN-BAD PROOF via the live entrypoint: the engine goes degraded
    mid-canary -> the served tool's readiness payload must say UNAVAILABLE,
    never a false 'ready' because a socket answered."""
    engine = _FakeEngine(raise_exc=ConnectionError("engine transport down"))
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    tool = _get_tool()
    out = asyncio.run(tool(action="readiness", query="graph_analyze"))

    payload = out.claims[0]
    assert payload["schema_version"] == "graphos.readiness.v1"
    assert payload["checks"]["synthetic_query"]["state"] == "unavailable"
    assert payload["checks"]["synthetic_query"]["reason"] == "engine_degraded"
    assert payload["overall"] == "unavailable"
    assert "synthetic_query" in payload["required_failures"]


def test_readiness_wiring_reports_ready_when_query_genuinely_resolves(monkeypatch):
    """KNOWN-GOOD PROOF via the live entrypoint: a real grounded answer comes
    back through the actual dispatch path -> and only then -> ready.

    ``runtime_health.collect_health`` (a real bounded socket probe against
    whatever engine this sandbox happens to have running — unrelated to the
    synthetic-query wiring this test targets) is stubbed to a healthy report
    so the assertion isolates the seam under test: the served ``readiness``
    action actually reaches ``build_code_context`` through the real engine
    handle this test injects.
    """
    engine = _FakeEngine(anchor_rows=[_ANCHOR_ROW])
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.connector_coverage.enumerate_expected_connectors",
        lambda: [],
    )
    monkeypatch.setattr(
        "agent_utilities.observability.runtime_health.collect_health",
        lambda: {
            "status": "healthy",
            "generated_at": "2026-08-16T00:00:00+00:00",
            "checks": [
                {"name": "engine", "status": "ok", "detail": {"resolved_mode": "shared"}, "latency_ms": 1.0},
                {"name": "embedding_endpoint", "status": "ok", "detail": {}, "latency_ms": 0.5},
            ],
        },
    )

    tool = _get_tool()
    out = asyncio.run(tool(action="readiness", query="graph_analyze"))

    payload = out.claims[0]
    assert payload["checks"]["synthetic_query"]["state"] == "ready"
    assert payload["checks"]["synthetic_query"]["evidence_count"] >= 1
    assert payload["overall"] == "ready"
    # No engine handle, credential, or raw query text ever leaks into the
    # served payload.
    serialized = json.dumps(payload)
    assert "ConnectionError" not in serialized


def test_readiness_action_rejected_by_the_generic_analyze_tool_before_the_split(monkeypatch):
    """Sanity: an unrelated bogus action still gets the existing focused-tool
    redirect, proving the allowlist add didn't loosen the guard."""
    engine = _FakeEngine(anchor_rows=[_ANCHOR_ROW])
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    tool = _get_tool()
    out = asyncio.run(tool(action="code_context", query="graph_analyze"))
    payload = out.claims[0]
    assert payload["error"] == "action belongs to a focused graph tool"
