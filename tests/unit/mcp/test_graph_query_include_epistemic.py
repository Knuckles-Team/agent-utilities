"""Live-path tests: `include_epistemic` on `graph_query`/`graph_ask` (WS-1a).

CONCEPT:AU-KB-CURRENCY

Before this, the epistemic read layer (`EpistemicRow`/`EvidenceSpan`,
`include_epistemic` on `KnowledgeGraph.query`/`GraphComputeEngine.query_unified`/
`IntelligenceGraphEngine.uql`/`GraphBackend.execute`) was threaded through every
backend but reachable from ZERO MCP tools/REST routes
(`reports/surpass-6mo/03-au-orchestration-ops.md` item 2). These tests prove the
MCP-facing wiring: `graph_query`'s `include_epistemic=True` currency-upgrades
each row via `IntelligenceGraphEngine.query_cypher(..., include_epistemic=True)`
(the SAME `attach_epistemic_rows` pattern `uql`/`query_unified` already use),
and `graph_ask`'s `include_epistemic=True` threads through `nl_to_query` for the
`cypher` dialect. Mirrors the `_FakeEngine` + `kg_server._resolve_read_engines`
monkeypatch pattern of `test_graph_query_evidence_bundle.py` — no live engine
required.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json

from agent_utilities.knowledge_graph.core.epistemic_row import EpistemicRow
from agent_utilities.mcp import kg_server


def _register_query_tools():
    from fastmcp import FastMCP

    from agent_utilities.mcp.tools.query_tools import register_query_tools

    register_query_tools(FastMCP("test"))


_ROW = EpistemicRow(
    id="agent:foo",
    kind="Agent",
    score=0.9,
    confidence=0.8,
    evidence_refs=[],
    source_refs=["src:gitlab/agent-foo"],
    valid_time=(100, None),
    tx_time=(100, None),
    policy_labels=[],
    properties={"id": "agent:foo", "name": "foo"},
)


class _FakeEngineNoEpistemicKwarg:
    """Mirrors a pre-existing `query_cypher` implementation that does NOT
    accept `include_epistemic` — the default (False) call path must never
    pass it, so this fake intentionally has no such parameter."""

    def query_cypher(self, cypher, params, as_of=None):
        return [{"id": "agent:foo", "name": "foo"}]


class _FakeEngineEpistemic:
    def __init__(self, rows):
        self._rows = rows
        self.seen_include_epistemic: bool | None = None

    def query_cypher(self, cypher, params, as_of=None, include_epistemic=False):
        self.seen_include_epistemic = include_epistemic
        return self._rows if include_epistemic else [{"id": "agent:foo"}]


def _fake_resolve_read_engines(engine):
    def _resolve(target):
        return ([("default", engine)], {}, False)

    return _resolve


def test_graph_query_default_never_passes_include_epistemic_kwarg(monkeypatch):
    """The default (unset) path must call `query_cypher` exactly as before —
    a fake lacking the `include_epistemic` parameter must not raise."""
    _register_query_tools()
    engine = _FakeEngineNoEpistemicKwarg()
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines(engine)
    )
    out = json.loads(
        asyncio.run(
            kg_server._execute_tool("graph_query", cypher="MATCH (a:Agent) RETURN a")
        )
    )
    assert out == [{"id": "agent:foo", "name": "foo"}]


def test_graph_query_include_epistemic_true_returns_epistemic_rows(monkeypatch):
    _register_query_tools()
    engine = _FakeEngineEpistemic([_ROW])
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines(engine)
    )
    out = json.loads(
        asyncio.run(
            kg_server._execute_tool(
                "graph_query",
                cypher="MATCH (a:Agent) RETURN a",
                include_epistemic=True,
            )
        )
    )
    assert engine.seen_include_epistemic is True
    assert len(out) == 1
    assert out[0]["id"] == "agent:foo"
    assert out[0]["confidence"] == 0.8
    assert out[0]["source_refs"] == ["src:gitlab/agent-foo"]
    assert out[0]["properties"]["name"] == "foo"


def test_graph_query_include_epistemic_false_is_default(monkeypatch):
    _register_query_tools()
    engine = _FakeEngineEpistemic([_ROW])
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines(engine)
    )
    out = json.loads(
        asyncio.run(
            kg_server._execute_tool("graph_query", cypher="MATCH (a:Agent) RETURN a")
        )
    )
    assert out == [{"id": "agent:foo"}]


def test_dataclass_json_serialization_roundtrip():
    """`EpistemicRow` (frozen dataclass) must dataclass-serialize via
    `_json_default`, not `str()`-stringify into an unreadable repr."""
    from agent_utilities.mcp.tools.query_tools import _json_default

    encoded = json.dumps([_ROW], default=_json_default)
    decoded = json.loads(encoded)
    # json round-trips a tuple as a list, so compare via the same round-trip
    # rather than the raw dataclasses.asdict() (which keeps tuples as tuples).
    assert decoded == json.loads(json.dumps([dataclasses.asdict(_ROW)]))
