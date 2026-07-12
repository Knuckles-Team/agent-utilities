"""Live-path test: `graph_query`'s additive `envelope='bundle'` param wires the
previously-dead :meth:`EvidenceBundle.from_engine_wire` (Epistemic Substrate
Program C1/D11) onto a real entry point.

Before this test, `EvidenceBundle.from_engine_wire` had unit tests (see
`tests/unit/test_evidence_bundle.py`) but NO live caller anywhere in the
codebase — a Wire-First gap (code exists + is unit-tested, but nothing on any
real path ever invokes it). `graph_query` (the Cypher MCP tool / `/graph/query`
REST twin) now supports the same raw/bundle envelope toggle `graph_ask`/
`nl_query`/`graph_analyze action=code_context` already use: on the
single-connection local Cypher path, `envelope='bundle'` currency-upgrades the
plain rows via `engine.graph.explain_provenance_by_ids` (mirroring
`KnowledgeGraph._attach_epistemic`'s per-row path) and folds the result into ONE
aggregate `EvidenceBundle` via `from_engine_wire`, attached under
`evidence_bundle`. `envelope='raw'` (the default) stays byte-identical.
"""

from __future__ import annotations

import asyncio
import json

from agent_utilities.mcp import kg_server


def _register_query_tools():
    from fastmcp import FastMCP

    from agent_utilities.mcp.tools.query_tools import register_query_tools

    register_query_tools(FastMCP("test"))


class _FakeCompute:
    """Stand-in for `IntelligenceGraphEngine.graph` (a GraphComputeEngine)."""

    def __init__(self, wire_rows):
        self._wire_rows = wire_rows
        self.seen_ids: list[str] | None = None

    def explain_provenance_by_ids(self, ids):
        self.seen_ids = list(ids)
        return self._wire_rows


class _FakeEngine:
    """Stand-in for IntelligenceGraphEngine exposing .query_cypher()/.graph."""

    def __init__(self, rows, wire_rows):
        self._rows = rows
        self.graph = _FakeCompute(wire_rows)

    def query_cypher(self, cypher, params, as_of=None):
        return self._rows


_ROWS = [{"id": "agent:foo", "name": "foo"}, {"id": "agent:bar", "name": "bar"}]
_WIRE_ROWS = [
    {
        "id": "agent:foo",
        "kind": "Agent",
        "score": 0.9,
        "confidence": 0.8,
        "valid_time": [100, None],
        "tx_time": [100, None],
        "source_refs": ["src:gitlab/agent-foo"],
        "evidence_refs": [],
        "policy_labels": [],
    },
    {
        "id": "agent:bar",
        "kind": "Agent",
        "score": 0.4,
        "confidence": 0.5,
        "valid_time": [50, None],
        "tx_time": [50, None],
        "source_refs": [],
        "evidence_refs": [],
        "policy_labels": [],
    },
]


def _fake_resolve_read_engines(engine):
    def _resolve(target):
        return ([("default", engine)], {}, False)

    return _resolve


def test_graph_query_envelope_raw_is_byte_identical(monkeypatch):
    _register_query_tools()
    engine = _FakeEngine(_ROWS, _WIRE_ROWS)
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines(engine)
    )
    expected = json.dumps(_ROWS, default=str)

    # envelope entirely unset (the pre-existing call shape) — via _execute_tool
    # so an omitted Field(default=...) resolves exactly like a real MCP call.
    default_out = asyncio.run(
        kg_server._execute_tool("graph_query", cypher="MATCH (a:Agent) RETURN a")
    )
    explicit_raw_out = asyncio.run(
        kg_server._execute_tool(
            "graph_query", cypher="MATCH (a:Agent) RETURN a", envelope="raw"
        )
    )

    assert default_out == expected
    assert explicit_raw_out == expected
    assert engine.graph.seen_ids is None  # never called on the raw path


def test_graph_query_envelope_bundle_wires_evidence_bundle_from_engine_wire(
    monkeypatch,
):
    _register_query_tools()
    engine = _FakeEngine(_ROWS, _WIRE_ROWS)
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines(engine)
    )

    out = json.loads(
        asyncio.run(
            kg_server._execute_tool(
                "graph_query",
                cypher="MATCH (a:Agent) RETURN a",
                envelope="bundle",
            )
        )
    )

    # rows are still present, unmodified.
    assert out["rows"] == _ROWS
    # explain_provenance_by_ids was actually invoked with the row ids extracted
    # from the plain Cypher result — proves the live wiring, not a stub return.
    assert sorted(engine.graph.seen_ids) == ["agent:bar", "agent:foo"]

    bundle = out["evidence_bundle"]
    # This is EvidenceBundle.from_engine_wire's REAL KnowledgeSet-row mapping
    # (not the pre-D11 passthrough) — one claim per row, confidence from the
    # top-scored row (agent:foo, score 0.9 -> confidence 0.8), and the
    # source_refs surfaced as evidence_spans.
    assert {c["id"] for c in bundle["claims"]} == {"agent:foo", "agent:bar"}
    assert bundle["confidence"] == 0.8
    assert {
        "ref": "src:gitlab/agent-foo",
        "row_id": "agent:foo",
        "type": "source_ref",
    } in (bundle["evidence_spans"])
    # `from_engine_wire` compares each row's raw `valid_time` value (a
    # `[from, until]` pair, as it arrives over msgpack) directly — min/max here
    # is a real bitemporal coverage signal over those pairs, not decomposed.
    assert bundle["freshness"]["valid_time"] == {"min": [50, None], "max": [100, None]}


def test_graph_query_envelope_bundle_degrades_cleanly_with_no_compute_surface(
    monkeypatch,
):
    """An engine with no `.graph.explain_provenance_by_ids` (or empty rows) must
    degrade to an empty bundle, never raise — the no-fabrication contract."""
    _register_query_tools()

    class _NoComputeEngine:
        def query_cypher(self, cypher, params, as_of=None):
            return _ROWS

    engine = _NoComputeEngine()
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines(engine)
    )

    out = json.loads(
        asyncio.run(
            kg_server._execute_tool(
                "graph_query",
                cypher="MATCH (a:Agent) RETURN a",
                envelope="bundle",
            )
        )
    )
    assert out["rows"] == _ROWS
    assert out["evidence_bundle"]["claims"] == []
    assert out["evidence_bundle"]["confidence"] is None
