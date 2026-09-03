"""Governed UQL support on the canonical `graph_query` MCP/REST surface.

The tests intentionally use small engine doubles: the production contract is
that `graph_query(scope='uql')` calls `IntelligenceGraphEngine.uql` and
never reaches a backend/client object's raw UQL method.  The engine method is
the authority for parser execution and row ACL/owner filtering; this boundary
adds only the bounded read-only LIMIT contract and EvidenceBundle projection.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import engine_tools

pytestmark = pytest.mark.concept("AU-KG.query.au-engine-execution-path")


def _register_graph_query():
    from fastmcp import FastMCP

    from agent_utilities.mcp.tools.query_tools import register_query_tools

    register_query_tools(FastMCP("test"))
    return kg_server.REGISTERED_TOOLS["graph_query"]


class _FakeCompute:
    def __init__(self, wire_rows):
        self.wire_rows = wire_rows
        self.seen_ids: list[str] = []

    def explain_provenance_by_ids(self, ids):
        self.seen_ids = list(ids)
        return self.wire_rows


class _FakeEngine:
    def __init__(self, rows, wire_rows=None):
        self.rows = rows
        self.graph = _FakeCompute(wire_rows or [])
        self.seen: list[tuple[str, bool]] = []

    def uql(self, query, include_epistemic=False):
        self.seen.append((query, include_epistemic))
        return self.rows


def _resolve_read_engines(*entries):
    def _resolve(_target):
        return (list(entries), {}, len(entries) > 1)

    return _resolve


_UQL = "MATCH (:Agent) |> LIMIT 5"
_ROWS = [{"id": "agent:foo", "score": 0.9}]
_ROWS_B = [{"id": "agent:b", "score": 0.7}]
_WIRE_ROWS = [
    {
        "id": "agent:foo",
        "kind": "Agent",
        "score": 0.9,
        "confidence": 0.8,
        "valid_time": [100, None],
        "tx_time": [100, None],
        "source_refs": ["src:fixture"],
        "evidence_refs": [],
        "policy_labels": [],
    }
]
_WIRE_ROWS_A = [
    {
        "id": "agent:a",
        "kind": "Agent",
        "score": 0.8,
        "confidence": 0.7,
        "valid_time": [95, None],
        "tx_time": [95, None],
        "source_refs": ["src:first"],
        "evidence_refs": [],
        "policy_labels": [],
    }
]
_WIRE_ROWS_B = [
    {
        "id": "agent:b",
        "kind": "Agent",
        "score": 0.7,
        "confidence": 0.6,
        "valid_time": [90, None],
        "tx_time": [90, None],
        "source_refs": ["src:second"],
        "evidence_refs": [],
        "policy_labels": [],
    }
]


def test_uql_scope_uses_governed_engine_method_and_bundle(monkeypatch):
    graph_query = _register_graph_query()
    engine = _FakeEngine(_ROWS, _WIRE_ROWS)
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _resolve_read_engines(("default", engine))
    )

    out = graph_query(query=_UQL, scope="uql").model_dump()

    assert engine.seen == [(_UQL, False)]
    assert engine.graph.seen_ids == ["agent:foo"]
    assert out["claims"][0]["id"] == "agent:foo"
    assert out["confidence"] == 0.8
    assert out["reasoning_trace"][-1]["payload"]["rows"] == _ROWS


def test_uql_include_epistemic_is_threaded_to_governed_method(monkeypatch):
    graph_query = _register_graph_query()
    engine = _FakeEngine([{"id": "agent:foo", "confidence": 0.8}])
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _resolve_read_engines(("default", engine))
    )

    out = graph_query(
        query=_UQL, scope="uql", include_epistemic=True, params="{}"
    ).model_dump()

    assert engine.seen == [(_UQL, True)]
    assert out["claims"] == [{"id": "agent:foo", "confidence": 0.8}]
    assert out["error"] is None


def test_uql_rejects_unbounded_or_mutating_requests_before_engine(monkeypatch):
    graph_query = _register_graph_query()
    engine = _FakeEngine(_ROWS)
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _resolve_read_engines(("default", engine))
    )

    for query in (
        "MATCH (:Agent)",
        "REASON Agent |> LIMIT 5",
        "MATCH (:Agent) |> LIMIT 1001",
        "MATCH (:Agent) |> SET owner='x' |> LIMIT 5",
        "MATCH (:Agent) |> LIMIT 5 |> WHERE score > 0",
    ):
        out = graph_query(query=query, scope="uql", params="{}").model_dump()
        assert out["error"]["code"] == "invalid_request"
    assert engine.seen == []


def test_uql_limit_bound_is_intentionally_inclusive_and_bounded(monkeypatch):
    graph_query = _register_graph_query()
    engine = _FakeEngine(_ROWS)
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _resolve_read_engines(("default", engine))
    )

    for query in (
        "MATCH (:Agent) |> LIMIT 1",
        "MATCH (:Agent) |> LIMIT 1000",
    ):
        assert graph_query(query=query, scope="uql", params="{}").error is None

    for query in (
        "MATCH (:Agent) |> LIMIT 0",
        "MATCH (:Agent) |> LIMIT 1001",
    ):
        out = graph_query(query=query, scope="uql", params="{}").model_dump()
        assert out["error"]["code"] == "invalid_request"

    assert engine.seen == [
        ("MATCH (:Agent) |> LIMIT 1", False),
        ("MATCH (:Agent) |> LIMIT 1000", False),
    ]


def test_uql_lexical_guard_does_not_reject_mutation_word_in_literal(monkeypatch):
    graph_query = _register_graph_query()
    engine = _FakeEngine(_ROWS)
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _resolve_read_engines(("default", engine))
    )

    query = "MATCH (:Agent) |> WHERE note = 'DELETE' |> LIMIT 5"
    out = graph_query(query=query, scope="uql", params="{}").model_dump()

    assert out["error"] is None
    assert engine.seen == [(query, False)]


def test_uql_rejects_unused_query_parameters_before_engine(monkeypatch):
    graph_query = _register_graph_query()
    engine = _FakeEngine(_ROWS)
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _resolve_read_engines(("default", engine))
    )

    out = graph_query(
        query=_UQL, scope="uql", params='{"agent_id": "agent:foo"}'
    ).model_dump()

    assert out["error"]["code"] == "invalid_request"
    assert engine.seen == []


@pytest.mark.parametrize("params", ["{", "", "[]", "1", None, 3])
def test_uql_params_fail_closed_with_stable_invalid_request(monkeypatch, params):
    graph_query = _register_graph_query()
    engine = _FakeEngine(_ROWS)
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _resolve_read_engines(("default", engine))
    )

    out = graph_query(query=_UQL, scope="uql", params=params).model_dump()

    assert out["error"]["code"] == "invalid_request"
    assert out["next_actions"] == ["review the structured error and retry"]
    assert engine.seen == []


def test_uql_rejects_top_level_as_of_before_engine(monkeypatch):
    graph_query = _register_graph_query()
    engine = _FakeEngine(_ROWS)
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _resolve_read_engines(("default", engine))
    )

    out = graph_query(
        query=_UQL,
        scope="uql",
        params="{}",
        as_of="2026-01-01T00:00:00Z",
    ).model_dump()

    assert out["error"]["code"] == "invalid_request"
    assert out["next_actions"] == ["review the structured error and retry"]
    assert engine.seen == []


def test_uql_external_connection_without_surface_fails_clearly(monkeypatch):
    graph_query = _register_graph_query()

    class _CypherOnlyConnection:
        def query_cypher(self, query, params):
            raise AssertionError("UQL must not fall back to external Cypher")

    monkeypatch.setattr(
        kg_server,
        "_resolve_read_engines",
        _resolve_read_engines(("teradata", _CypherOnlyConnection())),
    )

    out = graph_query(
        query=_UQL, scope="uql", connection="teradata", params="{}"
    ).model_dump()

    assert out["error"]["code"] == "dependency_unavailable"
    # The public EvidenceBundle intentionally exposes only the stable typed
    # error contract.  The internal exception class is retained in the
    # server-side diagnostic record, not promoted to a bundle field or used as
    # a client-facing assertion.
    assert out["next_actions"] == ["review the structured error and retry"]


def test_uql_fanout_preserves_targets_errors_and_per_target_evidence(monkeypatch):
    graph_query = _register_graph_query()
    first = _FakeEngine([{"id": "agent:a", "score": 0.8}], _WIRE_ROWS_A)
    second = _FakeEngine(_ROWS_B, _WIRE_ROWS_B)

    class _NoUqlConnection:
        pass

    monkeypatch.setattr(
        kg_server,
        "_resolve_read_engines",
        _resolve_read_engines(
            ("one", first), ("two", second), ("postgres", _NoUqlConnection())
        ),
    )

    out = graph_query(
        query=_UQL, scope="uql", connection="all", params="{}"
    ).model_dump()
    payload = out["reasoning_trace"][-1]["payload"]

    assert payload["targets"] == {
        "one": [{"id": "agent:a", "score": 0.8}],
        "two": [{"id": "agent:b", "score": 0.7}],
    }
    assert payload["errors"] == {"postgres": "uql_surface_unavailable"}
    assert {claim["id"] for claim in out["claims"]} == {"agent:a", "agent:b"}


def test_uql_fanout_all_unsupported_targets_returns_typed_failure(monkeypatch):
    graph_query = _register_graph_query()

    class _NoUqlConnection:
        pass

    monkeypatch.setattr(
        kg_server,
        "_resolve_read_engines",
        _resolve_read_engines(
            ("postgres", _NoUqlConnection()), ("teradata", _NoUqlConnection())
        ),
    )

    out = graph_query(
        query=_UQL, scope="uql", connection="all", params="{}"
    ).model_dump()
    payload = out["reasoning_trace"][-1]["payload"]

    assert out["error"]["code"] == "dependency_unavailable"
    assert payload["targets"] == {}
    assert payload["errors"] == {
        "postgres": "uql_surface_unavailable",
        "teradata": "uql_surface_unavailable",
    }


def test_uql_fanout_all_failed_targets_returns_typed_failure(monkeypatch):
    graph_query = _register_graph_query()

    class _FailingEngine:
        def uql(self, query, include_epistemic=False):
            raise RuntimeError("engine unavailable")

    monkeypatch.setattr(
        kg_server,
        "_resolve_read_engines",
        _resolve_read_engines(("one", _FailingEngine()), ("two", _FailingEngine())),
    )

    out = graph_query(
        query=_UQL, scope="uql", connection="all", params="{}"
    ).model_dump()

    assert out["error"]["code"] == "dependency_unavailable"
    assert out["reasoning_trace"][-1]["payload"]["targets"] == {}


def test_uql_rank_text_uses_shared_au_preembedding(monkeypatch):
    graph_query = _register_graph_query()
    engine = _FakeEngine(_ROWS)
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _resolve_read_engines(("default", engine))
    )
    monkeypatch.setattr(
        engine_tools,
        "_embed_texts",
        lambda texts: [[0.1, 0.2] for _ in texts],
    )

    query = 'MATCH (:Agent) |> RANK BY ~"agents" |> LIMIT 5'
    out = graph_query(query=query, scope="uql", params="{}").model_dump()

    assert out["error"] is None
    assert engine.seen == [
        (
            "MATCH (:Agent) |> RANK BY ~[0.10000000,0.20000000] |> LIMIT 5",
            False,
        )
    ]


def test_uql_rest_twin_keeps_scope_and_route_contract():
    kwargs = kg_server._graph_query_request_kwargs(
        {"query": _UQL, "scope": "uql", "params": "{}"}
    )
    retired = kg_server._graph_query_request_kwargs({"cypher": _UQL, "scope": "uql"})

    assert kwargs == {"query": _UQL, "scope": "uql", "params": "{}"}
    assert retired[1] == 400
    assert "cypher" in retired[0]["message"]
    assert kg_server.ACTION_TOOL_ROUTES["graph_query"] == "/graph/query"


@pytest.mark.asyncio
async def test_uql_rest_endpoint_dispatches_scope_to_same_tool(monkeypatch):
    body = json.dumps({"query": _UQL, "scope": "uql", "params": "{}"}).encode()
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/graph/query",
            "headers": [],
            "query_string": b"",
        },
        receive,
    )
    execute_tool = AsyncMock(return_value={"rows": []})
    monkeypatch.setattr(kg_server, "_execute_tool", execute_tool)

    response = await kg_server.graph_query_endpoint(request)

    assert response.status_code == 200
    assert json.loads(response.body)["result"] == {"rows": []}
    execute_tool.assert_awaited_once_with(
        "graph_query", query=_UQL, scope="uql", params="{}"
    )
