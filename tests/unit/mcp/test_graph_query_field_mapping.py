"""LANE 9 — `POST /api/graph/query` field-name mapping and blind-splat regression tests.

`graph_query_endpoint` (`agent_utilities/mcp/kg_server.py`) used to forward the raw
request body straight into `_execute_tool("graph_query", **body)`. The tool's real
parameter is `cypher`, not `query`, so any client sending `{"query": "..."}` — the
natural wire name, and the genuine field name of the sibling `execute_cypher` route
used by `CypherReplView.tsx`/`TemporalGraphView.tsx`/`GraphView.tsx` — got a raw
`UnsupportedToolFieldError` surfaced as an opaque 500.

The fix maps `query` (and `cypher`) onto the tool's real `cypher` parameter at the
REST boundary, rejects an ambiguous both-supplied-and-different body deterministically,
and stops blind-splatting the body at all: only the tool's documented fields are ever
forwarded, so a genuinely unknown field is a clean, immediate 4xx that never reaches
`_execute_tool`.
"""

from unittest.mock import AsyncMock, patch

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from agent_utilities.mcp.kg_server import graph_query_endpoint


@pytest.fixture
def client():
    app = Starlette()
    app.add_route("/graph/query", graph_query_endpoint, methods=["POST"])
    return TestClient(app)


@pytest.mark.asyncio
@patch("agent_utilities.mcp.kg_server._execute_tool", new_callable=AsyncMock)
async def test_query_field_maps_to_cypher(mock_execute_tool, client):
    """A caller sending the natural `query` wire field succeeds (LANE 9 defect)."""
    mock_execute_tool.return_value = {"rows": []}

    res = client.post("/graph/query", json={"query": "MATCH (n) RETURN count(n)"})

    assert res.status_code == 200
    assert res.json() == {"status": "success", "result": {"rows": []}}
    mock_execute_tool.assert_awaited_once_with(
        "graph_query", cypher="MATCH (n) RETURN count(n)"
    )


@pytest.mark.asyncio
@patch("agent_utilities.mcp.kg_server._execute_tool", new_callable=AsyncMock)
async def test_cypher_field_still_works(mock_execute_tool, client):
    """The tool's real `cypher` field keeps working unmodified (no regression)."""
    mock_execute_tool.return_value = {"rows": []}

    res = client.post("/graph/query", json={"cypher": "MATCH (n) RETURN count(n)"})

    assert res.status_code == 200
    assert res.json() == {"status": "success", "result": {"rows": []}}
    mock_execute_tool.assert_awaited_once_with(
        "graph_query", cypher="MATCH (n) RETURN count(n)"
    )


@pytest.mark.asyncio
@patch("agent_utilities.mcp.kg_server._execute_tool", new_callable=AsyncMock)
async def test_query_and_cypher_identical_is_not_ambiguous(mock_execute_tool, client):
    """Both fields supplied with the SAME value is not an ambiguity error."""
    mock_execute_tool.return_value = {"rows": []}

    res = client.post(
        "/graph/query",
        json={"query": "MATCH (n) RETURN n", "cypher": "MATCH (n) RETURN n"},
    )

    assert res.status_code == 200
    mock_execute_tool.assert_awaited_once_with("graph_query", cypher="MATCH (n) RETURN n")


@pytest.mark.asyncio
@patch("agent_utilities.mcp.kg_server._execute_tool", new_callable=AsyncMock)
async def test_query_and_cypher_conflict_is_a_deterministic_4xx(mock_execute_tool, client):
    """Both fields supplied with DIFFERENT values is a loud, deterministic client error —
    never a silent pick of one over the other."""
    res = client.post(
        "/graph/query",
        json={"query": "MATCH (a) RETURN a", "cypher": "MATCH (b) RETURN b"},
    )

    assert res.status_code == 400
    body = res.json()
    assert body["status"] == "error"
    assert "query" in body["message"] and "cypher" in body["message"]
    mock_execute_tool.assert_not_awaited()


@pytest.mark.asyncio
@patch("agent_utilities.mcp.kg_server._execute_tool", new_callable=AsyncMock)
async def test_unknown_field_is_a_clean_4xx_not_a_500(mock_execute_tool, client):
    """A genuinely unrecognized field is a clean 4xx, not a 500."""
    res = client.post(
        "/graph/query",
        json={"cypher": "MATCH (n) RETURN n", "dialect": "neo4j"},
    )

    assert res.status_code == 400
    body = res.json()
    assert body["status"] == "error"
    assert "dialect" in body["message"]
    mock_execute_tool.assert_not_awaited()


@pytest.mark.asyncio
@patch("agent_utilities.mcp.kg_server._execute_tool", new_callable=AsyncMock)
async def test_does_not_blind_splat_unknown_kwargs_into_execute_tool(
    mock_execute_tool, client
):
    """Regression pin: the endpoint must not forward arbitrary body keys into
    `_execute_tool` unfiltered. Even a field that merely LOOKS plausible (not one of
    the tool's documented parameters, and not the `query` alias) must never reach the
    tool dispatch at all."""
    res = client.post(
        "/graph/query",
        json={"cypher": "MATCH (n) RETURN n", "extra_unexpected_field": "x"},
    )

    assert res.status_code == 400
    mock_execute_tool.assert_not_awaited()


@pytest.mark.asyncio
@patch("agent_utilities.mcp.kg_server._execute_tool", new_callable=AsyncMock)
async def test_documented_fields_all_pass_through(mock_execute_tool, client):
    """Every one of the tool's real documented parameters still reaches `_execute_tool`
    (the allowlist fix must not drop any legitimate field)."""
    mock_execute_tool.return_value = {"rows": []}

    body = {
        "cypher": "MATCH (n) RETURN n",
        "params": '{"id": 1}',
        "scope": "local",
        "reference_id": "ref-1",
        "as_of": "2026-01-01T00:00:00Z",
        "connection": "default",
        "graph": "primary",
        "include_epistemic": True,
    }
    res = client.post("/graph/query", json=body)

    assert res.status_code == 200
    mock_execute_tool.assert_awaited_once_with("graph_query", **body)


@pytest.mark.asyncio
@patch("agent_utilities.mcp.kg_server._execute_tool", new_callable=AsyncMock)
async def test_unsupported_tool_field_error_maps_to_4xx_not_500(mock_execute_tool, client):
    """Defense-in-depth: even if `_execute_tool` itself raises
    `UnsupportedToolFieldError` (e.g. the allowlist above ever drifts from the tool's
    real signature), the endpoint still returns a deterministic 4xx, not a 500."""
    from agent_utilities.mcp.kg_server import UnsupportedToolFieldError

    mock_execute_tool.side_effect = UnsupportedToolFieldError(
        "Tool 'graph_query' does not accept field(s): bogus."
    )

    res = client.post("/graph/query", json={"cypher": "MATCH (n) RETURN n"})

    assert res.status_code == 400
    assert res.json()["status"] != "success"
