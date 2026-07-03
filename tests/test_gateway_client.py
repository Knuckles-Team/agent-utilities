"""Tests for the surface-facing gateway client SDK (CONCEPT:ECO-4.37).

Exercises the real :class:`GatewayClient` against a mock gateway transport so the
envelope-unwrapping, the enhanced/graph/fleet methods, and the SSE stream parser
are all verified end-to-end (not just the helper in isolation).
"""

from __future__ import annotations

import json

import httpx
import pytest

from agent_utilities.gateway_client import GatewayClient


def _json(payload: object, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def _make_gateway() -> GatewayClient:
    """A GatewayClient wired to a deterministic mock gateway."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/enhanced/agents":
            return _json({"status": "ok", "agents": [{"name": "planner"}]})
        if path == "/api/enhanced/maintenance/status":
            return _json({"maintenance_required": False})
        if path == "/api/enhanced/commands/autocomplete":
            assert request.url.params.get("query") == "/he"
            return _json({"suggestions": ["/help", "/health"]})
        if path == "/api/enhanced/commands/execute":
            body = json.loads(request.content)
            return _json(
                {"response_markdown": f"ran {body['command']}", "client_actions": []}
            )
        if path == "/api/graph/query":
            return _json({"rows": [{"n": 1}]})
        if path == "/api/fleet/approvals":
            return _json({"approvals": [{"id": "a1", "action": "restart"}]})
        if path == "/api/fleet/approvals/grant":
            body = json.loads(request.content)
            return _json({"granted": body["approval_id"]})
        if path == "/stream":
            lines = (
                'data: {"type": "thought", "thought": "thinking"}\n\n'
                "data: not-json\n\n"
                'data: {"type": "final_output", "content": "done"}\n\n'
            )
            return httpx.Response(200, text=lines)
        return _json({"error": "not found"}, status=404)

    return GatewayClient("http://gw.test", transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_list_agents_unwraps_envelope() -> None:
    async with _make_gateway() as gw:
        agents = await gw.list_agents()
    assert agents == [{"name": "planner"}]


@pytest.mark.asyncio
async def test_maintenance_and_autocomplete() -> None:
    async with _make_gateway() as gw:
        status = await gw.maintenance_status()
        suggestions = await gw.autocomplete("/he")
    assert status == {"maintenance_required": False}
    assert suggestions == ["/help", "/health"]


@pytest.mark.asyncio
async def test_execute_command_shape() -> None:
    async with _make_gateway() as gw:
        result = await gw.execute_command("/status")
    assert result == {"result": "ran /status", "client_actions": []}


@pytest.mark.asyncio
async def test_graph_query() -> None:
    async with _make_gateway() as gw:
        out = await gw.graph_query("MATCH (n) RETURN n")
    assert out == {"rows": [{"n": 1}]}


@pytest.mark.asyncio
async def test_fleet_approvals_and_grant() -> None:
    async with _make_gateway() as gw:
        pending = await gw.fleet_approvals()
        granted = await gw.grant_approval("a1")
    assert pending == [{"id": "a1", "action": "restart"}]
    assert granted == {"granted": "a1"}


@pytest.mark.asyncio
async def test_stream_parses_sse_and_skips_malformed() -> None:
    events = []
    async with _make_gateway() as gw:
        async for event in gw.stream("hello"):
            events.append(event)
    # The malformed ``data: not-json`` line is skipped; the two valid ones survive.
    assert [e["type"] for e in events] == ["thought", "final_output"]
    assert events[-1]["content"] == "done"
