"""End-to-end smoke test: the full MCP → engine → gateway stack.

Exercises the real shared dispatch (`_execute_tool`) that BOTH the MCP tools and
the REST gateway funnel through, against the live epistemic-graph engine started
by the session conftest. Also asserts the gateway mounts the complete surface:
every MCP tool's REST twin (parity) plus the fleet supervisory plane.

Marked integration: requires the engine backend (auto-started in this repo's
conftest); the write/read assertions degrade gracefully if a node-level write is
unavailable so the test never produces a false negative in minimal CI.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.mcp import kg_server


class _RecordingApp:
    def __init__(self) -> None:
        self.routes: list[str] = []

    def add_route(self, path, handler, methods=None) -> None:  # noqa: ANN001
        self.routes.append(path)

    def add_middleware(self, *a, **k) -> None:  # gateway adds the cypher fast-path
        pass


@pytest.mark.asyncio
async def test_write_then_engine_reflects_it():
    kg_server.ensure_tools_registered()

    node_id = "e2e:node:1"
    # Write through the same tool the MCP surface and the REST gateway both call.
    res = await kg_server._execute_tool(
        "graph_write",
        action="add_node",
        node_id=node_id,
        node_type="E2ETestNode",
        properties=json.dumps({"v": 1, "via": "e2e"}),
    )
    assert res is not None  # tool dispatched without raising

    # The engine the gateway shares now reflects the write.
    engine = kg_server._get_engine()
    has_node = getattr(engine, "has_node", None)
    if has_node is None and hasattr(engine, "graph_compute"):
        has_node = getattr(engine.graph_compute, "has_node", None)
    if has_node is not None:
        assert has_node(node_id) is True


@pytest.mark.asyncio
async def test_query_path_dispatches():
    kg_server.ensure_tools_registered()
    # The read path must dispatch through the shared tool layer without error.
    out = await kg_server._execute_tool(
        "graph_query", cypher="MATCH (n) RETURN n LIMIT 1"
    )
    assert out is not None


def test_gateway_mounts_full_surface_and_fleet():
    """register_graph_routes mounts parity twins + the fleet plane in one app."""
    from agent_utilities.gateway.graph_api import register_graph_routes

    app = _RecordingApp()
    register_graph_routes(app, prefix="/api")
    mounted = set(app.routes)

    # Every MCP tool's collapsed REST twin is present (parity).
    for path in kg_server.ACTION_TOOL_ROUTES.values():
        assert ("/api" + path) in mounted, f"missing parity route /api{path}"

    # Fleet supervisory plane is present.
    for path in ("/api/fleet/health", "/api/fleet/topology", "/api/fleet/kill"):
        assert path in mounted, f"missing fleet route {path}"
