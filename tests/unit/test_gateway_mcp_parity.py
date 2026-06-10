"""Contract test: the MCP tool surface and the REST gateway must stay in lockstep.

Both surfaces dispatch through ``_execute_tool`` against the same in-process
engine, so ``_execute_tool`` is the single source of truth. This test enforces
that every action-routed MCP tool in ``REGISTERED_TOOLS`` has exactly one
collapsed REST twin (via ``ACTION_TOOL_ROUTES``) and that the twin is actually
mounted by ``_mount_rest_routes`` — and vice versa. If a new MCP tool is added
without a REST route (or a route is added for a non-existent tool), this fails,
preventing the two surfaces from drifting.
"""

from __future__ import annotations

from agent_utilities.mcp import kg_server


class _RecordingApp:
    """Minimal Starlette-compatible app that records mounted routes."""

    def __init__(self) -> None:
        self.routes: list[tuple[str, list[str]]] = []

    def add_route(self, path, handler, methods=None) -> None:  # noqa: ANN001
        self.routes.append((path, list(methods or [])))


def _mounted_paths(prefix: str = "") -> set[str]:
    kg_server.ensure_tools_registered()
    app = _RecordingApp()
    kg_server._mount_rest_routes(app, prefix=prefix)
    return {path for path, _methods in app.routes}


def test_every_mcp_tool_has_a_rest_route():
    kg_server.ensure_tools_registered()
    tools = set(kg_server.REGISTERED_TOOLS)
    mapped = set(kg_server.ACTION_TOOL_ROUTES)

    missing = tools - mapped
    assert not missing, (
        f"MCP tools with no REST twin in ACTION_TOOL_ROUTES: {sorted(missing)}. "
        "Add a collapsed REST route so the gateway reaches every MCP capability."
    )


def test_no_phantom_routes_for_missing_tools():
    kg_server.ensure_tools_registered()
    tools = set(kg_server.REGISTERED_TOOLS)
    mapped = set(kg_server.ACTION_TOOL_ROUTES)

    phantom = mapped - tools
    assert not phantom, (
        f"ACTION_TOOL_ROUTES references tools not in REGISTERED_TOOLS: {sorted(phantom)}."
    )


def test_mapped_routes_are_actually_mounted():
    paths = _mounted_paths()
    for tool, path in kg_server.ACTION_TOOL_ROUTES.items():
        assert path in paths, (
            f"Tool '{tool}' maps to '{path}' but that route is not mounted by "
            "_mount_rest_routes."
        )


def test_prefix_is_applied_to_mounted_routes():
    # The API gateway mounts these under /api — confirm the prefix is honoured.
    paths = _mounted_paths(prefix="/api")
    for path in kg_server.ACTION_TOOL_ROUTES.values():
        assert ("/api" + path) in paths
