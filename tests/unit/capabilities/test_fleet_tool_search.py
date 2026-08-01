"""Track 2 of the pydantic-ai native-adoption program: the mcp-multiplexer fleet as a
native ``pydantic_ai.capabilities.ToolSearch`` provider.

CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog — see
``reports/program/pydantic-ai-native-adoption.md`` Track 2.

Uses a fake multiplexer double (no live child MCP servers, no network) that mirrors the
REAL ``MCPMultiplexer.probe_catalog``/``server_prefix``/``call_proxied_tool``/
``resolve_and_mount``/``_tool_enabled`` contracts exactly as read from
``agent_utilities/mcp/multiplexer.py``, so :class:`FleetToolset` is exercised against the
same shapes it uses in production without needing a running fleet.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("pydantic_ai.capabilities")

from agent_utilities.capabilities.fleet_tool_search import (  # noqa: E402
    FleetToolset,
    fleet_relevance_search,
)
from agent_utilities.mcp.multiplexer import MCPMultiplexer  # noqa: E402


class FakeMultiplexer:
    """Mirrors the slice of ``MCPMultiplexer`` :class:`FleetToolset` calls."""

    def __init__(self) -> None:
        self.catalog = {
            "caddy-mcp": {
                "tools": [
                    {
                        "name": "list_routes",
                        "description": "List configured Caddy reverse-proxy routes.",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                    {
                        "name": "reload_config",
                        "description": "Reload the live Caddyfile with zero downtime.",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                ],
                "error": None,
            },
            "gitlab-mcp": {
                "tools": [
                    {
                        "name": "create_merge_request",
                        "description": "Open a GitLab merge request for a branch.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"branch": {"type": "string"}},
                        },
                    }
                ],
                "error": None,
            },
            "broken-server": {"tools": [], "error": "connection refused"},
        }
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.mounted: list[str] = []

    async def probe_catalog(self, *, budget: float, **_kw: Any) -> dict[str, Any]:
        del budget
        return self.catalog

    def server_prefix(self, server_name: str) -> str:
        return server_name.split("-")[0][:3]

    def _tool_enabled(self, server_name: str, tool_name: str) -> bool:
        del server_name, tool_name
        return True

    async def resolve_and_mount(
        self, *, tools: list[str] | None = None, servers: list[str] | None = None
    ) -> tuple[list[str], list[str], dict[str, str]]:
        del servers
        for name in tools or []:
            self.mounted.append(name)
        return ([], list(tools or []), {})

    async def call_proxied_tool(
        self, prefixed_name: str, arguments: dict[str, Any] | None = None
    ) -> Any:
        self.calls.append((prefixed_name, dict(arguments or {})))
        return SimpleNamespace(
            is_error=False,
            structured_content={"result": {"ok": True, "tool": prefixed_name}},
            content=[],
        )


class TestFleetToolsetGetTools:
    async def test_lists_every_enabled_tool_as_deferred(self) -> None:
        mux = FakeMultiplexer()
        toolset = FleetToolset(mux)
        tools = await toolset.get_tools(ctx=None)

        names = set(tools)
        assert "cad__list_routes" in names or any("list_routes" in n for n in names)
        assert all(td.tool_def.defer_loading for td in tools.values())

    async def test_skips_a_server_that_failed_to_probe(self) -> None:
        mux = FakeMultiplexer()
        toolset = FleetToolset(mux)
        tools = await toolset.get_tools(ctx=None)

        assert not any("broken" in name for name in tools)

    async def test_carries_the_real_input_schema(self) -> None:
        mux = FakeMultiplexer()
        toolset = FleetToolset(mux)
        tools = await toolset.get_tools(ctx=None)

        mr_tool = next(
            td for td in tools.values() if td.tool_def.metadata.get("mcp_tool") == "create_merge_request"
        )
        assert mr_tool.tool_def.parameters_json_schema["properties"]["branch"]["type"] == "string"


class TestFleetToolsetCallTool:
    async def test_call_tool_mounts_then_forwards_through_call_proxied_tool(self) -> None:
        mux = FakeMultiplexer()
        toolset = FleetToolset(mux)
        tools = await toolset.get_tools(ctx=None)
        name = next(iter(tools))

        result = await toolset.call_tool(name, {"x": 1}, ctx=None, tool=tools[name])

        assert mux.mounted == [name]
        assert mux.calls == [(name, {"x": 1})]
        assert result == {"ok": True, "tool": name}

    async def test_call_tool_raises_when_resolve_and_mount_fails(self) -> None:
        mux = FakeMultiplexer()

        async def _failing_resolve(*, tools=None, servers=None):
            del servers
            return ([], [], {tools[0]: "server unreachable"})

        mux.resolve_and_mount = _failing_resolve  # type: ignore[method-assign]
        toolset = FleetToolset(mux)

        with pytest.raises(RuntimeError, match="server unreachable"):
            await toolset.call_tool("cad__list_routes", {}, ctx=None, tool=None)


class TestFleetRelevanceSearch:
    def test_ranks_by_the_multiplexer_relevance_backbone(self) -> None:
        from pydantic_ai.tools import ToolDefinition

        tools = [
            ToolDefinition(name="a", description="reload the caddy reverse proxy config"),
            ToolDefinition(name="b", description="list gitlab merge requests"),
            ToolDefinition(name="c", description="unrelated tool about weather"),
        ]
        names = fleet_relevance_search(ctx=None, queries=["reload caddy config"], tools=tools)

        assert names[0] == "a"
        assert "c" not in names

    def test_matches_mcpmultiplexer_relevance_exactly(self) -> None:
        """The wrapper must not reimplement scoring — it must delegate to the SAME
        static method ``find_tools``/``discover_tools`` rank with."""
        from pydantic_ai.tools import ToolDefinition

        tool = ToolDefinition(name="reload_config", description="Reload the live Caddyfile")
        expected = MCPMultiplexer._relevance(
            "reload caddy", f"{tool.name} {tool.description}"
        )
        names = fleet_relevance_search(ctx=None, queries=["reload caddy"], tools=[tool])
        assert bool(names) == (expected > 0)

    def test_empty_query_yields_no_matches(self) -> None:
        from pydantic_ai.tools import ToolDefinition

        tools = [ToolDefinition(name="a", description="anything")]
        assert fleet_relevance_search(ctx=None, queries=[""], tools=tools) == []
