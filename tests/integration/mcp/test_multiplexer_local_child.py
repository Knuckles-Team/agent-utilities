"""Real local-child dynamic-session and reconnect journey.

This is intentionally a subprocess integration test rather than another child
session double.  It covers the graph-os fleet loader's real stdio boundary:
the child is started from its catalog entry, its exact host-tool name is
prefixed and forwarded, and a dead generation is replaced before a retry is
returned to the caller.

CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import PaginatedRequestParams

from agent_utilities.mcp.multiplexer import (
    _LOCAL_SESSION_META_KEY,
    MCPMultiplexer,
    SessionVisibilityMiddleware,
    _register_meta_tools,
)

pytestmark = pytest.mark.integration

_SERVER_NAME = "container-manager-mcp"
_PREFIXED_TOOL = "cm__list_hosts"

_CHILD_SCRIPT = textwrap.dedent(
    """
    import asyncio
    import json
    import os
    import signal
    import sys
    from pathlib import Path

    import mcp.types as mcp_types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server
    from mcp.shared.exceptions import MCPError

    generation_file = Path(sys.argv[1])
    try:
        generation = int(generation_file.read_text(encoding="utf-8")) + 1
    except (FileNotFoundError, ValueError):
        generation = 1
    generation_file.write_text(str(generation), encoding="utf-8")

    async def terminate_first_generation() -> None:
        await asyncio.sleep(0.2)
        os.kill(os.getpid(), signal.SIGTERM)

    async def list_tools(_ctx, _params):
        return mcp_types.ListToolsResult(
            tools=[
                mcp_types.Tool(
                    name="cm_list_hosts",
                    description="Return synthetic host rows for this integration journey.",
                    inputSchema={"type": "object", "properties": {}},
                )
            ]
        )

    async def call_tool(_ctx, params):
        '''Return a bounded host fixture and retire the first child process.'''
        if params.name != "cm_list_hosts":
            raise MCPError(-32601, "Method not found")
        if generation == 1:
            asyncio.create_task(terminate_first_generation())
        return mcp_types.CallToolResult(
            content=[
                mcp_types.TextContent(
                    text=json.dumps([f"lane23-child-generation-{generation}"])
                )
            ]
        )

    mcp = Server(
        "container-manager-mcp-real-child",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )

    async def main() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await mcp.run(
                read_stream,
                write_stream,
                mcp.create_initialization_options(),
            )

    if __name__ == "__main__":
        asyncio.run(main())
    """
)


def _write_config(tmp_path: Path, script: Path, generation_file: Path) -> Path:
    config_path = tmp_path / "mcp_config.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    _SERVER_NAME: {
                        "command": sys.executable,
                        "args": [
                            str(script),
                            str(generation_file),
                        ],
                        "timeout": 10,
                        "max_restarts": 3,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return config_path


async def _list_tool_names(client: Client, session_id: str) -> set[str]:
    result = await client.session.list_tools(
        params=PaginatedRequestParams(meta={_LOCAL_SESSION_META_KEY: session_id})
    )
    return {tool.name for tool in result.tools}


@pytest.mark.timeout(90)
@pytest.mark.asyncio
async def test_real_child_host_tool_load_call_and_reconnect(tmp_path: Path) -> None:
    """Start, call, terminate, and reconnect one real stdio MCP child.

    The first child generation exits after returning its first result.  The
    second call therefore cannot succeed by replaying a mocked result:
    ``ChildRuntime`` must classify the SDK's ``MCPError("Connection closed")``,
    reconnect, initialize a new process, and retry the idempotent host read
    exactly once.
    """
    script = tmp_path / "real_container_manager_child.py"
    generation_file = tmp_path / "generation"
    script.write_text(_CHILD_SCRIPT, encoding="utf-8")
    mux = MCPMultiplexer(_write_config(tmp_path, script, generation_file))
    mcp = FastMCP("real-child-session-host")
    _register_meta_tools(mcp, mux)
    mux._global_visible = {
        "find_tools",
        "list_catalog",
        "load_tools",
        "unload_tools",
        "multiplexer_status",
    }
    mcp.add_middleware(SessionVisibilityMiddleware(mux, mcp))

    try:
        async with Client(mcp) as owner:
            cold_tools = await _list_tool_names(owner, "lane23-owner")
            assert _PREFIXED_TOOL not in cold_tools

            loaded = await owner.call_tool(
                "load_tools",
                {"tools": [_PREFIXED_TOOL]},
                meta={_LOCAL_SESSION_META_KEY: "lane23-owner"},
            )
            assert loaded.structured_content["mounted_servers"] == [_SERVER_NAME]
            assert loaded.structured_content["newly_exposed"] == [_PREFIXED_TOOL]
            assert loaded.structured_content["failed"] == {}
            assert mux.tool_to_server[_PREFIXED_TOOL] == (
                _SERVER_NAME,
                "cm_list_hosts",
            )

            owner_tools = await _list_tool_names(owner, "lane23-owner")
            assert _PREFIXED_TOOL in owner_tools
            first = await owner.call_tool(
                _PREFIXED_TOOL,
                {},
                meta={_LOCAL_SESSION_META_KEY: "lane23-owner"},
            )
            assert json.loads(first.content[0].text) == ["lane23-child-generation-1"]

            # The first process exits after sending the first result; wait for
            # the real pipe to close before forcing the second call through it.
            await asyncio.sleep(0.35)
            second = await owner.call_tool(
                _PREFIXED_TOOL,
                {},
                meta={_LOCAL_SESSION_META_KEY: "lane23-owner"},
            )
            assert json.loads(second.content[0].text) == ["lane23-child-generation-2"]
            runtime = mux.children[_SERVER_NAME]
            assert runtime.restart_count == 1
            assert runtime.generation == 2
            assert generation_file.read_text(encoding="utf-8") == "2"

            async with Client(mcp) as sibling:
                sibling_tools = await _list_tool_names(sibling, "lane23-sibling")
                assert _PREFIXED_TOOL not in sibling_tools
                with pytest.raises(ToolError, match="not loaded in this session"):
                    await sibling.call_tool(
                        _PREFIXED_TOOL,
                        {},
                        meta={_LOCAL_SESSION_META_KEY: "lane23-sibling"},
                    )
    finally:
        await mux.aclose()
