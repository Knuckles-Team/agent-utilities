"""Validate gateway results with the isolated official MCP 2.x SDK models."""

from __future__ import annotations

import importlib.metadata

import pytest

_MCP_MAJOR = int(importlib.metadata.version("mcp").split(".", 1)[0])
pytestmark = pytest.mark.skipif(
    _MCP_MAJOR < 2,
    reason="The main GraphOS environment intentionally uses MCP 1.x.",
)


@pytest.mark.asyncio
async def test_core_results_validate_with_official_mcp_v2_models() -> None:
    import mcp.types as mcp_types

    from mcp_v2_gateway.gateway import (
        MCP_V2_PROTOCOL_VERSION,
        GatewayRequestContext,
        GraphOSV2Gateway,
    )

    class Downstream:
        async def list_tools(
            self, _context: GatewayRequestContext
        ) -> dict[str, object]:
            return {
                "tools": [
                    {
                        "name": "echo",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                        },
                    }
                ]
            }

        async def call_tool(
            self,
            _name: str,
            _arguments: dict[str, object],
            _context: GatewayRequestContext,
        ) -> dict[str, object]:
            return {
                "content": [{"type": "text", "text": "ok"}],
                "isError": False,
            }

    gateway = GraphOSV2Gateway(Downstream())
    context = GatewayRequestContext(authorization="Bearer synthetic-token")
    meta = {
        "io.modelcontextprotocol/protocolVersion": MCP_V2_PROTOCOL_VERSION,
        "io.modelcontextprotocol/clientCapabilities": {},
    }

    discovered = await gateway.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "server/discover",
            "params": {"_meta": meta},
        },
        context=context,
    )
    listed = await gateway.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {"_meta": meta},
        },
        context=context,
    )
    called = await gateway.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "_meta": meta,
                "name": "echo",
                "arguments": {"text": "hello"},
            },
        },
        context=context,
    )

    for model_name, response in (
        ("DiscoverResult", discovered),
        ("ListToolsResult", listed),
        ("CallToolResult", called),
    ):
        getattr(mcp_types, model_name).model_validate(response["result"])
