"""Provider-neutral live child-policy coverage for the MCP multiplexer."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import mcp.types
import pytest
from fastmcp.exceptions import ToolError

from agent_utilities.mcp import multiplexer as mod
from agent_utilities.mcp.multiplexer import MCPMultiplexer


class _Policy:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.closed = False

    def transport_config(self):
        return {"command": "/preinstalled/helper", "args": []}

    def child_environment(self):
        return {"PROVIDER_CONFIGS": "{}"}

    def verify_before_spawn(self):
        self.events.append("verify")

    def allows_tool(self, _name, annotations):
        return annotations.get("readOnlyHint") is True

    def fingerprint_catalog(self, _tools):
        self.events.append("fingerprint")
        return "a" * 64

    def close(self):
        self.events.append("policy-close")
        self.closed = True


class _Session:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def initialize(self):
        self.events.append("initialize")

    async def list_tools(self):
        self.events.append("list")
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="read_inventory",
                    description="read",
                    inputSchema={"type": "object"},
                    annotations={"readOnlyHint": True},
                    meta=None,
                ),
                SimpleNamespace(
                    name="write_inventory",
                    description="write",
                    inputSchema={"type": "object"},
                    annotations={"readOnlyHint": False},
                    meta=None,
                ),
            ]
        )

    async def call_tool(self, name, _arguments):
        self.events.append(f"call:{name}")
        return mcp.types.CallToolResult(content=[])


class _SessionContext:
    def __init__(self, events: list[str]) -> None:
        self.session = _Session(events)

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *_exc):
        return False


@pytest.mark.asyncio
async def test_runtime_policy_owns_spawn_catalog_call_and_teardown(
    monkeypatch, tmp_path
):
    events: list[str] = []
    policy = _Policy(events)

    def factory(*, profile_name, config):
        assert profile_name == "deployment-profile"
        assert config is not None
        return policy

    monkeypatch.setattr(
        mod,
        "_load_runtime_child_policy_factory",
        lambda name: factory if name == "official-provider" else None,
    )

    @contextlib.asynccontextmanager
    async def stdio_client(params, *, errlog):
        assert params.command == "/preinstalled/helper"
        assert params.env["PROVIDER_CONFIGS"] == "{}"
        assert errlog is not None
        events.append("spawn")
        try:
            yield ("read", "write")
        finally:
            events.append("child-close")

    monkeypatch.setattr(mod, "stdio_client", stdio_client)
    monkeypatch.setattr(
        mod,
        "ClientSession",
        lambda *_args, **_kwargs: _SessionContext(events),
    )

    mux = MCPMultiplexer(tmp_path / "mcp_config.json")
    declaration = {
        "runtime_policy": "official-provider",
        "provider_profile": "deployment-profile",
    }
    mux._catalog = {"hosted-enterprise-architecture": declaration}
    result = await mux._start_child(
        "hosted-enterprise-architecture",
        declaration,
    )
    assert result is not None
    server_name, runtime, tools, cfg = result
    assert [tool.name for tool in tools] == ["read_inventory"]
    assert events[:4] == ["verify", "spawn", "initialize", "list"]
    assert events[4] == "fingerprint"

    registered = mux._register_child_result(server_name, runtime, tools, cfg)
    assert len(registered) == 1
    await mux.call_proxied_tool(registered[0].name, {})
    assert "call:read_inventory" in events

    mux.tool_to_server["hosted__write"] = (server_name, "write_inventory")
    with pytest.raises(ToolError, match="not admitted"):
        await mux.call_proxied_tool("hosted__write", {})

    status = mux.status_snapshot()["children"][server_name]
    assert status["catalog_fingerprint"] == "a" * 64
    await mux.aclose()
    assert events.index("child-close") < events.index("policy-close")
    assert policy.closed is True


@pytest.mark.asyncio
async def test_runtime_policy_rejects_persistent_transport_overrides(
    monkeypatch, tmp_path
):
    called = False

    def load(_name):
        nonlocal called
        called = True

    monkeypatch.setattr(mod, "_load_runtime_child_policy_factory", load)
    mux = MCPMultiplexer(tmp_path / "mcp_config.json")

    result = await mux._start_child(
        "hosted-enterprise-architecture",
        {
            "runtime_policy": "official-provider",
            "provider_profile": "deployment-profile",
            "command": "inline-command",
        },
    )

    assert result is None
    assert called is False
