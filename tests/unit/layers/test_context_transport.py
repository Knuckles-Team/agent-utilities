"""A harness run can bind only its verified, authenticated EG MCP URL."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_utilities.layers.context_transport import (
    ContextEndpointUnavailable,
    bind_context_toolset,
)
from agent_utilities.layers.contracts import McpEndpoint


def test_bind_context_toolset_uses_exact_url_and_secret_ref(
    monkeypatch: Any,
) -> None:
    from agent_utilities.layers import context_transport
    from agent_utilities.mcp import toolset_factory

    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        context_transport,
        "resolve_runtime_secret_reference",
        lambda ref: "opaque-token" if ref == "env://GRAPHOS_BEARER" else "",
    )

    def build(url: str, **kwargs: Any) -> object:
        seen.update(url=url, **kwargs)
        return object()

    monkeypatch.setattr(toolset_factory, "build_http_toolset", build)
    endpoint = McpEndpoint(
        name="graphos",
        url="https://graphos.example/mcp",
        bearer_ref="env://GRAPHOS_BEARER",
    )
    bound = bind_context_toolset(endpoint)
    assert bound is not None
    assert seen["url"] == endpoint.url
    assert seen["toolset_id"] == endpoint.name
    assert seen["auth"].reference == endpoint.bearer_ref


@pytest.mark.parametrize(
    "endpoint",
    [
        McpEndpoint(name="graphos", url="https://graphos.example/mcp"),
        McpEndpoint(
            name="graphos", url="http://graphos.example/mcp", bearer_ref="env://TOKEN"
        ),
        McpEndpoint(
            name="graphos",
            url="https://graphos.example/other",
            bearer_ref="env://TOKEN",
        ),
    ],
)
def test_context_transport_refuses_unverified_connection(endpoint: McpEndpoint) -> None:
    with pytest.raises(ContextEndpointUnavailable):
        bind_context_toolset(endpoint)


def test_verified_context_dispatch_has_no_fleet_fallback(monkeypatch: Any) -> None:
    from agent_utilities.orchestration import agent_runner

    seen: dict[str, Any] = {}

    async def direct(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"results": {"output": "grounded"}}

    monkeypatch.setattr(agent_runner, "_execute_single_server", direct)
    state: dict[str, Any] = {}
    config = {"verified_context_endpoint": "graphos", "mcp_toolsets": [object()]}
    result = asyncio.run(
        agent_runner._select_and_dispatch(
            state,
            "auto",
            {"type": "agent_template"},
            config,
            None,
            "Inspect",
            "job-one",
            3,
            "package-agent",
            None,
            None,
            None,
        )
    )
    assert result["results"]["output"] == "grounded"
    assert seen["config"] is config
    assert seen["bound_tool_grounding"] is True
    assert state["stage_reached"] == "verified-context-endpoint"
