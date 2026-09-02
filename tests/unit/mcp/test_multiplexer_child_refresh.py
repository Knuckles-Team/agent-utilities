"""Protocol probing and governed catalog-refresh surface regressions."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import mcp.types as mcp_types
import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from agent_utilities.mcp.multiplexer import MCPMultiplexer, _register_meta_tools


def _write_config(tmp_path) -> object:
    path = tmp_path / "mcp_config.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "alpha-mcp": {"command": "python", "prefix": "alpha"},
                    "beta-mcp": {"command": "python", "prefix": "beta"},
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _resource(uri: str, description: str) -> MagicMock:
    item = MagicMock()
    item.uri = uri
    item.description = description
    return item


def _session(*, resources: list[MagicMock] | None = None) -> AsyncMock:
    session = AsyncMock()
    session.list_resources = AsyncMock(
        return_value=SimpleNamespace(resources=list(resources or []))
    )
    session.list_resource_templates = AsyncMock(
        return_value=SimpleNamespace(resource_templates=[])
    )
    session.list_prompts = AsyncMock(return_value=SimpleNamespace(prompts=[]))

    async def _read(uri):
        return SimpleNamespace(contents=[SimpleNamespace(text=f"fresh body for {uri}")])

    session.read_resource = AsyncMock(side_effect=_read)
    return session


async def test_protocol_probe_reads_each_native_catalog_family_once(tmp_path) -> None:
    mux = MCPMultiplexer(_write_config(tmp_path))
    session = _session(resources=[_resource("data://alpha", "alpha resource")])
    session.list_resource_templates = AsyncMock(
        return_value=SimpleNamespace(
            resource_templates=[
                mcp_types.ResourceTemplate(
                    name="alpha-template", uriTemplate="data://alpha/{id}"
                )
            ]
        )
    )
    session.list_prompts = AsyncMock(
        return_value=SimpleNamespace(
            prompts=[mcp_types.Prompt(name="review", description="Review data")]
        )
    )

    (
        resources,
        templates,
        prompts,
        skills,
        prompt_resources,
        errors,
    ) = await mux._probe_protocol_families("alpha-mcp", session)

    assert resources[0]["uri"] == "data://alpha"
    assert templates[0]["uriTemplate"] == "data://alpha/{id}"
    assert prompts[0]["name"] == "review"
    assert skills == []
    assert prompt_resources == []
    assert errors == {}
    session.list_resources.assert_awaited_once()
    session.list_resource_templates.assert_awaited_once()
    session.list_prompts.assert_awaited_once()


async def test_catalog_refresh_meta_tool_is_admin_only_and_uses_same_mux_method(
    tmp_path, monkeypatch
) -> None:
    mux = MCPMultiplexer(_write_config(tmp_path))
    identity = mux.catalog_identity()
    mux.refresh_catalog = AsyncMock(
        return_value=SimpleNamespace(model_dump=lambda **_: {"status": "refreshed"})
    )  # type: ignore[method-assign]
    host = FastMCP("refresh-auth-host")
    _register_meta_tools(host, mux)
    tool = await host.get_tool("catalog_refresh")

    monkeypatch.setattr(
        "agent_utilities.mcp.multiplexer._request_capabilities",
        lambda: frozenset({"mcp:delegate"}),
    )
    with pytest.raises(ToolError, match="manage capability required"):
        await tool.fn(
            request_id="refresh-1",
            expected_config_revision=identity.config_revision,
            expected_catalog_generation=identity.catalog_generation,
            expected_snapshot_digest=identity.snapshot_digest,
            deadline_ms=1000,
        )

    monkeypatch.setattr(
        "agent_utilities.mcp.multiplexer._request_capabilities",
        lambda: frozenset({"mcp:admin"}),
    )
    result = await tool.fn(
        request_id="refresh-1",
        expected_config_revision=identity.config_revision,
        expected_catalog_generation=identity.catalog_generation,
        expected_snapshot_digest=identity.snapshot_digest,
        deadline_ms=1000,
    )
    assert result.structured_content == {"status": "refreshed"}
    mux.refresh_catalog.assert_awaited_once()
