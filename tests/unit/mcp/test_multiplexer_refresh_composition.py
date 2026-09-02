"""Wire-first composition coverage for governed MCP catalog refresh writes."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastmcp import FastMCP
from mcp import types as mcp_types

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.mcp import kg_server
from agent_utilities.mcp import shared_multiplexer as shared_mux
from agent_utilities.mcp.catalog_reconciliation import (
    CatalogContractError,
    CatalogIdentity,
    CatalogRefreshRequest,
)
from agent_utilities.mcp.multiplexer import MCPMultiplexer, attach_fleet_loader


async def test_graphos_and_rest_mux_composition_reach_one_canonical_writer(
    tmp_path, monkeypatch
) -> None:
    """Both process compositions execute the real public source-sync seam."""
    config_path = tmp_path / "mcp_config.json"
    config_path.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
    engine = object()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", engine)
    shared_mux._reset_served_multiplexer_for_tests()

    host = FastMCP("graphos-refresh-composition")
    graphos_mux = attach_fleet_loader(
        host,
        config_path=str(config_path),
        catalog_writer=kg_server._write_refreshed_fleet_catalog,
    )
    rest_mux = await shared_mux.get_served_multiplexer()

    # An authoritative empty snapshot drives the complete real canonical
    # writer (preflight, fresh authority, relational projection, promotions,
    # KG slice accounting) without requiring a live database backend.
    catalog: dict = {}
    configs: dict = {}
    bindings: dict = {}
    graphos_result = await graphos_mux._fleet_catalog_writer(catalog, configs, bindings)
    assert rest_mux is graphos_mux
    assert graphos_result["status"] == "ok"
    await graphos_mux.aclose()
    shared_mux._reset_served_multiplexer_for_tests()


def _request(mux: MCPMultiplexer) -> CatalogRefreshRequest:
    identity = mux.catalog_identity()
    return CatalogRefreshRequest(
        request_id="refresh-all-families",
        expected_config_revision=identity.config_revision,
        expected_catalog_generation=identity.catalog_generation,
        expected_snapshot_digest=identity.snapshot_digest,
        deadline_ms=1000,
    )


async def test_refresh_publishes_one_four_family_generation_after_writer_ack(
    tmp_path,
) -> None:
    config_path = tmp_path / "mcp_config.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"alpha-mcp": {"command": "alpha"}}}),
        encoding="utf-8",
    )
    mux = MCPMultiplexer(config_path)
    session = AsyncMock()
    session.list_tools = AsyncMock(
        return_value=SimpleNamespace(
            tools=[mcp_types.Tool(name="work", inputSchema={"type": "object"})]
        )
    )
    mux.children["alpha-mcp"] = SimpleNamespace(
        primary_session=session, restart_count=0
    )
    mux.probe_server = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "tools": [{"name": "work", "inputSchema": {"type": "object"}}],
            "resources": [{"uri": "data://alpha", "name": "alpha"}],
            "resource_templates": [
                {"uriTemplate": "data://alpha/{id}", "name": "alpha-template"}
            ],
            "native_prompts": [{"name": "review"}],
            "skills": [],
            "prompts": [],
            "error": None,
        }
    )
    mux._fleet_catalog_writer = AsyncMock(return_value={"status": "ok"})

    result = await mux.refresh_catalog(_request(mux))
    snapshot = mux.catalog_snapshot()

    assert result.catalog_generation == 1
    assert result.changed is True
    child = snapshot.children[0]
    assert [entry.key for entry in child.tools] == [
        f"{mux.server_prefix('alpha-mcp')}__work"
    ]
    assert [entry.key for entry in child.resources] == ["data://alpha"]
    assert [entry.key for entry in child.resource_templates] == ["data://alpha/{id}"]
    assert [entry.key for entry in child.prompts] == ["review"]
    mux._fleet_catalog_writer.assert_awaited_once()


async def test_writer_or_replica_failure_never_publishes_candidate(tmp_path) -> None:
    config_path = tmp_path / "mcp_config.json"
    config_path.write_text("{}", encoding="utf-8")
    mux = MCPMultiplexer(config_path)
    mux._fleet_catalog_writer = AsyncMock(return_value={"status": "error"})

    with pytest.raises(CatalogContractError, match="not acknowledged"):
        await mux.refresh_catalog(_request(mux))
    assert mux.catalog_identity().catalog_generation == 0

    mux._fleet_catalog_writer = AsyncMock(return_value={"status": "ok"})
    local = mux.catalog_identity()
    mux._replica_catalog_identities = lambda: [
        CatalogIdentity(
            **{
                **local.model_dump(),
                "catalog_generation": local.catalog_generation + 1,
                "snapshot_digest": "c" * 64,
            }
        )
    ]
    with pytest.raises(CatalogContractError, match="divergent"):
        await mux.refresh_catalog(_request(mux))
    assert mux.catalog_identity().catalog_generation == 0


async def test_stable_dispatch_relists_and_retries_one_read_only_unknown(
    tmp_path,
) -> None:
    config_path = tmp_path / "mcp_config.json"
    config_path.write_text(
        json.dumps(
            {"mcpServers": {"alpha-mcp": {"command": "alpha", "prefix": "alpha"}}}
        ),
        encoding="utf-8",
    )
    mux = MCPMultiplexer(config_path)
    session = AsyncMock()
    session.list_tools = AsyncMock(
        return_value=SimpleNamespace(
            tools=[mcp_types.Tool(name="work", inputSchema={"type": "object"})]
        )
    )
    mux.children["alpha-mcp"] = SimpleNamespace(
        primary_session=session, restart_count=0
    )
    mux.probe_server = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "tools": [
                {
                    "name": "work",
                    "inputSchema": {"type": "object"},
                    "annotations": {"readOnlyHint": True},
                }
            ],
            "resources": [],
            "resource_templates": [],
            "native_prompts": [],
            "skills": [],
            "prompts": [],
            "error": None,
        }
    )
    mux._fleet_catalog_writer = AsyncMock(return_value={"status": "ok"})
    await mux.refresh_catalog(_request(mux))
    identity = mux.catalog_identity()
    public_name = "alpha__work"
    mux.tool_to_server[public_name] = ("alpha-mcp", "work")
    failed = mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text="Unknown tool: work")],
        isError=True,
    )
    succeeded = mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text="ok")]
    )
    mux.call_proxied_tool = AsyncMock(  # type: ignore[method-assign]
        side_effect=[failed, succeeded]
    )

    result = await mux.dispatch_catalog_tool(
        tool_name=public_name,
        arguments={},
        expected_catalog_generation=identity.catalog_generation,
        expected_snapshot_digest=identity.snapshot_digest,
    )

    assert result is succeeded
    assert mux.call_proxied_tool.await_count == 2
    session.list_tools.assert_awaited_once()
