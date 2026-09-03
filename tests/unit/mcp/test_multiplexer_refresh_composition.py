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


async def _echo_ack(
    _catalog, _configs, _bindings, *, catalog_generation, snapshot_digest
):
    """Stand-in writer that acknowledges exactly the generation/digest it was
    called with — the only shape ``_write_catalog_candidate`` now accepts as
    a genuine reingestion receipt (blocker: reingestion must bind + ack the
    precise generation + digest it ingested, not a generic ``status: "ok"``)."""
    return {
        "status": "ok",
        "catalog_generation": catalog_generation,
        "snapshot_digest": snapshot_digest,
    }


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
    mux._fleet_catalog_writer = AsyncMock(side_effect=_echo_ack)

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

    mux._fleet_catalog_writer = AsyncMock(side_effect=_echo_ack)
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
    mux._fleet_catalog_writer = AsyncMock(side_effect=_echo_ack)
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


async def _stable_dispatch_mux(
    tmp_path, *, relisted_schema: dict
) -> tuple[MCPMultiplexer, str, object]:
    """Shared setup for the unknown-tool-retry bound/schema-proof tests below."""
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
            tools=[mcp_types.Tool(name="work", inputSchema=relisted_schema)]
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
    mux._fleet_catalog_writer = AsyncMock(side_effect=_echo_ack)
    await mux.refresh_catalog(_request(mux))
    public_name = "alpha__work"
    mux.tool_to_server[public_name] = ("alpha-mcp", "work")
    return mux, public_name, session


async def test_stable_dispatch_never_retries_a_relisted_tool_with_a_changed_schema(
    tmp_path,
) -> None:
    """A name match alone is not proof the retry is safe: a relisted tool
    whose `inputSchema` changed must NOT be blindly retried with the same
    arguments — that fails closed exactly like a still-missing name."""
    mux, public_name, session = await _stable_dispatch_mux(
        tmp_path, relisted_schema={"type": "object", "required": ["new_field"]}
    )
    identity = mux.catalog_identity()
    failed = mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text="Unknown tool: work")],
        isError=True,
    )
    mux.call_proxied_tool = AsyncMock(return_value=failed)  # type: ignore[method-assign]

    result = await mux.dispatch_catalog_tool(
        tool_name=public_name,
        arguments={},
        expected_catalog_generation=identity.catalog_generation,
        expected_snapshot_digest=identity.snapshot_digest,
    )

    assert result is failed
    # Only the original call — the schema mismatch must forbid the retry.
    assert mux.call_proxied_tool.await_count == 1
    session.list_tools.assert_awaited_once()


async def test_stable_dispatch_retry_budget_is_bounded_per_child_generation(
    tmp_path,
) -> None:
    """Total unknown-tool retries against one child generation are FINITE:
    once `_UNKNOWN_TOOL_RETRY_MAX_PER_GENERATION` is spent, further dispatch
    calls that keep missing get no more relist-and-retry — proving the retry
    is bounded, not merely "one per call" with no aggregate cap."""
    from agent_utilities.mcp.multiplexer import _UNKNOWN_TOOL_RETRY_MAX_PER_GENERATION

    mux, public_name, session = await _stable_dispatch_mux(
        tmp_path, relisted_schema={"type": "object"}
    )
    identity = mux.catalog_identity()
    failed = mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text="Unknown tool: work")],
        isError=True,
    )
    succeeded = mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text="ok")]
    )
    # Every dispatch call: the primary attempt misses, the retry succeeds —
    # so exactly one retry is spent from the budget per dispatch call.
    mux.call_proxied_tool = AsyncMock(  # type: ignore[method-assign]
        side_effect=[failed, succeeded] * (_UNKNOWN_TOOL_RETRY_MAX_PER_GENERATION + 1)
    )

    for _ in range(_UNKNOWN_TOOL_RETRY_MAX_PER_GENERATION):
        result = await mux.dispatch_catalog_tool(
            tool_name=public_name,
            arguments={},
            expected_catalog_generation=identity.catalog_generation,
            expected_snapshot_digest=identity.snapshot_digest,
        )
        assert result is succeeded

    # The budget is now exhausted: the NEXT dispatch call's miss gets no retry.
    result = await mux.dispatch_catalog_tool(
        tool_name=public_name,
        arguments={},
        expected_catalog_generation=identity.catalog_generation,
        expected_snapshot_digest=identity.snapshot_digest,
    )
    assert result is failed
    assert (
        mux.call_proxied_tool.await_count
        == 2 * _UNKNOWN_TOOL_RETRY_MAX_PER_GENERATION + 1
    )
