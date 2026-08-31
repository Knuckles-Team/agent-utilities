"""Wire-first composition coverage for governed MCP catalog refresh writes."""

from __future__ import annotations

import json

from fastmcp import FastMCP

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.mcp import kg_server
from agent_utilities.mcp import shared_multiplexer as shared_mux
from agent_utilities.mcp.multiplexer import attach_fleet_loader


async def test_graphos_and_rest_mux_composition_reach_one_canonical_writer(
    tmp_path, monkeypatch
) -> None:
    """Both process compositions execute the real public source-sync seam."""
    config_path = tmp_path / "mcp_config.json"
    config_path.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
    engine = object()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    monkeypatch.setattr(IntelligenceGraphEngine, "_ACTIVE_ENGINE", engine)
    monkeypatch.setattr(shared_mux, "_default_config_path", lambda: config_path)

    host = FastMCP("graphos-refresh-composition")
    graphos_mux = attach_fleet_loader(
        host,
        config_path=str(config_path),
        catalog_writer=kg_server._write_refreshed_fleet_catalog,
    )
    rest_mux = shared_mux._new_multiplexer()

    # An authoritative empty snapshot drives the complete real canonical
    # writer (preflight, fresh authority, relational projection, promotions,
    # KG slice accounting) without requiring a live database backend.
    catalog: dict = {}
    configs: dict = {}
    bindings: dict = {}
    graphos_result = await graphos_mux._fleet_catalog_writer(catalog, configs, bindings)
    rest_result = await rest_mux._fleet_catalog_writer(catalog, configs, bindings)

    assert graphos_result["status"] == "ok"
    assert graphos_result["source"] == "fleet"
    assert graphos_result["servers_seen"] == 0
    assert graphos_result["servers_written"] == 0
    assert graphos_result["relational"] == {
        "status": "skipped",
        "reason": "no engine SQL surface",
    }
    assert rest_result == graphos_result
    await graphos_mux.aclose()
    await rest_mux.aclose()
