"""Wiring contracts for the single composed MCP discovery/source-sync seam."""

from __future__ import annotations

import ast
import contextlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_utilities.knowledge_graph.core import source_sync
from agent_utilities.knowledge_graph.core.engine_mcp_discovery import (
    MCPDiscoveryMixin,
    MCPProbePort,
)
from agent_utilities.mcp import kg_server, multiplexer


class _DiscoveryEngine(MCPDiscoveryMixin):
    backend = None


def _fleet_config(path: Path) -> Path:
    path.write_text(
        '{"mcpServers":{"synthetic-child":{"command":"child"}}}',
        encoding="utf-8",
    )
    return path


def _imported_modules(module: object) -> set[str]:
    """Return static import targets for one inspected module."""
    tree = ast.parse(inspect.getsource(module))
    imported_from = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    imported_direct = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    return imported_from | imported_direct


def _is_mcp_import(name: str) -> bool:
    return (
        name == "agent_utilities.mcp"
        or name.startswith("agent_utilities.mcp.")
        or name == "mcp"
        or name.startswith("mcp.")
    )


def test_kg_probe_consumers_do_not_import_the_mcp_layer() -> None:
    """Both lower-layer consumers remain inverted at the static boundary."""
    for module in (
        inspect.getmodule(MCPDiscoveryMixin),
        source_sync,
    ):
        assert not any(map(_is_mcp_import, _imported_modules(module)))


@pytest.mark.asyncio
async def test_get_engine_binds_canonical_probe_used_by_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real GraphOS engine composition reaches the injected probe port."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.mcp.tools import data_prep_tools

    engine = _DiscoveryEngine()
    probe = AsyncMock(
        return_value={
            "tools": [
                {
                    "name": "inspect",
                    "description": "Inspect safely",
                    "inputSchema": {"type": "object"},
                }
            ],
            "error": None,
        }
    )
    monkeypatch.setattr(multiplexer.MCPMultiplexer, "probe_declaration", probe)
    monkeypatch.setattr(
        IntelligenceGraphEngine,
        "get_active",
        classmethod(lambda _cls: engine),
    )
    monkeypatch.setattr(
        data_prep_tools,
        "register_process_data_prep_runtime",
        lambda _engine: True,
    )

    composed = kg_server._get_engine()
    entry = composed.parse_mcp_config(
        {"mcpServers": {"synthetic-child": {"command": "child"}}}
    )[0]
    tools = await composed.discover_mcp_tools(entry, timeout=7.0)

    assert composed is engine
    assert isinstance(engine.mcp_probe_port, MCPProbePort)
    assert tools == [
        {
            "name": "inspect",
            "description": "Inspect safely",
            "input_schema": {"type": "object"},
        }
    ]
    probe.assert_awaited_once_with("synthetic-child", entry, timeout=7.0)


def test_fleet_sync_uses_composed_probe_and_governed_snapshot_writer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The source lifecycle consumes one probe and one governed writer seam."""
    config_path = _fleet_config(tmp_path / "mcp_config.json")
    observed: dict[str, object] = {}

    class _Mux:
        def __init__(self, path: Path) -> None:
            observed["path"] = path

        def probe_catalog(self, *, budget: float) -> object:
            observed["budget"] = budget
            return "probe-awaitable"

        def load_catalog(self) -> dict[str, dict]:
            return {"synthetic-child": {"command": "child"}}

        def _bind_local_discovery_bindings(self, _catalog: dict) -> None:
            return None

        def _take_discovery_bindings(self, _catalog: dict) -> dict:
            return {}

    catalog = {
        "synthetic-child": {
            "tools": [{"name": "inspect", "description": "Inspect safely"}],
            "skills": [],
            "prompts": [],
            "error": None,
        }
    }
    port = MCPProbePort(
        probe_declaration=AsyncMock(),
        resolve_config_path=lambda _explicit: config_path,
        multiplexer_factory=_Mux,
        run_async=lambda awaitable, *, timeout: (
            observed.update(awaitable=awaitable, timeout=timeout) or catalog
        ),
    )
    captured: dict[str, object] = {}

    def write_snapshot(
        engine: object,
        received: dict,
        *,
        configs: dict | None,
        discovery_bindings: dict | None,
    ) -> dict[str, object]:
        captured.update(
            engine=engine,
            catalog=received,
            configs=configs,
            discovery_bindings=discovery_bindings,
        )
        return {"status": "ok", "source": "fleet", "servers_written": 1}

    monkeypatch.setattr(source_sync, "write_fleet_catalog_snapshot", write_snapshot)
    monkeypatch.setattr(source_sync, "_fleet_probe_budget", lambda: 10.0)
    monkeypatch.setattr(source_sync, "_fresh_write_authority", contextlib.nullcontext)
    # NOTE: `_reconcile_declared_fleet` (the declared-fleet-registry coverage
    # reconcile) was removed from source_sync.py by this candidate — its
    # `declared_total`/`declared_uncovered` accounting is no longer part of
    # `_sync_fleet`'s result, so there is nothing left here to monkeypatch.
    engine = SimpleNamespace(mcp_probe_port=port)

    result = source_sync._sync_fleet(engine)

    assert result["status"] == "ok"
    assert captured == {
        "engine": engine,
        "catalog": catalog,
        "configs": {"synthetic-child": {"command": "child"}},
        "discovery_bindings": {},
    }
    assert observed == {
        "path": config_path,
        "budget": 10.0,
        "awaitable": "probe-awaitable",
        "timeout": 25.0,
    }


def test_fleet_sync_without_composed_probe_is_explicitly_unavailable() -> None:
    result = source_sync._sync_fleet(SimpleNamespace())

    assert result == {
        "status": "unavailable",
        "source": "fleet",
        "reason": "mcp fleet probe is unavailable",
    }
