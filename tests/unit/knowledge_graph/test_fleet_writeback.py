"""Tests for governed fleet write-back actions (CONCEPT:KG-2.42).

Exercises the reusable ``call_tool_once`` write helper against an in-process
FastMCP server (the write-side twin of the KG-2.59 source connector) and asserts
``fleet.write_record`` is registered as a governed EXTERNAL action.
"""

from __future__ import annotations

import json

import pytest
from fastmcp import FastMCP

from agent_utilities.knowledge_graph.actions import DEFAULT_REGISTRY, fleet_writeback
from agent_utilities.knowledge_graph.actions.fleet_writeback import FLEET_WRITE_RECORD
from agent_utilities.knowledge_graph.actions.models import ActionEffect
from agent_utilities.protocols.source_connectors.connectors.mcp_tool import (
    call_tool_once,
)


def _fake_fleet_server():
    server = FastMCP("fake-servicenow-mcp")
    calls: list = []

    @server.tool
    def servicenow_table_api(action: str, params_json: str = "{}") -> dict:
        calls.append((action, json.loads(params_json)))
        return {"result": {"sys_id": "abc", "state": "6"}}

    return server, calls


@pytest.mark.concept("KG-2.42")
async def test_call_tool_once_invokes_injected_fleet_tool():
    server, calls = _fake_fleet_server()
    result = await call_tool_once(
        client=server,
        tool="servicenow_table_api",
        action="patch_table_record",
        params={"table": "incident", "sys_id": "abc", "state": "6"},
    )
    # fleet convention: action routed separately, params JSON-encoded into params_json
    assert calls == [
        ("patch_table_record", {"table": "incident", "sys_id": "abc", "state": "6"})
    ]
    assert result["result"]["state"] == "6"


@pytest.mark.concept("KG-2.42")
def test_fleet_write_record_registered_as_external_action():
    action = DEFAULT_REGISTRY.get("fleet.write_record")
    assert action is FLEET_WRITE_RECORD
    assert action.produces_effect == ActionEffect.EXTERNAL
    assert action.idempotent is False
    assert DEFAULT_REGISTRY.get_handler("fleet.write_record") is not None


@pytest.mark.concept("KG-2.42")
def test_handler_threads_params_and_wraps_result(monkeypatch):
    captured: dict = {}

    async def _fake_call(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(fleet_writeback, "call_tool_once", _fake_call)
    out = fleet_writeback._handle_fleet_write_record(
        {
            "server": "servicenow-mcp",
            "tool": "servicenow_table_api",
            "action": "patch_table_record",
            "params": {"sys_id": "abc"},
        }
    )
    assert captured == {
        "server": "servicenow-mcp",
        "tool": "servicenow_table_api",
        "action": "patch_table_record",
        "params": {"sys_id": "abc"},
    }
    assert out == {
        "server": "servicenow-mcp",
        "tool": "servicenow_table_api",
        "action": "patch_table_record",
        "result": {"ok": True},
    }
