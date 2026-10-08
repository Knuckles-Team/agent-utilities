"""The shared MCP builder serves the one condensed intent contract (ECO-4.82).

CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse — every fleet server built
with :func:`agent_utilities.mcp.verbose_tools.register_tool_surface` lists only
the intent tools ``find``/``ask``/``act``; each takes ``action`` + ``params``
(+ optional ``intent``), and every backing operation stays reachable through
them.
"""

from __future__ import annotations

import asyncio
import json
import types
from typing import Literal

import pytest
from fastmcp import FastMCP
from pydantic import Field

from agent_utilities.mcp.connector_surface import (
    CONNECTOR_INTENT_TOOLS,
    connector_operations,
)
from agent_utilities.mcp.intent_contract import (
    DEFAULT_SURFACE_TOKEN_BUDGET,
    DEFAULT_SURFACE_TOOL_LIMIT,
    INTENT_TOOL_TOKEN_BUDGET,
    estimate_tokens,
    tool_definition,
)
from agent_utilities.mcp.verbose_tools import (
    register_tool_surface,
    register_verbose_tools,
)


class _ServiceNowApiBase:
    def paginate(self, **kwargs):  # infra on the Base -> never an operation
        return "infra"


class _ServiceNowApiCmdb(_ServiceNowApiBase):
    def get_cmdb_instance(self, **kwargs):
        "Return attributes for a CI record."
        return {"op": "get_cmdb_instance", "kwargs": kwargs}

    def create_cmdb_instance(self, **kwargs):
        "Create a configuration item."
        return {"op": "create_cmdb_instance", "kwargs": kwargs}


class _Api(_ServiceNowApiCmdb):
    pass


def _get_client():
    return _Api()


def _tools_module():
    mod = types.ModuleType("fake_pkg_mcp")

    def register_cmdb_tools(mcp):
        @mcp.tool(name="svc_cmdb", tags={"cmdb"})
        async def svc_cmdb(
            action: Literal["get_instance", "delete_instance"] = Field(
                description="op"
            ),
            params_json: str = Field(default="{}", description="args"),
        ) -> dict:
            "Manage CMDB records."
            return {"action": action, "params": json.loads(params_json or "{}")}

    def register_change_management_tools(mcp):
        @mcp.tool(name="svc_change_management", tags={"change_management"})
        async def svc_change_management(
            action: str = Field(description="op"),
            params_json: str = Field(default="{}", description="args"),
        ) -> dict:
            "Manage change requests."
            return {"action": action, "params": json.loads(params_json or "{}")}

    mod.register_cmdb_tools = register_cmdb_tools
    mod.register_change_management_tools = register_change_management_tools
    # Shared helpers imported into a connector module are never registrars.
    mod.register_verbose_tools = register_verbose_tools
    mod.register_tool_surface = register_tool_surface
    return mod


class _Changes:
    def get_change(self, **kwargs):
        return kwargs

    def update_change(self, **kwargs):
        return kwargs


def _surface(monkeypatch: pytest.MonkeyPatch | None = None) -> FastMCP:
    mcp = FastMCP("t")
    tags = register_tool_surface(
        mcp,
        client_cls=_Api,
        get_client=_get_client,
        service="servicenow-api",
        tools_module=_tools_module(),
        action_providers={"svc_change_management": _Changes},
    )
    assert set(tags) == {"cmdb", "change_management"}
    return mcp


def _call(mcp: FastMCP, tool: str, **arguments) -> dict:
    result = asyncio.run(mcp.call_tool(tool, arguments))
    payload = result.structured_content
    if isinstance(payload, dict) and set(payload) == {"result"}:
        payload = payload["result"]
    return payload


def _listed(mcp: FastMCP) -> list:
    return asyncio.run(mcp.list_tools())


def test_served_surface_is_only_the_intent_tools() -> None:
    names = [tool.name for tool in _listed(_surface())]
    assert names == list(CONNECTOR_INTENT_TOOLS)


def test_intent_tools_use_the_condensed_schema_within_budget() -> None:
    listed = _listed(_surface())
    assert len(listed) <= DEFAULT_SURFACE_TOOL_LIMIT
    total = 0
    for tool in listed:
        definition = tool_definition(tool)
        properties = definition["inputSchema"]["properties"]
        assert set(properties) == {"action", "params", "intent", "execute"}
        assert properties["params"]["type"] == "object"
        assert len(definition.get("description", "").splitlines()) <= 2
        tokens = estimate_tokens(definition)
        assert tokens <= INTENT_TOOL_TOKEN_BUDGET, (tool.name, tokens)
        total += tokens
    assert total <= DEFAULT_SURFACE_TOKEN_BUDGET


def test_every_backing_operation_is_an_addressable_operation() -> None:
    mcp = _surface()
    operations = connector_operations(mcp._intent_backing)
    assert {
        "svc_cmdb.get_instance",
        "svc_cmdb.delete_instance",
        "svc_change_management.get_change",
        "svc_change_management.update_change",
        "servicenow_get_cmdb_instance",
        "servicenow_create_cmdb_instance",
    } <= set(operations)
    assert not any("paginate" in op for op in operations)
    described = _call(mcp, "find", action="describe")
    listed = {op for ops in described["operations"].values() for op in ops}
    assert listed == set(operations)


def test_describe_serves_one_operation_schema_on_demand() -> None:
    out = _call(
        _surface(),
        "find",
        action="describe",
        params={"action": "svc_cmdb.get_instance"},
    )
    assert out["action"] == "svc_cmdb.get_instance"
    assert out["reads"] is True
    assert "action" not in out["params_schema"]["properties"]
    assert "params_json" in out["params_schema"]["properties"]


def test_find_ranks_operations_for_an_intent() -> None:
    out = _call(_surface(), "find", intent="get the cmdb instance record")
    actions = [row["action"] for row in out["results"]]
    assert "servicenow_get_cmdb_instance" in actions


def test_ask_runs_a_read_operation_with_params_shaped_onto_the_tool() -> None:
    out = _call(
        _surface(),
        "ask",
        action="svc_cmdb.get_instance",
        params={"sys_id": "abc"},
    )
    assert out == {"action": "get_instance", "params": {"sys_id": "abc"}}


def test_ask_refuses_a_mutating_operation() -> None:
    out = _call(_surface(), "ask", action="svc_cmdb.delete_instance")
    assert "act" in out["error"]


def test_act_runs_a_client_method_operation() -> None:
    out = _call(
        _surface(),
        "act",
        action="servicenow_create_cmdb_instance",
        params={"name": "web-01"},
    )
    assert out == {"op": "create_cmdb_instance", "kwargs": {"name": "web-01"}}


def test_act_without_execute_previews_the_shaped_call() -> None:
    out = _call(
        _surface(),
        "act",
        action="svc_change_management.update_change",
        params={"number": "CHG1"},
        execute=False,
    )
    assert out == {
        "executed": False,
        "action": "svc_change_management.update_change",
        "arguments": ["action", "params_json"],
    }


def test_unknown_operation_and_action_inside_params_are_rejected() -> None:
    mcp = _surface()
    assert "Unknown operation" in _call(mcp, "act", action="nope")["error"]
    with pytest.raises(Exception, match="not inside params"):
        _call(mcp, "act", action="svc_cmdb.get_instance", params={"action": "x"})


def test_per_domain_toggle_still_disables_a_backing_registrar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHANGE_MANAGEMENTTOOL", "False")
    mcp = FastMCP("t")
    tags = register_tool_surface(
        mcp, service="servicenow-api", tools_module=_tools_module()
    )
    assert tags == ["cmdb"]
    operations = connector_operations(mcp._intent_backing)
    assert not any(op.startswith("svc_change_management") for op in operations)


def test_multi_client_targets_each_contribute_operations() -> None:
    class _Sonarr:
        def get_series(self, **k):
            "List series."
            return k

    class _Radarr:
        def get_movies(self, **k):
            "List movies."
            return k

    mcp = FastMCP("t")
    register_tool_surface(
        mcp,
        service="arr-mcp",
        verbose_targets=[
            {"client_cls": _Sonarr, "get_client": _Sonarr, "tool_prefix": "sonarr"},
            {"client_cls": _Radarr, "get_client": _Radarr, "tool_prefix": "radarr"},
        ],
    )
    assert [tool.name for tool in _listed(mcp)] == list(CONNECTOR_INTENT_TOOLS)
    operations = connector_operations(mcp._intent_backing)
    assert {"sonarr_get_series", "radarr_get_movies"} <= set(operations)
