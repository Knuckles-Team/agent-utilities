"""Verbose 1:1 MCP tool surface + the MCP_TOOL_MODE knob (ECO-4.82)."""

from __future__ import annotations

import asyncio

import pytest
from fastmcp import FastMCP

from agent_utilities.mcp.verbose_tools import (
    VALID_TOOL_MODES,
    _build_params_json_tool,
    _build_typed_tool,
    _camel_to_snake,
    _derive_domains,
    _domain_methods,
    _is_destructive,
    _tool_prefix,
    register_verbose_tools,
    tool_mode,
)


# --- A fleet-shaped client: domain mixins over a *Base infra class ----------
class _ServiceNowApiBase:
    def __init__(self):
        pass

    def paginate(self, **kwargs):  # public, but infra on the Base -> excluded
        return "infra"

    def _auth(self):  # private -> excluded
        return None


class _ServiceNowApiCmdb(_ServiceNowApiBase):
    def get_cmdb_instance(self, **kwargs):
        "Return attributes for a CI record."
        return {"op": "get_cmdb_instance", "kwargs": kwargs}

    def create_cmdb_instance(self, **kwargs):
        "Create a configuration item."
        return {"op": "create_cmdb_instance", "kwargs": kwargs}


class _ServiceNowApiChangeManagement(_ServiceNowApiBase):
    def get_change(self, **kwargs):
        "Get a change request."
        return {"op": "get_change", "kwargs": kwargs}


class _Api(_ServiceNowApiCmdb, _ServiceNowApiChangeManagement):
    pass


def _get_client():
    return _Api()


def _get(mcp: FastMCP, name: str):
    return asyncio.run(mcp.get_tool(name))


def _tools_list(mcp: FastMCP):
    return asyncio.run(mcp._list_tools())


# --- helpers ----------------------------------------------------------------
def test_camel_to_snake():
    assert _camel_to_snake("ChangeManagement") == "change_management"
    assert _camel_to_snake("Cmdb") == "cmdb"


def test_tool_prefix_strips_suffix():
    assert _tool_prefix("servicenow-api") == "servicenow"
    assert _tool_prefix("gitlab-api") == "gitlab"
    assert _tool_prefix("jellyfin-mcp") == "jellyfin"


def test_domain_methods_excludes_base_and_private():
    owners = _domain_methods(_Api)
    assert set(owners) == {"get_cmdb_instance", "create_cmdb_instance", "get_change"}
    assert "paginate" not in owners  # infra on *Base
    assert "_auth" not in owners


def test_derive_domains_camelcase_boundary():
    # Regression: char-level commonprefix must not cut a token (no stray "mdb").
    domains = _derive_domains(_domain_methods(_Api))
    assert domains["get_cmdb_instance"] == "cmdb"
    assert domains["get_change"] == "change_management"


# --- tool_mode knob ---------------------------------------------------------
def test_tool_mode_default_condensed(monkeypatch):
    monkeypatch.delenv("MCP_TOOL_MODE", raising=False)
    assert tool_mode() == "condensed"


@pytest.mark.parametrize("mode", VALID_TOOL_MODES)
def test_tool_mode_valid_values(monkeypatch, mode):
    monkeypatch.setenv("MCP_TOOL_MODE", mode.upper())  # case-insensitive
    assert tool_mode() == mode


def test_tool_mode_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("MCP_TOOL_MODE", "bogus")
    assert tool_mode() == "condensed"


# --- introspection (params_json) tier ---------------------------------------
def test_register_introspection_one_tool_per_method():
    mcp = FastMCP("t")
    names = register_verbose_tools(mcp, _Api, _get_client, service="servicenow-api")
    assert set(names) == {
        "servicenow_get_cmdb_instance",
        "servicenow_create_cmdb_instance",
        "servicenow_get_change",
    }
    tool = _get(mcp, "servicenow_get_cmdb_instance")
    # docstring carried as description; tagged verbose + domain
    assert tool.description == "Return attributes for a CI record."
    assert {"verbose", "cmdb"} <= set(tool.tags)
    # params_json fallback signature
    assert list(tool.parameters["properties"]) == ["params_json"]


def test_register_introspection_no_methods_returns_empty():
    class _Empty:
        pass

    mcp = FastMCP("t")
    assert register_verbose_tools(mcp, _Empty, _get_client, service="x-api") == []


def test_introspection_live_dispatch():
    """LIVE-PATH: invoking the registered tool actually calls the client method."""
    mcp = FastMCP("t")
    register_verbose_tools(mcp, _Api, _get_client, service="servicenow-api")

    async def _run():
        tool = await mcp.get_tool("servicenow_get_cmdb_instance")
        result = await tool.run({"params_json": '{"sys_id": "abc", "blank": null}'})
        return result.structured_content

    out = asyncio.run(_run())
    # None-valued args are dropped before dispatch
    assert out == {"op": "get_cmdb_instance", "kwargs": {"sys_id": "abc"}}


# --- typed (manifest-driven) tier -------------------------------------------
_MANIFEST = [
    {
        "method": "get_cmdb_instance",
        "domain": "cmdb",
        "summary": "Return a CI record by sys_id.",
        "params": [
            {
                "name": "sys_id",
                "type": "string",
                "required": True,
                "description": "Sys ID of the CI.",
            },
            {
                "name": "className",
                "type": "string",
                "required": False,
                "description": "CMDB class name.",
            },
        ],
    },
    # an op whose method is not on the client -> skipped, not registered
    {"method": "nonexistent_op", "domain": "ghost", "params": [{"name": "x"}]},
]


def test_register_typed_from_manifest():
    mcp = FastMCP("t")
    names = register_verbose_tools(
        mcp, _Api, _get_client, service="servicenow-api", manifest=_MANIFEST
    )
    assert "servicenow_nonexistent_op" not in names  # skipped (not on client)
    typed = _get(mcp, "servicenow_get_cmdb_instance")
    schema = typed.parameters
    assert list(schema["properties"]) == ["sys_id", "className"]
    assert schema["required"] == ["sys_id"]
    assert schema["properties"]["sys_id"]["description"] == "Sys ID of the CI."
    assert typed.description == "Return a CI record by sys_id."
    # a method absent from the manifest still gets a params_json fallback tool
    assert list(_get(mcp, "servicenow_get_change").parameters["properties"]) == [
        "params_json"
    ]


def test_invalid_param_name_falls_back_to_params_json():
    """A param name that isn't a Python identifier (e.g. SCIM urn:) -> params_json."""

    class _ScimClient:
        def patch_group(self, **kwargs):
            "Patch a group."
            return {"patched": kwargs}

    manifest = [
        {
            "method": "patch_group",
            "domain": "scim",
            "summary": "Patch a SCIM group.",
            "params": [
                {"name": "id", "type": "string", "required": True},
                {
                    "name": "urn:ietf:params:scim:schemas:onetrust:Group",
                    "type": "object",
                    "required": False,
                },
            ],
        }
    ]
    mcp = FastMCP("t")
    register_verbose_tools(
        mcp,
        _ScimClient,
        lambda: _ScimClient(),
        service="onetrust-api",
        manifest=manifest,
    )
    tool = _get(mcp, "onetrust_patch_group")
    # falls back to params_json rather than crashing on the invalid identifier
    assert list(tool.parameters["properties"]) == ["params_json"]
    assert tool.description == "Patch a SCIM group."


def test_typed_live_dispatch():
    """LIVE-PATH: typed tool dispatches by-name to the client method."""
    mcp = FastMCP("t")
    register_verbose_tools(
        mcp, _Api, _get_client, service="servicenow-api", manifest=_MANIFEST
    )

    async def _run():
        tool = await mcp.get_tool("servicenow_get_cmdb_instance")
        result = await tool.run({"sys_id": "abc"})
        return result.structured_content

    out = asyncio.run(_run())
    assert out == {"op": "get_cmdb_instance", "kwargs": {"sys_id": "abc"}}


# --- Context-driven destructive elicitation ---------------------------------
class _DeleteClient:
    def delete_record(self, **kwargs):
        "Delete a record."
        return {"deleted": kwargs}


class _Elicit:
    def __init__(self, action, data=True):
        self.action = action
        self.data = data


class _FakeCtx:
    """Minimal fastmcp Context double exposing only ``elicit``."""

    def __init__(self, action):
        self._action = action
        self.asked = []

    async def elicit(self, message, response_type=bool):
        self.asked.append(message)
        return _Elicit(self._action)


def test_is_destructive_detection():
    assert _is_destructive("delete_record", None) is True
    assert _is_destructive("get_record", None) is False
    assert _is_destructive("get_record", {"http": "DELETE"}) is True
    assert _is_destructive("delete_record", {"destructive": False}) is False  # explicit


def test_destructive_params_json_tool_confirms():
    fn = _build_params_json_tool("delete_record", _DeleteClient, destructive=True)

    async def _run(ctx):
        return await fn(params_json='{"id": "x"}', client=_DeleteClient(), ctx=ctx)

    # rejected -> cancelled, method NOT called
    rejected = asyncio.run(_run(_FakeCtx("decline")))
    assert rejected == {"cancelled": True, "operation": "delete_record"}
    # accepted -> dispatched
    accepted = asyncio.run(_run(_FakeCtx("accept")))
    assert accepted == {"deleted": {"id": "x"}}
    # headless (ctx None) -> allowed by default
    headless = asyncio.run(_run(None))
    assert headless == {"deleted": {"id": "x"}}


def test_destructive_typed_tool_confirms():
    params = [{"name": "id", "type": "string", "required": True, "description": "ID."}]
    fn = _build_typed_tool("delete_record", params, _DeleteClient, destructive=True)

    async def _run(ctx):
        return await fn(id="x", client=_DeleteClient(), ctx=ctx)

    assert asyncio.run(_run(_FakeCtx("decline"))) == {
        "cancelled": True,
        "operation": "delete_record",
    }
    assert asyncio.run(_run(_FakeCtx("accept"))) == {"deleted": {"id": "x"}}


# --- register_tool_surface (central wiring) ---------------------------------
import types as _types  # noqa: E402

from agent_utilities.mcp.verbose_tools import register_tool_surface  # noqa: E402


def _surface_module():
    """A fake <pkg>.mcp module exposing register_<tag>_tools callables."""
    mod = _types.ModuleType("fake_pkg_mcp")

    def register_cmdb_tools(mcp):
        mcp.tool(name="svc_cmdb", tags={"cmdb"})(lambda: None)

    def register_change_management_tools(mcp):
        mcp.tool(name="svc_change_management", tags={"change_management"})(lambda: None)

    mod.register_cmdb_tools = register_cmdb_tools
    mod.register_change_management_tools = register_change_management_tools
    return mod


def test_surface_condensed_via_tools_module(monkeypatch):
    monkeypatch.delenv("MCP_TOOL_MODE", raising=False)  # condensed default
    mcp = FastMCP("t")
    tags = register_tool_surface(
        mcp,
        client_cls=_Api,
        get_client=_get_client,
        service="servicenow-api",
        tools_module=_surface_module(),
    )
    assert set(tags) == {"cmdb", "change_management"}
    names = {t.name for t in _tools_list(mcp)}
    assert "svc_cmdb" in names
    assert "servicenow_get_cmdb_instance" not in names  # no verbose in condensed


def test_surface_env_var_derivation_gates(monkeypatch):
    # <TAG>TOOL derived from register_<tag>_tools; CHANGE_MANAGEMENTTOOL=False disables it
    monkeypatch.delenv("MCP_TOOL_MODE", raising=False)
    monkeypatch.setenv("CHANGE_MANAGEMENTTOOL", "False")
    mcp = FastMCP("t")
    tags = register_tool_surface(
        mcp,
        client_cls=_Api,
        get_client=_get_client,
        service="servicenow-api",
        tools_module=_surface_module(),
    )
    assert tags == ["cmdb"]


def test_surface_both_adds_verbose(monkeypatch):
    monkeypatch.setenv("MCP_TOOL_MODE", "both")
    mcp = FastMCP("t")
    register_tool_surface(
        mcp,
        client_cls=_Api,
        get_client=_get_client,
        service="servicenow-api",
        tools_module=_surface_module(),
    )
    names = {t.name for t in _tools_list(mcp)}
    assert "svc_cmdb" in names  # condensed
    assert "servicenow_get_cmdb_instance" in names  # verbose


def test_surface_tool_registry(monkeypatch):
    monkeypatch.setenv("MCP_TOOL_MODE", "verbose")  # condensed registry skipped
    calls = []
    registry = [("a", "ATOOL", lambda mcp: calls.append("a"))]
    mcp = FastMCP("t")
    register_tool_surface(
        mcp,
        client_cls=_Api,
        get_client=_get_client,
        service="servicenow-api",
        tool_registry=registry,
    )
    assert calls == []  # verbose-only mode does not run condensed registry
    assert "servicenow_get_cmdb_instance" in {t.name for t in _tools_list(mcp)}
