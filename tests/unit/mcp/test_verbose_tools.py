"""Backing-operation registration for the shared MCP builder (ECO-4.82).

The served surface (the intent contract) is covered by
``test_connector_surface.py``; these tests pin how an agent's client methods
and action-routed tools become operations on its private backing server.
"""

from __future__ import annotations

import asyncio

from fastmcp import FastMCP

from agent_utilities.mcp.action_dispatch import is_destructive_action
from agent_utilities.mcp.verbose_tools import (
    _build_params_json_tool,
    _build_typed_tool,
    _camel_to_snake,
    _derive_domains,
    _domain_methods,
    _tool_prefix,
    register_verbose_tools,
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


# --- A github-agent-shaped client: domain methods over a Base*-PREFIXED infra
# class. Regression: the exclusion originally matched only the *Base SUFFIX
# convention, so a prefix-named base (``BaseApiClient`` — the shared
# agent_utilities.httpsupport.client base and 16+ per-connector api_client_base.py
# modules, e.g. github-agent's ``BaseApiClient.close()``) leaked its public
# infra methods as spurious verbose tools (e.g. ``github_close``). ------------
class _BaseApiClient:
    def close(self) -> None:  # public infra on the Base* PREFIX -> excluded
        return None

    def _auth(self):  # private -> excluded
        return None


class _GithubApi(_BaseApiClient):
    def get_issue(self, **kwargs):
        "Get an issue."
        return {"op": "get_issue", "kwargs": kwargs}


def test_domain_methods_excludes_base_prefix_convention():
    owners = _domain_methods(_GithubApi)
    assert set(owners) == {"get_issue"}
    assert "close" not in owners  # infra on Base* PREFIX, not just *Base suffix
    assert "_auth" not in owners


def test_register_verbose_tools_excludes_base_prefix_convention():
    """Integration-level proof: a BaseApiClient-shaped client's verbose surface
    covers the real domain method only — no spurious ``<prefix>_close`` tool
    from the prefix-named base leaking into the surface."""
    mcp = FastMCP("t")
    names = register_verbose_tools(
        mcp, _GithubApi, lambda: _GithubApi(), service="github-agent"
    )
    assert names == ["github_get_issue"]
    assert "github_close" not in names


def test_derive_domains_camelcase_boundary():
    # Regression: char-level commonprefix must not cut a token (no stray "mdb").
    domains = _derive_domains(_domain_methods(_Api))
    assert domains["get_cmdb_instance"] == "cmdb"
    assert domains["get_change"] == "change_management"


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
    assert is_destructive_action("delete_record", None) is True
    assert is_destructive_action("get_record", None) is False
    assert is_destructive_action("get_record", {"http": "DELETE"}) is True
    assert is_destructive_action("delete_record", {"destructive": False}) is False


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
    # Missing context cannot authorize a destructive operation.
    headless = asyncio.run(_run(None))
    assert headless == {"cancelled": True, "operation": "delete_record"}


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


# --- action-routed tools: static enums and dynamic providers (ECO-4.90) ----
from typing import Literal  # noqa: E402

from pydantic import Field  # noqa: E402

from agent_utilities.mcp.verbose_tools import (  # noqa: E402
    _action_enum,
    _provider_tools,
    _resolve_action_provider,
    _tool_action_names,
)


class _FakeDynamicClient:
    """An atlassian-shaped client whose actions are discovered at runtime."""

    def get_issue(self, **kwargs):
        return {"op": "get_issue"}

    def create_issue(self, **kwargs):
        return {"op": "create_issue"}

    def delete_issue(self, **kwargs):
        return {"op": "delete_issue"}

    def _internal(self):  # private -> excluded
        return None


def test_action_enum_reads_literal_and_skips_freeform():
    """_action_enum returns enum values for a Literal action, [] for free-form str."""
    mcp = FastMCP("t")

    @mcp.tool(name="enum_tool")
    async def enum_tool(
        action: Literal["a", "b"] = Field(description="op"),
        params_json: str = Field(default="{}"),
    ) -> dict:
        return {}

    @mcp.tool(name="freeform_tool")
    async def freeform_tool(
        action: str = Field(description="op"),
        params_json: str = Field(default="{}"),
    ) -> dict:
        return {}

    tools = _provider_tools(mcp)
    assert _action_enum(tools["enum_tool"]) == ["a", "b"]
    assert _action_enum(tools["freeform_tool"]) == []


def test_resolve_action_provider_forms():
    """A provider resolves from a list, a callable, or a client class — and a
    client class is introspected credential-free (the class, no live instance)."""
    assert _resolve_action_provider(["b", "a", "a"]) == ["a", "b"]
    assert _resolve_action_provider(lambda: ["x", "y"]) == ["x", "y"]
    assert _resolve_action_provider(_FakeDynamicClient) == [
        "create_issue",
        "delete_issue",
        "get_issue",
    ]


def test_resolve_action_provider_drops_discovery_keywords():
    """Discovery keywords (list_actions/help/actions) are never real operations."""
    assert _resolve_action_provider(["get_issue", "list_actions", "help"]) == [
        "get_issue"
    ]


def test_tool_action_names_prefers_static_enum_over_provider():
    """A static Literal enum wins; the dynamic provider is the fallback only."""
    mcp = FastMCP("t")

    @mcp.tool(name="enum_tool")
    async def enum_tool(
        action: Literal["a", "b"] = Field(description="op"),
        params_json: str = Field(default="{}"),
    ) -> dict:
        return {}

    tool = _provider_tools(mcp)["enum_tool"]
    assert _tool_action_names(tool, {"enum_tool": ["x", "y"]}) == ["a", "b"]
