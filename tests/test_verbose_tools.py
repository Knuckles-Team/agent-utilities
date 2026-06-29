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


def test_surface_discovery_excludes_shared_helpers(monkeypatch):
    """Auto-discovery must not treat imported register_verbose_tools/
    register_tool_surface (which match register_*_tools) as connector registrars."""
    monkeypatch.delenv("MCP_TOOL_MODE", raising=False)
    mod = _surface_module()
    # simulate the helpers being imported into the agent module namespace
    mod.register_verbose_tools = register_verbose_tools
    mod.register_tool_surface = register_tool_surface
    mcp = FastMCP("t")
    tags = register_tool_surface(
        mcp,
        client_cls=_Api,
        get_client=_get_client,
        service="servicenow-api",
        tools_module=mod,
    )
    assert set(tags) == {"cmdb", "change_management"}  # helpers excluded
    assert "verbose" not in tags and "tool_surface" not in tags


def test_surface_verbose_targets_multiclient(monkeypatch):
    """A multi-client agent gets a verbose surface per target, all mode-gated centrally."""
    monkeypatch.setenv("MCP_TOOL_MODE", "verbose")

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
            {
                "client_cls": _Sonarr,
                "get_client": lambda: _Sonarr(),
                "tool_prefix": "sonarr",
            },
            {
                "client_cls": _Radarr,
                "get_client": lambda: _Radarr(),
                "tool_prefix": "radarr",
            },
        ],
    )
    names = {t.name for t in _tools_list(mcp)}
    assert "sonarr_get_series" in names
    assert "radarr_get_movies" in names


def test_surface_condensed_fallback_when_no_verbose_target(monkeypatch):
    """Verbose mode with no verbose target falls back to condensed (never empty)."""
    monkeypatch.setenv("MCP_TOOL_MODE", "verbose")
    mcp = FastMCP("t")
    tags = register_tool_surface(mcp, service="x", tools_module=_surface_module())
    # No client/verbose_targets -> condensed registers so the server isn't empty.
    assert set(tags) == {"cmdb", "change_management"}
    assert {t.name for t in _tools_list(mcp)} == {"svc_cmdb", "svc_change_management"}


def test_surface_condensed_only_server_survives_global_verbose(monkeypatch):
    """A condensed-only server (no verbose target) keeps its tools under a
    deployment-wide MCP_TOOL_MODE=verbose — never left empty (graph-os case)."""
    monkeypatch.setenv("MCP_TOOL_MODE", "verbose")
    calls = []
    mcp = FastMCP("t")
    tags = register_tool_surface(
        mcp,
        service="graph-os",
        registrars=[
            ("query", "QUERYTOOL", lambda m: calls.append("query")),
            ("ontology", "ONTOLOGYTOOL", lambda m: calls.append("ontology")),
        ],
    )
    assert set(tags) == {
        "query",
        "ontology",
    }  # condensed fell back on (no verbose target)
    assert calls == ["query", "ontology"]


def test_surface_verbose_register_hook(monkeypatch):
    """verbose_register builds a custom 1:1 surface (graph-os action-core case)."""
    seen = {}

    def _custom_verbose(mcp):
        seen["called"] = True
        mcp.tool(name="gx_write_add_node", tags={"verbose", "graph_write"})(
            lambda: None
        )

    # verbose mode: condensed skipped, custom verbose runs (not left empty)
    monkeypatch.setenv("MCP_TOOL_MODE", "verbose")
    mcp = FastMCP("t")
    tags = register_tool_surface(
        mcp,
        service="graph-os",
        tools_module=_surface_module(),
        verbose_register=_custom_verbose,
    )
    assert tags == []  # condensed NOT force-added — a verbose surface exists
    assert seen.get("called") is True
    assert "gx_write_add_node" in {t.name for t in _tools_list(mcp)}

    # both mode: condensed AND custom verbose
    seen.clear()
    monkeypatch.setenv("MCP_TOOL_MODE", "both")
    mcp2 = FastMCP("t")
    tags2 = register_tool_surface(
        mcp2,
        service="graph-os",
        tools_module=_surface_module(),
        verbose_register=_custom_verbose,
    )
    assert set(tags2) == {"cmdb", "change_management"}
    assert seen.get("called") is True


def test_surface_stamps_domain_tag_and_records_exact_toggle(monkeypatch):
    """register_tool_surface stamps the canonical domain tag on each condensed tool
    (standardizing ad-hoc author tags) and records the exact tool->toggle map."""
    import types

    monkeypatch.delenv("MCP_TOOL_MODE", raising=False)
    mod = types.ModuleType("fake_mcp_mod")

    def register_observability_tools(mcp):
        # author mis-tagged by service name, not domain
        mcp.tool(name="langfuse_observability", tags={"langfuse"})(lambda: None)

    mod.register_observability_tools = register_observability_tools

    mcp = FastMCP("t")
    register_tool_surface(mcp, service="langfuse-agent", tools_module=mod)

    tool = asyncio.run(mcp.get_tool("langfuse_observability"))
    assert "observability" in tool.tags  # canonical domain tag stamped
    # exact toggle map matches the gating env var (not the service-name guess)
    assert mcp._condensed_tool_toggles["langfuse_observability"] == "OBSERVABILITYTOOL"


# --- Fleet-wide verbose auto-wire from condensed action enums (ECO-4.89) -----
from typing import Literal  # noqa: E402

from pydantic import Field  # noqa: E402

from agent_utilities.mcp.verbose_tools import (  # noqa: E402
    _action_enum,
    autowire_verbose_from_condensed,
)


def _action_routed_module():
    """A fake <pkg>.mcp module of condensed *action-enum* tools (atlassian-shaped).

    Each register_<tag>_tools adds one tool whose ``action`` is a ``Literal`` enum
    plus a ``params_json`` passthrough — the universal condensed shape the auto-wire
    expands one verbose tool per action from.
    """
    mod = _types.ModuleType("fake_action_mcp")

    def register_issue_tools(mcp):
        @mcp.tool(name="svc_issue", tags={"issue"})
        async def svc_issue(
            action: Literal["get_issue", "create_issue", "delete_issue"] = Field(
                description="op"
            ),
            params_json: str = Field(default="{}", description="args"),
        ) -> dict:
            "Manage issues."
            return {"action": action, "params_json": params_json}

    def register_comment_tools(mcp):
        @mcp.tool(name="svc_comment", tags={"comment"})
        async def svc_comment(
            action: Literal["add_comment", "list_comments"] = Field(description="op"),
            params_json: str = Field(default="{}", description="args"),
        ) -> dict:
            "Manage comments."
            return {"action": action, "params_json": params_json}

    mod.register_issue_tools = register_issue_tools
    mod.register_comment_tools = register_comment_tools
    return mod


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

    from agent_utilities.mcp.verbose_tools import _provider_tools

    tools = _provider_tools(mcp)
    assert _action_enum(tools["enum_tool"]) == ["a", "b"]
    assert _action_enum(tools["freeform_tool"]) == []


def test_autowire_one_verbose_tool_per_action():
    """The core guarantee: one dispatching verbose tool per action enum value,
    derived from already-registered condensed action-routed tools (no per-connector
    edits). Verifies the named offline contract for ECO-4.89."""
    mcp = FastMCP("t")
    mod = _action_routed_module()
    mod.register_issue_tools(mcp)
    mod.register_comment_tools(mcp)

    derived = autowire_verbose_from_condensed(mcp)
    assert set(derived) == {
        "svc_issue__get_issue",
        "svc_issue__create_issue",
        "svc_issue__delete_issue",
        "svc_comment__add_comment",
        "svc_comment__list_comments",
    }
    # one verbose tool per action present on the server, tagged for slicing
    tool = asyncio.run(mcp.get_tool("svc_issue__create_issue"))
    assert "verbose" in tool.tags and "issue" in tool.tags
    # the verbose tool hides `action` (preset) and keeps `params_json` passthrough
    props = tool.parameters.get("properties", {})
    assert "action" not in props
    assert "params_json" in props


def test_autowire_dispatches_to_condensed_handler_with_action_preset():
    """A derived verbose tool routes to the SAME condensed handler with its action
    preset — not a re-implementation."""
    mcp = FastMCP("t")
    mod = _action_routed_module()
    mod.register_issue_tools(mcp)
    autowire_verbose_from_condensed(mcp)

    tool = asyncio.run(mcp.get_tool("svc_issue__delete_issue"))
    res = asyncio.run(tool.run({"params_json": '{"id": 7}'}))
    assert res.structured_content == {
        "action": "delete_issue",
        "params_json": '{"id": 7}',
    }


def test_autowire_is_idempotent():
    """Re-running the auto-wire derives nothing new (won't double-register or
    expand its own verbose tools)."""
    mcp = FastMCP("t")
    _action_routed_module().register_issue_tools(mcp)
    first = autowire_verbose_from_condensed(mcp)
    assert first
    assert autowire_verbose_from_condensed(mcp) == []


def test_surface_both_autowires_condensed_action_tools(monkeypatch):
    """register_tool_surface in `both` mode auto-wires verbose tools from condensed
    action-routed tools with no client_cls/verbose_targets (the atlassian case)."""
    monkeypatch.setenv("MCP_TOOL_MODE", "both")
    mcp = FastMCP("t")
    register_tool_surface(
        mcp, service="atlassian-agent", tools_module=_action_routed_module()
    )
    names = {t.name for t in _tools_list(mcp)}
    assert "svc_issue" in names  # condensed kept
    assert "svc_issue__create_issue" in names  # verbose auto-derived
    assert "svc_comment__add_comment" in names


def test_surface_autowire_opt_out(monkeypatch):
    """autowire_condensed=False opts a server out of the action-enum expansion."""
    monkeypatch.setenv("MCP_TOOL_MODE", "both")
    mcp = FastMCP("t")
    register_tool_surface(
        mcp,
        service="atlassian-agent",
        tools_module=_action_routed_module(),
        autowire_condensed=False,
    )
    names = {t.name for t in _tools_list(mcp)}
    assert "svc_issue" in names  # condensed kept
    assert "svc_issue__create_issue" not in names  # no verbose derived


# --- Dynamic (runtime) action enumeration (ECO-4.90) ------------------------
from agent_utilities.mcp.verbose_tools import (  # noqa: E402
    _resolve_action_provider,
    _tool_action_names,
    register_action_provider,
)


class _FakeDynamicClient:
    """An atlassian-shaped client whose actions are discovered at runtime.

    Mirrors a real connector: free-form ``action: str`` dispatched via
    ``getattr(client, action)``; the valid names come from ``public_actions`` on
    the client (not a static Literal). Private/dunder attrs are excluded.
    """

    def get_issue(self, **kwargs):
        return {"op": "get_issue"}

    def create_issue(self, **kwargs):
        return {"op": "create_issue"}

    def delete_issue(self, **kwargs):
        return {"op": "delete_issue"}

    def _internal(self):  # private -> excluded
        return None


def _freeform_action_module():
    """A condensed connector module whose ``action`` is a free-form ``str``.

    This is the atlassian shape that the static-enum auto-wire SKIPS: there is no
    Literal, so the action list must come from a registered dynamic-action
    provider. The handler dispatches ``getattr(client, action)``.
    """
    mod = _types.ModuleType("fake_freeform_mcp")
    client = _FakeDynamicClient()

    def register_jira_tools(mcp):
        @mcp.tool(name="atlassian_jira", tags={"jira"})
        async def atlassian_jira(
            action: str = Field(description="op"),
            params_json: str = Field(default="{}", description="args"),
        ) -> dict:
            "Manage jira via free-form action dispatch."
            return {"action": action, "params_json": params_json}

    mod.register_jira_tools = register_jira_tools
    mod._client = client
    return mod


def test_resolve_action_provider_forms():
    """A provider resolves from a list, a callable, or a client class — and a
    client class is introspected credential-free (the class, no live instance)."""
    assert _resolve_action_provider(["b", "a", "a"]) == ["a", "b"]
    assert _resolve_action_provider(lambda: ["x", "y"]) == ["x", "y"]
    # client CLASS (not instance) introspected; private methods excluded
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

    from agent_utilities.mcp.verbose_tools import _provider_tools

    tool = _provider_tools(mcp)["enum_tool"]
    # provider present, but static enum takes precedence
    assert _tool_action_names(tool, {"enum_tool": ["x", "y"]}) == ["a", "b"]


def test_autowire_enumerates_dynamic_actions_via_provider():
    """The headline ECO-4.90 fix: a free-form ``action: str`` condensed tool (no
    Literal) still gets one verbose tool per runtime action, sourced from a
    registered action provider (a client class)."""
    mcp = FastMCP("t")
    mod = _freeform_action_module()
    mod.register_jira_tools(mcp)

    # without a provider the free-form tool is skipped (the OLD behavior / the bug)
    assert autowire_verbose_from_condensed(mcp) == []

    # register the dynamic action surface, then auto-wire derives 1:1 tools
    register_action_provider(mcp, "atlassian_jira", _FakeDynamicClient)
    derived = autowire_verbose_from_condensed(mcp)
    assert set(derived) == {
        "atlassian_jira__get_issue",
        "atlassian_jira__create_issue",
        "atlassian_jira__delete_issue",
    }
    tool = asyncio.run(mcp.get_tool("atlassian_jira__create_issue"))
    assert "verbose" in tool.tags and "jira" in tool.tags
    props = tool.parameters.get("properties", {})
    assert "action" not in props  # preset+hidden
    assert "params_json" in props  # passthrough


def test_autowire_dynamic_dispatches_with_action_preset():
    """A dynamic-derived verbose tool routes to the SAME condensed handler with
    its action preset — same passthrough guarantee as the static path."""
    mcp = FastMCP("t")
    mod = _freeform_action_module()
    mod.register_jira_tools(mcp)
    register_action_provider(mcp, "atlassian_jira", _FakeDynamicClient)
    autowire_verbose_from_condensed(mcp)

    tool = asyncio.run(mcp.get_tool("atlassian_jira__delete_issue"))
    res = asyncio.run(tool.run({"params_json": '{"id": 9}'}))
    assert res.structured_content == {
        "action": "delete_issue",
        "params_json": '{"id": 9}',
    }


def test_autowire_dynamic_is_idempotent():
    """Re-running with a dynamic provider derives nothing new on the second pass."""
    mcp = FastMCP("t")
    _freeform_action_module().register_jira_tools(mcp)
    register_action_provider(
        mcp, "atlassian_jira", lambda: ["get_issue", "create_issue"]
    )
    first = autowire_verbose_from_condensed(mcp)
    assert len(first) == 2
    assert autowire_verbose_from_condensed(mcp) == []


def test_surface_both_autowires_dynamic_action_tools(monkeypatch):
    """register_tool_surface threads action_providers through to the auto-wire so a
    free-form connector (atlassian) gets a verbose surface centrally (ECO-4.90)."""
    monkeypatch.setenv("MCP_TOOL_MODE", "both")
    mcp = FastMCP("t")
    register_tool_surface(
        mcp,
        service="atlassian-agent",
        tools_module=_freeform_action_module(),
        action_providers={"atlassian_jira": _FakeDynamicClient},
    )
    names = {t.name for t in _tools_list(mcp)}
    assert "atlassian_jira" in names  # condensed kept
    assert "atlassian_jira__get_issue" in names  # verbose auto-derived from runtime
    assert "atlassian_jira__create_issue" in names
    assert "atlassian_jira__delete_issue" in names


def test_surface_dynamic_autowire_opt_out(monkeypatch):
    """autowire_condensed=False also suppresses dynamic-action expansion."""
    monkeypatch.setenv("MCP_TOOL_MODE", "both")
    mcp = FastMCP("t")
    register_tool_surface(
        mcp,
        service="atlassian-agent",
        tools_module=_freeform_action_module(),
        action_providers={"atlassian_jira": _FakeDynamicClient},
        autowire_condensed=False,
    )
    names = {t.name for t in _tools_list(mcp)}
    assert "atlassian_jira" in names
    assert "atlassian_jira__get_issue" not in names
