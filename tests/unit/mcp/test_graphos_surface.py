"""GATE — graph-os serves ONE condensed intent contract (ECO-4.82).

CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse

Pins the operator's tool-surface decision (MCP-TOOL-SURFACE-SPEC, 2026-10-07
amendment):

* the default ``tools/list`` is the six intent verbs plus the two MCP Apps
  launchers — at most :data:`DEFAULT_SURFACE_TOOL_LIMIT` tools within
  :data:`DEFAULT_SURFACE_TOKEN_BUDGET` tokens, no tool over
  :data:`MAX_TOOL_TOKENS`;
* every intent verb takes the condensed contract (``action`` + ``params``
  [+ ``intent`` / ``execute``]) with a one-to-two-line description;
* every operation in the generated action manifest
  (``_graphos_action_manifest.GRAPHOS_ACTIONS``) belongs to exactly one job-based
  routing group and is reachable through the intent router;
* the action-routed tools stay registered in the dispatch core
  (``REGISTERED_TOOLS``) the REST gateway and the router share.

The MCP Apps launchers are the one justified exception to "intent tools only":
the MCP Apps extension (``io.modelcontextprotocol/ui``) binds a UI resource to a
tool through that tool's ``_meta.ui.resourceUri`` in ``tools/list``, so a host
can only render an app whose launcher is a listed tool.

``bootstrap=False`` skips the engine/daemon startup — registration and routing
only, no live engine.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Literal

import pytest

from agent_utilities.mcp import kg_server
from agent_utilities.mcp._graphos_action_manifest import GRAPHOS_ACTIONS
from agent_utilities.mcp.graphos_surface import (
    ROUTING_GROUPS,
    backing_tools,
    group_for_tool,
    manifest_operations,
)
from agent_utilities.mcp.intent_contract import (
    DEFAULT_SURFACE_TOKEN_BUDGET,
    DEFAULT_SURFACE_TOOL_LIMIT,
    estimate_tokens,
    tool_definition,
)
from agent_utilities.mcp.optional_tool_features import OPTIONAL_TOOL_FEATURES
from agent_utilities.mcp.tool_specs import INTENT_VERBS
from agent_utilities.mcp.tools import engine_tools, intent_tools

#: No single listed tool may exceed this (spec: "no tool > ~500 tokens").
MAX_TOOL_TOKENS = 500
#: MCP Apps launchers — listed because the MCP Apps extension requires it.
MCP_APP_TOOLS = frozenset({"graph_task_progress_app", "graph_trace_waterfall_app"})


@pytest.fixture(scope="module")
def served():
    """``(mcp, {name: tools/list definition})`` for a freshly built graph-os."""
    saved = dict(kg_server.REGISTERED_TOOLS)
    routes = dict(kg_server.ACTION_TOOL_ROUTES)
    kg_server.REGISTERED_TOOLS.clear()
    try:
        _args, mcp, _middlewares = kg_server._build_server(bootstrap=False)
        listed = asyncio.run(mcp.list_tools())
        registry = dict(kg_server.REGISTERED_TOOLS)
        yield mcp, {tool.name: tool_definition(tool) for tool in listed}, registry
    finally:
        kg_server.REGISTERED_TOOLS.clear()
        kg_server.REGISTERED_TOOLS.update(saved)
        kg_server.ACTION_TOOL_ROUTES.clear()
        kg_server.ACTION_TOOL_ROUTES.update(routes)


def test_default_tools_list_is_the_intent_verbs_and_the_app_launchers(served):
    _mcp, listed, _registry = served
    assert set(listed) == set(INTENT_VERBS) | MCP_APP_TOOLS
    assert len(listed) <= DEFAULT_SURFACE_TOOL_LIMIT


def test_default_tools_list_fits_the_token_budget(served):
    _mcp, listed, _registry = served
    sizes = {name: estimate_tokens(definition) for name, definition in listed.items()}
    oversized = {name: n for name, n in sizes.items() if n > MAX_TOOL_TOKENS}
    assert not oversized, oversized
    assert sum(sizes.values()) <= DEFAULT_SURFACE_TOKEN_BUDGET, sizes


@pytest.mark.parametrize("verb", sorted(INTENT_VERBS))
def test_every_intent_verb_takes_the_condensed_contract(served, verb):
    _mcp, listed, _registry = served
    definition = listed[verb]
    properties = definition["inputSchema"]["properties"]
    assert set(properties) == {"action", "params", "intent", "execute"}
    assert properties["params"]["type"] == "object"
    assert 1 <= len(definition["description"].splitlines()) <= 2


def test_every_manifest_family_has_exactly_one_routing_group():
    owners = Counter(family for group in ROUTING_GROUPS for family in group.families)
    assert not [family for family, n in owners.items() if n > 1]
    families = {op["tool"] for op in GRAPHOS_ACTIONS} - set(INTENT_VERBS)
    assert not sorted(family for family in families if group_for_tool(family) is None)
    assert set(owners) <= families, "routing table names a family the manifest lacks"


def _engine_client_absent_families(registry) -> set[str]:
    """``engine_<domain>`` families that cannot register without the engine client.

    ``engine_tools`` discovers its domains from the ``epistemic_graph`` client.
    A test environment without that wheel registers none of them.
    """
    if engine_tools.ENGINE_DOMAINS:
        return set()
    return {
        tool
        for tool, _ in manifest_operations().values()
        if tool.startswith("engine_") and tool not in registry
    }


def test_the_dispatch_core_holds_every_served_manifest_family(served):
    """Registered is not listed: the backing tools populate ``REGISTERED_TOOLS``
    (the REST gateway's and the router's dispatch table) while only the intent
    tools are listed."""
    mcp, _listed, registry = served
    families = {
        tool for tool, _ in manifest_operations().values()
    } - _engine_client_absent_families(registry)
    unserved = families - set(registry) - set(OPTIONAL_TOOL_FEATURES)
    assert not unserved, sorted(unserved)
    assert set(INTENT_VERBS) <= set(registry)
    for verb in INTENT_VERBS:
        assert kg_server.ACTION_TOOL_ROUTES[verb] == f"/intent/{verb}"
    assert families - set(OPTIONAL_TOOL_FEATURES) <= set(backing_tools(mcp))


def test_every_manifest_operation_is_reachable_through_the_intent_router(served):
    """Each operation previews through the first verb that accepts it, routed
    to exactly its backing tool and action. An optional-feature family that is
    not installed answers with an actionable error instead."""
    mcp, _listed, registry = served
    verbs_by_op = intent_tools.operation_verbs()
    unreachable: dict[str, str] = {}
    absent = _engine_client_absent_families(registry)

    async def _probe() -> None:
        for op_id, (tool, action) in manifest_operations().items():
            if tool in absent:
                continue
            verbs = verbs_by_op.get(op_id) or ()
            if not verbs:
                unreachable[op_id] = "no verb accepts it"
                continue
            result = await intent_tools._dispatch_verb(
                mcp, verbs[0], op_id, {}, "", False
            )
            if tool in OPTIONAL_TOOL_FEATURES and tool not in registry:
                if OPTIONAL_TOOL_FEATURES[tool] not in str(result.get("error")):
                    unreachable[op_id] = str(result)
                continue
            routing = result.get("routing") or {}
            if result.get("error") or (
                routing.get("chosen_tool"),
                routing.get("action"),
            ) != (tool, action):
                unreachable[op_id] = str(result.get("error") or routing)[:200]

    asyncio.run(_probe())
    assert not unreachable, dict(list(unreachable.items())[:20])


def test_describe_serves_one_operations_schema_on_demand(served):
    mcp, _listed, _registry = served
    out = intent_tools._describe(mcp, "write", {"action": "graph_write.add_node"})
    assert out["group"] == "write"
    assert "write" in out["verbs"]
    assert "action" not in out["params_schema"]["properties"]
    catalog = intent_tools._describe(mcp, "manage", {})
    assert "approve" in catalog["operations"]["host"]
    assert "fleet.load" in catalog["operations"]["host"]


def test_a_host_operation_is_reached_through_act_with_preview(served):
    """A host-native backing tool (graph-os's A2A / browser control shape)
    becomes ``act`` operations ``<tool>.<action>`` that preview first."""
    from agent_utilities.mcp.graphos_surface import backing_server, host_operations

    mcp, _listed, _registry = served
    calls: list[dict] = []

    @backing_server(mcp).tool(name="host_echo")
    async def host_echo(action: Literal["say", "drop"], text: str = "") -> str:
        calls.append({"action": action, "text": text})
        return text

    assert host_operations(mcp)["host_echo.say"] == ("host_echo", "say")
    described = intent_tools._describe(mcp, "act", {"action": "host_echo.say"})
    assert "text" in described["params_schema"]["properties"]

    async def _run() -> tuple[dict, dict, dict]:
        params = {"text": "hi"}
        preview = await intent_tools._dispatch_verb(
            mcp, "act", "host_echo.say", dict(params), "", False
        )
        refused = await intent_tools._dispatch_verb(
            mcp, "act", "host_echo.say", dict(params), "", True
        )
        ref = preview["plan"]["plan_ref"]
        done = await intent_tools._dispatch_verb(
            mcp, "act", "host_echo.say", {**params, "plan_ref": ref}, "", True
        )
        return preview, refused, done

    preview, refused, done = asyncio.run(_run())
    assert preview["executed"] is False and refused["executed"] is False
    assert done["executed"] is True
    assert calls == [{"action": "say", "text": "hi"}]
