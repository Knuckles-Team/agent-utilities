"""CX-AU-03 characterization for ``register_analysis_tools.graph_configure``
(CCN 244 pre-refactor) in ``agent_utilities/mcp/tools/analysis_tools.py``.

Pins the OBSERVED action-dispatch behaviour of the giant sequential
``if action == "X": ... return`` chain BEFORE it is decomposed into a dict
dispatch of module-level handlers, per the two-commit characterize-then-
refactor discipline (AGENTS.md). Written and made green against the
UNMODIFIED function.

``graph_configure`` is registered directly as the ``@mcp.tool`` (unlike
``_run_analysis_action``, which is wrapped by ``graph_analyze``), so it is
reached the same way real callers reach it: through
``kg_server.REGISTERED_TOOLS`` / ``kg_server._execute_tool``, which also
resolves omitted Field defaults the way FastMCP itself does.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.core.connection_registry import ConnectionRegistry
from agent_utilities.mcp import kg_server


@pytest.fixture(autouse=True)
def _reset_connection_registry():
    saved = kg_server._CONNECTION_REGISTRY
    kg_server._CONNECTION_REGISTRY = None
    yield
    kg_server._CONNECTION_REGISTRY = saved


def _install_registry() -> None:
    kg_server._CONNECTION_REGISTRY = ConnectionRegistry(
        default_engine_provider=lambda: object()
    )


async def _call(action: str, **kwargs) -> str:
    kg_server.ensure_tools_registered()
    return await kg_server._execute_tool("graph_configure", action=action, **kwargs)


async def test_unknown_action_returns_the_fixed_error_payload():
    out = await _call("definitely-not-a-real-action")
    assert json.loads(out) == {"error": "unknown configuration action"}


async def test_add_connection_requires_config_key():
    _install_registry()
    out = await _call("add_connection")
    assert json.loads(out) == {
        "error": "config_key (connection name) required for add_connection"
    }


async def test_remove_connection_requires_config_key():
    _install_registry()
    out = await _call("remove_connection")
    assert json.loads(out) == {
        "error": "config_key (connection name) required for remove_connection"
    }


async def test_list_connections_returns_registry_status():
    _install_registry()
    out = await _call("list_connections")
    payload = json.loads(out)
    # Pinning the OBSERVED shape (a dict with a "connections" list), not
    # asserting anything about its contents -- this is a fresh registry.
    assert isinstance(payload, dict)


async def test_get_config_requires_config_key():
    out = await _call("get_config")
    payload = json.loads(out)
    assert payload == {"error": "config_key (env name) required for get_config"}


async def test_set_config_requires_config_key():
    out = await _call("set_config")
    payload = json.loads(out)
    assert payload == {"error": "config_key (env name) required for set_config"}


async def test_get_config_unknown_key_is_rejected():
    out = await _call("get_config", config_key="THIS_ENV_KEY_DOES_NOT_EXIST_XYZ")
    payload = json.loads(out)
    assert payload == {"error": "Unknown config key (see config_reference)"}


async def test_config_reference_and_list_connections_are_independently_routed():
    # A dict-dispatch bug class this specifically guards against: two action
    # names accidentally mapped to the SAME handler.
    _install_registry()
    add = await _call("add_connection")
    remove = await _call("remove_connection")
    assert add != remove
    assert "add_connection" in add
    assert "remove_connection" in remove


# ---------------------------------------------------------------------------
# BUG-CX-025: graph_configure's ``action`` Field description -- the documented
# surface a caller actually reads to know what actions exist -- must name
# every action ``_CONFIGURE_ACTION_DISPATCH`` actually dispatches. Five
# dispatchable actions (profile_connection, setup_databases,
# verify_databases, doctor, set_role_routing) were missing from it.
# ---------------------------------------------------------------------------


def test_every_dispatchable_action_is_named_in_the_action_field_description():
    import inspect
    import re

    from agent_utilities.mcp.tools import analysis_tools

    kg_server.ensure_tools_registered()
    tool = kg_server.REGISTERED_TOOLS["graph_configure"]
    action_param = inspect.signature(tool).parameters["action"]
    description = action_param.default.description
    assert description, "graph_configure's 'action' Field has no description"

    dispatch_actions = set(analysis_tools._CONFIGURE_ACTION_DISPATCH.keys())
    undocumented = {
        action
        for action in dispatch_actions
        if not re.search(rf"\b{re.escape(action)}\b", description)
    }
    assert undocumented == set(), (
        "graph_configure dispatches action(s) absent from its own 'action' "
        f"Field description: {sorted(undocumented)}"
    )
