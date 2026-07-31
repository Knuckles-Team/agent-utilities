"""The evidence spine is reachable from BOTH operator surfaces.

CONCEPT:AU-KG.ingest.stable-fragment-address

Contract level: "Two surfaces by default" says every capability must be reachable
from the REST gateway AND from MCP, dispatching into ONE core.  The spine's read
surface is carried by ``graph_document_tree``, whose REST twin
``/graph/document-tree`` is already registered in ``ACTION_TOOL_ROUTES`` — so the
contract to pin is that the tool's *action set* actually grew, and that the
generated graph-os verbose manifest agrees.  A superset check would pass while a
route silently vanished, so the action set is asserted exactly.
"""

from __future__ import annotations

from agent_utilities.mcp import kg_server
from agent_utilities.mcp._graphos_action_manifest import GRAPHOS_ACTIONS
from tests.wiring import assert_surface

TOOL = "graph_document_tree"


def actions_for(tool: str) -> set[str]:
    return {
        str(entry["action"])
        for entry in GRAPHOS_ACTIONS
        if entry["tool"] == tool and entry["action"]
    }


def test_document_tree_action_surface_is_exactly_pinned() -> None:
    assert_surface(
        actions_for(TOOL),
        {"build", "structure", "content", "retrieve", "fragments", "cite"},
        surface=f"{TOOL} actions",
    )


def test_the_spine_actions_have_a_rest_twin() -> None:
    """Both surfaces dispatch into the one tool — no second implementation."""
    assert kg_server.ACTION_TOOL_ROUTES[TOOL] == "/graph/document-tree"


def test_every_spine_action_is_named_in_the_verbose_manifest() -> None:
    names = {entry["name"] for entry in GRAPHOS_ACTIONS if entry["tool"] == TOOL}
    assert {
        f"{TOOL}_fragments",
        f"{TOOL}_cite",
    } <= names, "the generated graph-os verbose surface must expose both spine ops"
