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

import asyncio
import json

import pytest

from agent_utilities.mcp import kg_server
from agent_utilities.mcp._graphos_action_manifest import GRAPHOS_ACTIONS
from tests.wiring import assert_surface

TOOL = "graph_document_tree"

DOC = "# Title\n\nHello world.\n\n## Sub\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"


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


@pytest.fixture
def tool(monkeypatch: pytest.MonkeyPatch):
    """Invoke the action through ``_execute_tool`` — the ONE dispatch core.

    Registration alone proves nothing about whether the action body runs; a typo
    in the dispatch branch would pass every assertion above.  Driving
    ``_execute_tool`` (rather than the raw function) is also what makes this a
    two-surfaces proof: it is the same core the REST gateway and the MCP server
    both call, including its resolution of ``Field`` defaults for omitted args.
    Inline text needs no engine, so the action is exercised for real.
    """
    kg_server.ensure_tools_registered()
    monkeypatch.setattr(kg_server, "_get_engine", lambda *a, **k: None)

    def call(**kwargs: object) -> str:
        return asyncio.run(kg_server._execute_tool(TOOL, **kwargs))

    return call


def test_fragments_action_returns_an_addressable_spine(tool) -> None:
    out = json.loads(tool(action="fragments", text=DOC))
    assert out["fragment_count"] == len(out["fragments"]) > 0
    for fragment in out["fragments"]:
        assert fragment["address"]
        assert fragment["content_hash"].startswith("sha256:")
        # '#', not '@' -- see D-GM-4 / D-GS856-6 / D-MW-1 / D-MW-2.
        assert fragment["version_id"].startswith(fragment["fragment_id"] + "#")
        assert "@" not in fragment["version_id"]
    kinds = {f["kind"] for f in out["fragments"]}
    assert {"heading", "paragraph", "table", "table_row"} <= kinds


def test_fragments_action_filters_by_kind(tool) -> None:
    out = json.loads(tool(action="fragments", text=DOC, kinds="table_row"))
    assert {f["kind"] for f in out["fragments"]} == {"table_row"}


def test_cite_action_distinguishes_current_from_stale(tool) -> None:
    listed = json.loads(tool(action="fragments", text=DOC))["fragments"]
    cited = next(f for f in listed if f["kind"] == "paragraph")

    current = json.loads(
        tool(
            action="cite",
            text=DOC,
            fragment_id=cited["fragment_id"],
            content_hash=cited["content_hash"],
        )
    )
    assert current["status"] == "current"

    stale = json.loads(
        tool(
            action="cite",
            text=DOC.replace("Hello world.", "Hello there."),
            fragment_id=cited["fragment_id"],
            content_hash=cited["content_hash"],
        )
    )
    # The address survived the edit; only the content did not.
    assert stale["status"] == "stale"
    assert stale["fragment_id"] == cited["fragment_id"]


def test_spine_actions_refuse_an_unaddressable_request(tool) -> None:
    """No text, no artifact, no document — say so, never fabricate an empty spine."""
    assert "error" in json.loads(tool(action="fragments"))
    assert "error" in json.loads(tool(action="cite", text=DOC))
