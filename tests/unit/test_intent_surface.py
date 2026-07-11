"""Seam 8 (CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse) — the intent-verb resolver + dispatcher.

Covers the resolver ranking, the ``_execute_tool`` dispatch (with a fake
registered tool so this is hermetic — no live engine required), the
``ask``-specific NL-planner fallback, and that the resolver still reaches an
EXISTING granular tool (``graph_query``) a real ``kg-query`` skill documents —
proving no capability was lost by adding the intent surface.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import intent_tools


@pytest.fixture(autouse=True)
def _fresh_candidate_cache():
    """Force the candidate table to rebuild against whatever REGISTERED_TOOLS
    a test monkeypatched, and restore process-global state after.

    Populates the REAL ~95-tool surface FIRST (idempotent — a no-op if an
    earlier test already did) so a test's ``monkeypatch.setitem`` only ever
    OVERRIDES one entry rather than being mistaken for the whole surface (that
    ordering bug would make ``ensure_tools_registered``'s one-shot guard skip
    real registration if a monkeypatch touched the dict before anything else
    in the process did).
    """
    kg_server.ensure_tools_registered()
    intent_tools._CANDIDATES_CACHE = None
    intent_tools._ACTIONS_BY_TOOL_CACHE = None
    yield
    intent_tools._CANDIDATES_CACHE = None
    intent_tools._ACTIONS_BY_TOOL_CACHE = None


@pytest.mark.asyncio
async def test_ask_routes_to_the_right_tool_and_dispatches_via_execute_tool(
    monkeypatch,
):
    """The required end-to-end proof: ask(<NL read>) resolves + dispatches
    through _execute_tool and returns the result PLUS the routing justification."""
    seen: dict = {}

    async def fake_graph_search(query: str = "", **_kw) -> str:
        seen["query"] = query
        return json.dumps({"hits": [{"id": "n1"}]})

    monkeypatch.setitem(kg_server.REGISTERED_TOOLS, "graph_search", fake_graph_search)
    monkeypatch.setitem(intent_tools.TOOL_VERBS, "graph_search", ("ask",))

    result = await intent_tools.dispatch_intent(
        "ask", "search the knowledge graph for chronoid retrieval concepts"
    )

    assert result["executed"] is True
    assert result["routing"]["chosen_tool"] == "graph_search"
    assert result["routing"]["verb"] == "ask"
    assert "search" in result["routing"]["matched_terms"]
    assert result["routing"]["why"]
    assert result["routing"]["capability_source"]
    # The primary-text-param convenience actually reached the underlying tool.
    assert "chronoid" in seen["query"]
    assert json.loads(result["result"])["hits"][0]["id"] == "n1"


@pytest.mark.asyncio
async def test_ask_falls_back_to_nl_planner_for_structured_only_tools(monkeypatch):
    """A winning candidate with no free-text param and no caller hints falls back
    to nl_query (the engine's own NL planner) rather than dispatching a call
    that's missing required arguments."""
    seen: dict = {}

    async def fake_nl_query(text: str = "", **_kw) -> str:
        seen["text"] = text
        return json.dumps({"planned": True})

    monkeypatch.setitem(kg_server.REGISTERED_TOOLS, "nl_query", fake_nl_query)

    # graph_code_nav has no _PRIMARY_TEXT_PARAM entry and needs action+symbol —
    # an intent that strongly names it (via its own tool-name tokens) should
    # still win the ranking, but dispatch must fall back rather than call it
    # with zero arguments.
    result = await intent_tools.dispatch_intent(
        "ask", "code nav: navigate and look up symbols in the codebase"
    )

    assert result["executed"] is True
    assert result["routing"]["fell_back_to_nl_planner"] is True
    assert result["routing"]["chosen_tool"] == "nl_query"
    assert "codebase" in seen["text"] or "navigate" in seen["text"]


@pytest.mark.asyncio
async def test_explicit_tool_hint_pins_resolution(monkeypatch):
    """hints={'tool': ...} is the structured escape hatch — bypasses ranking."""
    seen: dict = {}

    async def fake_graph_write(node_id: str = "", **_kw) -> str:
        seen["node_id"] = node_id
        return "ok"

    monkeypatch.setitem(kg_server.REGISTERED_TOOLS, "graph_write", fake_graph_write)

    result = await intent_tools.dispatch_intent(
        "ask",  # deliberately the "wrong" verb for graph_write — the pin overrides it
        "irrelevant wording",
        hints={"tool": "graph_write", "node_id": "abc123"},
    )
    assert result["routing"]["chosen_tool"] == "graph_write"
    assert result["routing"]["matched_terms"] == ["explicit tool hint"]
    assert seen["node_id"] == "abc123"


@pytest.mark.asyncio
async def test_dispatch_reports_error_without_crashing_on_missing_required_args(
    monkeypatch,
):
    async def fake_tool(
        required_field: str,
    ) -> str:  # no default -> TypeError if omitted
        return required_field

    monkeypatch.setitem(kg_server.REGISTERED_TOOLS, "fake_strict_tool", fake_tool)
    monkeypatch.setitem(intent_tools.TOOL_VERBS, "fake_strict_tool", ("write",))

    result = await intent_tools.dispatch_intent(
        "write", "do the fake strict thing", hints={"tool": "fake_strict_tool"}
    )
    assert result["executed"] is False
    assert "error" in result
    assert result["routing"]["chosen_tool"] == "fake_strict_tool"


def test_resolve_intent_filters_by_verb():
    candidates = intent_tools.resolve_intent(
        "manage", "configure a connector", top_k=10
    )
    tools = {c.tool for c in candidates}
    assert tools <= set(intent_tools.TOOL_VERBS) or tools == set()
    for c in candidates:
        assert "manage" in c.verbs


@pytest.mark.asyncio
async def test_find_is_verb_agnostic_and_never_errors_without_a_fleet_mux():
    payload = await intent_tools._find_capability(
        object(), "ingest a document", top_k=5
    )
    assert payload["query"] == "ingest a document"
    assert isinstance(payload["results"], list)
    assert (
        "fleet_results" not in payload
    )  # no _fleet_mux on a bare object — degrades cleanly


# --------------------------------------------------------------------------- #
# No functionality lost: the resolver still reaches an EXISTING granular tool
# that a real kg-* skill documents.
# --------------------------------------------------------------------------- #


def test_graph_query_still_registered_and_resolvable_under_ask():
    kg_server.ensure_tools_registered()
    assert "graph_query" in kg_server.REGISTERED_TOOLS
    intent_tools._CANDIDATES_CACHE = None
    candidates = intent_tools.resolve_intent(
        "ask", "run a read-only cypher query against the knowledge graph", top_k=8
    )
    assert "graph_query" in {c.tool for c in candidates}


def test_kg_query_skill_still_documents_graph_query():
    """The kg-query skill (agent_utilities/skills/kg-query/SKILL.md) is the
    operator-facing doc for graph_query — the skill sweep must not have
    silently dropped that capability."""
    from pathlib import Path

    skill_path = (
        Path(__file__).resolve().parents[2]
        / "agent_utilities"
        / "skills"
        / "kg-query"
        / "SKILL.md"
    )
    text = skill_path.read_text(encoding="utf-8")
    assert "graph_query" in text
