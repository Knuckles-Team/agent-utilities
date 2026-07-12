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


# --------------------------------------------------------------------------- #
# Seam 8 remaining work: CPD-backed ranking, the outcome-learning loop, and
# resolution caching (docs/architecture/intent-surface.md §7).
# --------------------------------------------------------------------------- #


def test_resolver_ranks_against_the_generated_cpd_when_available():
    """CONCEPT:AU-ECO.mcp.intent-surface-cpd-ranking — a real tool with a generated CPD
    entry (docs/capabilities-power.json) is ranked using its CPD one_line/
    examples/does text, not just its bare docstring, and dispatch reports the
    CPD as the capability_source."""
    cpds = intent_tools._load_cpds_safe()
    assert "graph_query" in cpds, "docs/capabilities-power.json must be checked in"

    candidates = intent_tools.resolve_intent(
        "ask", "execute a read-only cypher query", top_k=8
    )
    by_tool = {c.tool: c for c in candidates}
    assert "graph_query" in by_tool
    # The CPD's own example text ("graph_query action=graph_query") and
    # one_line both feed the candidate doc — a term only present there (not in
    # the bare function docstring) should still be attributable to the match.
    assert "cypher" in by_tool["graph_query"].doc.lower()


@pytest.mark.asyncio
async def test_dispatch_reports_cpd_capability_source_for_a_cpd_backed_tool(
    monkeypatch,
):
    async def fake_graph_query(query: str = "", **_kw) -> str:
        return json.dumps({"rows": []})

    monkeypatch.setitem(kg_server.REGISTERED_TOOLS, "graph_query", fake_graph_query)

    result = await intent_tools.dispatch_intent(
        "ask", "run a cypher query against the graph", hints={"tool": "graph_query"}
    )
    assert result["executed"] is True
    assert "Capability Power Descriptor" in result["routing"]["capability_source"]
    assert "calibrated_outcome_reward" in result["routing"]


@pytest.mark.asyncio
async def test_outcome_learning_biases_a_later_resolution(monkeypatch):
    """CONCEPT:AU-ECO.mcp.intent-surface-outcome-learning — recording a dispatch outcome
    feeds the shared durable-bandit reward-EMA (OutcomeRouter, over
    CapabilityIndex.record_outcome/reward_of — no second learner), and a LATER
    unpinned resolution for the same verb prefers the capability that
    succeeded over one it exactly ties with lexically."""

    async def fake_a(**_kw) -> str:
        return "ok-a"

    async def fake_b(**_kw) -> str:
        return "ok-b"

    monkeypatch.setitem(kg_server.REGISTERED_TOOLS, "fake_tool_a", fake_a)
    monkeypatch.setitem(kg_server.REGISTERED_TOOLS, "fake_tool_b", fake_b)
    monkeypatch.setitem(intent_tools.TOOL_VERBS, "fake_tool_a", ("act",))
    monkeypatch.setitem(intent_tools.TOOL_VERBS, "fake_tool_b", ("act",))
    intent_tools._CANDIDATES_CACHE = None

    # Nonsense wording that matches neither tool's name/doc -> guaranteed tie
    # at score 0.0, so the reward-EMA blend is the ONLY thing that can decide
    # the ordering.
    nonsense_intent = "zzqzq wobbleflorp nonexistent gizmo"

    # top_k generous enough that BOTH fake tools survive the top-k slice
    # regardless of the (score, tool) tie-break among the real 'act' surface.
    before = intent_tools.resolve_intent("act", nonsense_intent, top_k=100)
    before_scores = {c.tool: c.score for c in before}
    assert before_scores["fake_tool_a"] == before_scores["fake_tool_b"] == 0.0

    result = await intent_tools.dispatch_intent(
        "act", nonsense_intent, hints={"tool": "fake_tool_a"}
    )
    assert result["executed"] is True
    assert result["result"] == "ok-a"

    after = intent_tools.resolve_intent("act", nonsense_intent, top_k=100)
    after_scores = {c.tool: c.score for c in after}
    assert after_scores["fake_tool_a"] > after_scores["fake_tool_b"]
    ranked_tools = [c.tool for c in after if c.tool in ("fake_tool_a", "fake_tool_b")]
    assert ranked_tools[0] == "fake_tool_a"


def test_resolution_cache_hits_repeat_intent_misses_a_different_one(monkeypatch):
    """CONCEPT:AU-ECO.mcp.intent-surface-resolution-cache — the SAME (verb, intent) resolves
    from the bounded cache (no new entry, identical ranking) while a
    differently-worded intent is a fresh miss (a new cache entry)."""

    async def fake_cache_tool(**_kw) -> str:
        return "ok"

    monkeypatch.setitem(kg_server.REGISTERED_TOOLS, "fake_cache_tool", fake_cache_tool)
    monkeypatch.setitem(intent_tools.TOOL_VERBS, "fake_cache_tool", ("ask",))
    intent_tools._CANDIDATES_CACHE = None
    intent_tools._RESOLUTION_CACHE.clear()

    size_before = len(intent_tools._RESOLUTION_CACHE)
    r1 = intent_tools.resolve_intent("ask", "  Find The Fake Cache Tool  ", top_k=5)
    size_after_first = len(intent_tools._RESOLUTION_CACHE)
    assert size_after_first == size_before + 1

    # Same intent modulo case/whitespace -> normalizes to the SAME cache key.
    r2 = intent_tools.resolve_intent("ask", "find the fake cache tool", top_k=5)
    size_after_second = len(intent_tools._RESOLUTION_CACHE)
    assert size_after_second == size_after_first  # cache hit — no new entry
    assert [c.tool for c in r1] == [c.tool for c in r2]
    assert [c.score for c in r1] == [c.score for c in r2]

    # A genuinely different intent is a fresh key.
    intent_tools.resolve_intent("ask", "an entirely unrelated intent phrase", top_k=5)
    size_after_third = len(intent_tools._RESOLUTION_CACHE)
    assert size_after_third == size_after_second + 1
