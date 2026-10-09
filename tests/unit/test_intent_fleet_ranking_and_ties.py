"""Intent routing: fleet ranking in ``find`` and non-refusing read ties.

AU-CONTROL-R031 (fleet hits ranked with native capabilities),
AU-CONTROL-R033 (a read-only action tie runs the declared default) and
AU-CONTROL-R034 (an unmatched read intent falls back to the NL planner).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from pydantic import Field

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import intent_tools


class _FakeFleetMux:
    def __init__(self, discovery: dict) -> None:
        self.discover_tools = AsyncMock(return_value=discovery)

    def require_capability(self, _capability: str) -> None:
        """The fake grants every capability."""


class _FakeMCP:
    def __init__(self, mux: object) -> None:
        self._fleet_mux = mux


_DOCKER_HIT = {
    "kind": "tool",
    "server": "container-manager-mcp",
    "tool": "list_containers",
    "prefixed_name": "cm__list_containers",
    "description": "List Docker containers on the host.",
    "score": 1.57,
}


@pytest.mark.asyncio
async def test_find_ranks_a_fleet_tool_above_unrelated_native_capabilities():
    mcp = _FakeMCP(_FakeFleetMux({"results": [_DOCKER_HIT], "unavailable": {}}))

    payload = await intent_tools._find_capability(
        mcp, "Find a tool that lists Docker containers.", top_k=5
    )

    top = payload["results"][0]
    assert top["source"] == "fleet"
    assert top["server"] == "container-manager-mcp"
    assert "docker" in top["matched_terms"]
    assert "act(action='fleet.call'" in top["how_to_call"]
    assert "cm__list_containers" in top["how_to_call"]
    assert payload["count"] == len(payload["results"]) <= 5
    assert {row["source"] for row in payload["results"]} == {"fleet", "graphos"}


@pytest.mark.asyncio
async def test_find_reports_a_failed_fleet_probe_instead_of_hiding_it():
    class _Broken:
        def require_capability(self, _capability: str) -> None:
            raise PermissionError("no discover")

    payload = await intent_tools._find_capability(
        _FakeMCP(_Broken()), "list docker containers", top_k=3
    )

    assert payload["fleet_error"] == "PermissionError"
    assert all(row["source"] == "graphos" for row in payload["results"])


def test_fleet_rows_without_a_tool_name_point_at_catalog_or_skill():
    tokens = intent_tools._tokenize("docker containers")
    skill = intent_tools._fleet_find_row(
        tokens,
        {"kind": "skill", "server": "s", "skill": "k", "uri": "skill://k/SKILL.md"},
    )
    server = intent_tools._fleet_find_row(tokens, {"kind": "server", "server": "s"})

    assert skill["how_to_call"].startswith("Read skill://k/SKILL.md")
    assert "find(action='catalog'" in server["how_to_call"]


def test_read_tie_runs_the_declared_default_action(monkeypatch):
    async def fake_explain(
        action: str = Field(default="explain"), query: str = ""
    ) -> str:
        return action

    monkeypatch.setitem(kg_server.REGISTERED_TOOLS, "graph_explain", fake_explain)
    tied = [("context", 0.405), ("explain", 0.405), ("recommend", 0.405)]

    ranked, error = intent_tools._restrict_to_read_actions(
        "why", "graph_explain", None, tied
    )

    assert error is None
    assert ranked[0][0] == "explain"
    assert {action for action, _ in ranked} >= {"context", "recommend"}


def test_read_tie_without_a_read_default_still_refuses(monkeypatch):
    async def fake_context(action: str = "put") -> str:
        return action

    monkeypatch.setitem(kg_server.REGISTERED_TOOLS, "graph_context", fake_context)

    _, error = intent_tools._restrict_to_read_actions(
        "ask", "graph_context", None, [("get", 0.3), ("list", 0.3)]
    )

    assert error is not None
    assert error["error"] == "Ambiguous read-only action requires an explicit action."


@pytest.mark.asyncio
async def test_unmatched_ask_falls_back_to_the_nl_planner(monkeypatch):
    seen: dict = {}

    async def fake_nl_query(text: str = "", **_kw) -> str:
        seen["text"] = text
        return json.dumps({"planned": True})

    monkeypatch.setitem(kg_server.REGISTERED_TOOLS, "nl_query", fake_nl_query)
    unmatched = intent_tools.CapabilityCandidate("usage_query", None, ("ask",), "")
    monkeypatch.setattr(
        intent_tools, "resolve_intent", lambda *_a, **_kw: [unmatched, unmatched]
    )

    result = await intent_tools.dispatch_intent("ask", "zzqx flurbish vormantle")

    assert result["executed"] is True
    assert result["routing"]["chosen_tool"] == "nl_query"
    assert result["routing"]["fell_back_to_nl_planner"] is True
    assert "no capability descriptor term matched" in result["routing"]["why"]
    assert seen["text"] == "zzqx flurbish vormantle"


@pytest.mark.asyncio
async def test_pinned_tool_never_takes_the_zero_score_fallback():
    candidate = intent_tools.CapabilityCandidate("usage_query", None, ("ask",), "")

    assert intent_tools._zero_score_falls_back("ask", candidate, True, None) is False
    assert intent_tools._zero_score_falls_back("ask", candidate, False, "x") is False
    assert intent_tools._zero_score_falls_back("why", candidate, False, None) is False
    assert intent_tools._zero_score_falls_back("ask", candidate, False, None) is True
