"""An agent saved via the Agent Library is reachable through the intent router.

The ``find`` verb ranks the ``agent_library`` tool, ``ask`` lists and
gets records read-only, and ``manage`` saves through a previewed plan.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import intent_tools
from agent_utilities.orchestration.agent_library import AgentLibrary, AgentRecord
from tests.unit.orchestration.agent_library_fakes import FakeLibraryEngine


@pytest.fixture
def engine(monkeypatch) -> FakeLibraryEngine:
    kg_server.ensure_tools_registered()
    fake = FakeLibraryEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: fake)
    intent_tools._CANDIDATES_CACHE = None
    intent_tools._ACTIONS_BY_TOOL_CACHE = None
    intent_tools._RESOLUTION_CACHE.clear()
    intent_tools._PREVIEW_PLAN_CACHE.clear()
    yield fake
    intent_tools._CANDIDATES_CACHE = None
    intent_tools._ACTIONS_BY_TOOL_CACHE = None
    intent_tools._RESOLUTION_CACHE.clear()
    intent_tools._PREVIEW_PLAN_CACHE.clear()


def _payload(result: dict) -> dict:
    raw = result["result"]
    return json.loads(raw) if isinstance(raw, str) else raw


@pytest.mark.asyncio
async def test_find_ranks_the_agent_library(engine) -> None:
    found = await intent_tools._find_capability(
        None, "list the agents saved in the agent library"
    )
    assert found["results"][0]["tool"] == "agent_library"


@pytest.mark.asyncio
async def test_a_library_saved_agent_is_listed_via_ask(engine) -> None:
    saved = AgentLibrary(engine).save(
        AgentRecord(name="billing-analyst", system_prompt="Analyse billing.")
    )
    result = await intent_tools.dispatch_intent(
        "ask",
        "list the agents in the agent library",
        hints={"tool": "agent_library", "action": "list"},
    )
    assert result["executed"] is True
    assert result["routing"]["chosen_tool"] == "agent_library"
    assert [a["id"] for a in _payload(result)["agents"]] == [saved.agent_id]


@pytest.mark.asyncio
async def test_ask_cannot_save(engine) -> None:
    result = await intent_tools.dispatch_intent(
        "ask",
        "save an agent to the agent library",
        hints={"tool": "agent_library", "action": "save"},
    )
    assert result.get("executed") is not True
    assert engine.nodes == {}


@pytest.mark.asyncio
async def test_manage_saves_through_a_previewed_plan(engine) -> None:
    agent = {"name": "triage", "system_prompt": "Triage incidents.", "role": "Triage"}
    hints = {"tool": "agent_library", "action": "save", "agent_json": json.dumps(agent)}
    preview = await intent_tools.dispatch_intent(
        "manage", "save an agent to the agent library", hints=hints
    )
    assert preview["executed"] is False and engine.nodes == {}
    plan_ref = preview["routing"]["plan"]["plan_ref"]
    done = await intent_tools.dispatch_intent(
        "manage",
        "save an agent to the agent library",
        hints={**hints, "plan_ref": plan_ref},
        execute=True,
    )
    assert done["executed"] is True
    listed = [r.name for r in AgentLibrary(engine).list()]
    assert listed == ["triage"]
