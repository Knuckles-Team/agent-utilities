"""Tests for TeamCapability — team coordination with ACP/A2A.

Concept: team-coordination
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import networkx as nx
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeDeps:
    graph_engine: Any = None
    acp_session: Any = None
    a2a_client: Any = None
    agent_id: str = "agent_alpha"


class FakeRunContext:
    def __init__(self, deps: FakeDeps) -> None:
        self.deps = deps


class FakeGraphEngine:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine() -> FakeGraphEngine:
    return FakeGraphEngine()


@pytest.fixture()
def ctx_with_graph(engine: FakeGraphEngine) -> FakeRunContext:
    return FakeRunContext(FakeDeps(graph_engine=engine))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.concept("team-coordination")
@pytest.mark.asyncio
async def test_message_member_a2a_fallback(engine: FakeGraphEngine) -> None:
    """When ACP is unavailable, message_member should try A2A fallback."""
    from agent_utilities.capabilities.teams import TeamCapability

    a2a_mock = AsyncMock()
    a2a_mock.send = AsyncMock(return_value=None)

    deps = FakeDeps(graph_engine=engine, a2a_client=a2a_mock)
    ctx = FakeRunContext(deps)

    cap = TeamCapability(team_id="team_test", members=["bob"])
    result = await cap.message_member(ctx, "bob", "hello from A2A")

    assert result is True
    a2a_mock.send.assert_awaited_once()
    call_kwargs = a2a_mock.send.call_args
    assert call_kwargs.kwargs["target_agent"] == "bob"


@pytest.mark.concept("team-coordination")
@pytest.mark.asyncio
async def test_discover_teams_returns_active(ctx_with_graph: FakeRunContext) -> None:
    """discover_teams should only return teams with status='active'."""
    from agent_utilities.capabilities.teams import TeamCapability

    g = ctx_with_graph.deps.graph_engine.graph
    g.add_node("t1", type="team", status="active", name="Alpha", member_count=3)
    g.add_node("t2", type="team", status="disbanded", name="Beta", member_count=2)
    g.add_node("t3", type="team", status="active", name="Gamma", member_count=5)

    cap = TeamCapability()
    teams = await cap.discover_teams(ctx_with_graph)

    assert len(teams) == 2
    names = {t["name"] for t in teams}
    assert names == {"Alpha", "Gamma"}


@pytest.mark.concept("team-coordination")
@pytest.mark.asyncio
async def test_update_task_status_persists(ctx_with_graph: FakeRunContext) -> None:
    """update_task_status should change the status in the graph."""
    from agent_utilities.capabilities.teams import TeamCapability

    g = ctx_with_graph.deps.graph_engine.graph
    g.add_node("task_001", type="task", status="pending", content="Fix bug")

    cap = TeamCapability()
    result = await cap.update_task_status(ctx_with_graph, "task_001", "done")

    assert result is True
    assert g.nodes["task_001"]["status"] == "done"
    assert "updated_at" in g.nodes["task_001"]


@pytest.mark.concept("team-coordination")
@pytest.mark.asyncio
async def test_get_team_members_from_graph(ctx_with_graph: FakeRunContext) -> None:
    """get_team_members should walk BELONGS_TO_TEAM edges in the graph."""
    from agent_utilities.capabilities.teams import TeamCapability

    g = ctx_with_graph.deps.graph_engine.graph
    g.add_node("team_abc", type="team", status="active", name="Test")
    g.add_node("agent_1")
    g.add_node("agent_2")
    g.add_edge("agent_1", "team_abc", type="BELONGS_TO_TEAM")
    g.add_edge("agent_2", "team_abc", type="BELONGS_TO_TEAM")

    cap = TeamCapability(team_id="team_abc", members=["fallback"])
    members = await cap.get_team_members(ctx_with_graph)

    assert set(members) == {"agent_1", "agent_2"}
