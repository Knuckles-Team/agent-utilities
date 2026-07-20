"""Team coordination uses WorkItem as its only task-state authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


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
        self.graph = GraphComputeEngine(backend_type="rust")


@pytest.fixture()
def engine() -> FakeGraphEngine:
    return FakeGraphEngine()


@pytest.fixture()
def ctx_with_graph(engine: FakeGraphEngine) -> FakeRunContext:
    return FakeRunContext(FakeDeps(graph_engine=engine))


@pytest.mark.asyncio
async def test_message_member_a2a_fallback(engine: FakeGraphEngine) -> None:
    from agent_utilities.capabilities.teams import TeamCapability

    a2a = AsyncMock()
    a2a.send = AsyncMock(return_value=None)
    ctx = FakeRunContext(FakeDeps(graph_engine=engine, a2a_client=a2a))
    cap = TeamCapability(team_id="team_ref", members=["member_ref"])
    assert await cap.message_member(ctx, "member_ref", "hello")
    a2a.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_discover_teams_returns_active(ctx_with_graph: FakeRunContext) -> None:
    from agent_utilities.capabilities.teams import TeamCapability

    graph = ctx_with_graph.deps.graph_engine.graph
    graph.add_node("t1", type="team", status="active", name="Alpha")
    graph.add_node("t2", type="team", status="disbanded", name="Beta")
    teams = await TeamCapability().discover_teams(ctx_with_graph)
    assert [team["team_id"] for team in teams] == ["t1"]


@pytest.mark.asyncio
async def test_add_task_persists_only_authoritative_work_item(
    ctx_with_graph: FakeRunContext,
) -> None:
    from agent_utilities.capabilities.teams import (
        TeamCapability,
        _GraphComputeWorkItemView,
    )
    from agent_utilities.orchestration import work_item

    task_ref = await TeamCapability(team_id="team_ref").add_task(
        ctx_with_graph, "do the thing", assigned_to="agent_ref"
    )
    graph = ctx_with_graph.deps.graph_engine.graph
    item_id = work_item.team_work_item_id(task_ref)
    item = work_item.get_work_item(_GraphComputeWorkItemView(graph), item_id)
    assert item is not None
    assert item["kind"] == "team_assignment"
    assert item["status"] == "ready"
    assert task_ref not in graph.nodes


@pytest.mark.asyncio
async def test_team_transition_changes_only_work_item(
    ctx_with_graph: FakeRunContext,
) -> None:
    from agent_utilities.capabilities.teams import (
        TeamCapability,
        _GraphComputeWorkItemView,
    )
    from agent_utilities.orchestration import work_item

    cap = TeamCapability(team_id="team_ref")
    task_ref = await cap.add_task(ctx_with_graph, "do the thing")
    view = _GraphComputeWorkItemView(ctx_with_graph.deps.graph_engine.graph)
    item_id = work_item.team_work_item_id(task_ref)

    assert await cap.update_task_status(ctx_with_graph, task_ref, "in_progress")
    assert work_item.get_work_item(view, item_id)["status"] == "running"
    assert await cap.update_task_status(ctx_with_graph, task_ref, "completed")
    assert work_item.get_work_item(view, item_id)["status"] == "succeeded"


@pytest.mark.asyncio
async def test_team_rejects_noncanonical_transition(
    ctx_with_graph: FakeRunContext,
) -> None:
    from agent_utilities.capabilities.teams import TeamCapability

    cap = TeamCapability(team_id="team_ref")
    task_ref = await cap.add_task(ctx_with_graph, "do the thing")
    with pytest.raises(ValueError, match="WorkItem owns"):
        await cap.update_task_status(ctx_with_graph, task_ref, "blocked")
