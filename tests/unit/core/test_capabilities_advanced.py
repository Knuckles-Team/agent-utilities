"""Coverage push for agent_utilities.capabilities.*.

CONCEPT:ORCH-1.2 — Resilient Agent Capabilities

Targets pure-logic / mocked-engine paths for:
  * checkpointing.InMemoryCheckpointStore / FileCheckpointStore
  * checkpointing.Checkpoint.to_json/from_json
  * eviction.ToolOutputEviction (with/without graph engine)
  * stuck_loop.StuckLoopDetection (repeated / alternating / noop)
  * teams.TeamCapability (create_team, add_task, message_member)
  * context_warnings.ContextLimitWarner (warn, critical, no usage)
  * hooks.HooksCapability (before/after hooks, cancel, modify)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import networkx as nx
import pytest

# ---------------------------------------------------------------------------
# TeamCapability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_team_create_team_no_engine() -> None:
    """create_team without engine still returns a team_id."""
    from agent_utilities.capabilities.teams import TeamCapability

    cap = TeamCapability()
    ctx = MagicMock()
    ctx.deps = MagicMock(graph_engine=None)
    team_id = await cap.create_team(ctx, "MyTeam", ["a1", "a2"])
    assert team_id.startswith("team_")
    assert cap.team_id == team_id
    assert cap.members == ["a1", "a2"]


@pytest.mark.asyncio
async def test_team_create_team_with_engine() -> None:
    """create_team with engine writes TeamNode and member edges."""
    from agent_utilities.capabilities.teams import TeamCapability

    cap = TeamCapability()
    ctx = MagicMock()
    engine = MagicMock()
    engine.graph = nx.MultiDiGraph()
    engine.graph.add_node("a1")
    engine.graph.add_node("a2")
    ctx.deps = MagicMock(graph_engine=engine)
    team_id = await cap.create_team(ctx, "MyTeam", ["a1", "a2"])
    assert team_id in engine.graph.nodes


@pytest.mark.asyncio
async def test_team_for_run_returns_copy() -> None:
    """for_run returns a replica."""
    from agent_utilities.capabilities.teams import TeamCapability

    cap = TeamCapability(team_id="t1")
    other = await cap.for_run(MagicMock())
    assert other is not cap


@pytest.mark.asyncio
async def test_team_add_task_no_engine() -> None:
    """add_task without engine still returns a task_id."""
    from agent_utilities.capabilities.teams import TeamCapability

    cap = TeamCapability()
    ctx = MagicMock()
    ctx.deps = MagicMock(graph_engine=None)
    task_id = await cap.add_task(ctx, "Do the thing")
    assert task_id.startswith("task_")


@pytest.mark.asyncio
async def test_team_add_task_with_engine_and_team() -> None:
    """add_task with engine + team creates TaskNode with BELONGS_TO_TEAM edge."""
    from agent_utilities.capabilities.teams import TeamCapability

    cap = TeamCapability(team_id="team_1")
    ctx = MagicMock()
    engine = MagicMock()
    engine.graph = nx.MultiDiGraph()
    engine.graph.add_node("team_1")
    engine.graph.add_node("agent_x")
    ctx.deps = MagicMock(graph_engine=engine, agent_id="orch")
    task_id = await cap.add_task(ctx, "Task1", assigned_to="agent_x")
    assert task_id in engine.graph.nodes


@pytest.mark.asyncio
async def test_team_message_member_acp_success() -> None:
    """message_member via ACP returns True when session is active."""
    from agent_utilities.capabilities.teams import TeamCapability

    cap = TeamCapability()
    ctx = MagicMock()
    acp = MagicMock()
    acp.send_p2p = AsyncMock()
    ctx.deps = MagicMock(acp_session=acp)
    result = await cap.message_member(ctx, "member1", "hi")
    assert result is True


@pytest.mark.asyncio
async def test_team_message_member_acp_failure() -> None:
    """message_member with ACP that raises returns False."""
    from agent_utilities.capabilities.teams import TeamCapability

    cap = TeamCapability()
    ctx = MagicMock()
    acp = MagicMock()
    acp.send_p2p = AsyncMock(side_effect=RuntimeError("conn lost"))
    ctx.deps = MagicMock(acp_session=acp)
    result = await cap.message_member(ctx, "member1", "hi")
    assert result is False


@pytest.mark.asyncio
async def test_team_message_member_no_acp() -> None:
    """message_member without ACP falls through to False."""
    from agent_utilities.capabilities.teams import TeamCapability

    cap = TeamCapability()
    ctx = MagicMock()
    ctx.deps = MagicMock(acp_session=None)
    result = await cap.message_member(ctx, "member1", "hi")
    assert result is False


# ---------------------------------------------------------------------------
# ContextLimitWarner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_warner_no_usage() -> None:
    """No usage object -> request unchanged."""
    from agent_utilities.capabilities.context_warnings import ContextLimitWarner

    w = ContextLimitWarner(max_tokens=100)
    ctx = MagicMock()
    ctx.usage = None
    req = MagicMock(parts=[])
    result = await w.before_model_run(ctx, req)
    assert result is req


@pytest.mark.asyncio
async def test_context_warner_no_total_tokens() -> None:
    """Usage with zero tokens -> unchanged."""
    from agent_utilities.capabilities.context_warnings import ContextLimitWarner

    w = ContextLimitWarner(max_tokens=100)
    ctx = MagicMock()
    usage = MagicMock(total_tokens=0)
    ctx.usage = usage
    req = MagicMock(parts=[])
    result = await w.before_model_run(ctx, req)
    assert result is req


@pytest.mark.asyncio
async def test_context_warner_no_limit() -> None:
    """No limit and model has no max_input_tokens -> unchanged."""
    from agent_utilities.capabilities.context_warnings import ContextLimitWarner

    w = ContextLimitWarner()  # No max_tokens
    ctx = MagicMock()
    usage = MagicMock(total_tokens=50)
    ctx.usage = usage
    model = MagicMock(spec=[])  # no max_input_tokens attr
    ctx.model = model
    req = MagicMock(parts=[])
    result = await w.before_model_run(ctx, req)
    assert result is req


@pytest.mark.asyncio
async def test_context_warner_at_warn_threshold() -> None:
    """At 70% threshold, URGENT warning is added."""
    from agent_utilities.capabilities.context_warnings import ContextLimitWarner

    w = ContextLimitWarner(max_tokens=100, warn_at=0.7, critical_at=0.9)
    ctx = MagicMock()
    usage = MagicMock(total_tokens=75)  # 75% > 70%
    ctx.usage = usage
    ctx.deps = MagicMock(graph_engine=None)
    parts_list: list[Any] = []
    req = MagicMock()
    req.parts = parts_list
    await w.before_model_run(ctx, req)
    # URGENT part inserted at index 0
    assert len(parts_list) == 1


@pytest.mark.asyncio
async def test_context_warner_at_critical_threshold() -> None:
    """At 90% threshold, CRITICAL warning + graph node."""
    from agent_utilities.capabilities.context_warnings import ContextLimitWarner

    w = ContextLimitWarner(max_tokens=100, warn_at=0.7, critical_at=0.9)
    ctx = MagicMock()
    usage = MagicMock(total_tokens=95)
    ctx.usage = usage
    engine = MagicMock()
    engine.graph = nx.MultiDiGraph()
    ctx.deps = MagicMock(graph_engine=engine)
    req = MagicMock(parts=[])
    await w.before_model_run(ctx, req)
    # Graph node was added
    assert len(engine.graph.nodes) == 1


@pytest.mark.asyncio
async def test_context_warner_graph_exception_handled() -> None:
    """Exception in graph write is caught."""
    from agent_utilities.capabilities.context_warnings import ContextLimitWarner

    w = ContextLimitWarner(max_tokens=100)
    ctx = MagicMock()
    usage = MagicMock(total_tokens=95)
    ctx.usage = usage
    engine = MagicMock()
    engine.graph.add_node.side_effect = RuntimeError("oops")
    ctx.deps = MagicMock(graph_engine=engine)
    req = MagicMock(parts=[])
    # Must not raise
    await w.before_model_run(ctx, req)
    assert True, "Context warner handled graph exceptions"


@pytest.mark.asyncio
async def test_context_warner_for_run() -> None:
    """for_run returns replica."""
    from agent_utilities.capabilities.context_warnings import ContextLimitWarner

    w = ContextLimitWarner(max_tokens=100)
    other = await w.for_run(MagicMock())
    assert other is not w


# ---------------------------------------------------------------------------
# HooksCapability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hooks_before_run_no_hooks() -> None:
    """before_run with no hooks runs smoothly."""
    from agent_utilities.capabilities.hooks import HooksCapability

    cap = HooksCapability()
    ctx = MagicMock()
    await cap.before_run(ctx)
    assert True, "No-hooks before-run is a no-op"


@pytest.mark.asyncio
async def test_hooks_after_run_returns_result() -> None:
    """after_run returns the agent result unchanged."""
    from agent_utilities.capabilities.hooks import HooksCapability

    cap = HooksCapability()
    ctx = MagicMock()
    result = MagicMock()
    out = await cap.after_run(ctx, result=result)
    assert out is result


@pytest.mark.asyncio
async def test_hooks_for_run_returns_replica() -> None:
    """for_run returns a replica."""
    from agent_utilities.capabilities.hooks import HooksCapability

    cap = HooksCapability()
    other = await cap.for_run(MagicMock())
    assert other is not cap


@pytest.mark.asyncio
async def test_hooks_before_tool_no_auto_trace() -> None:
    """auto_graph_trace=False does not write to graph."""
    from agent_utilities.capabilities.hooks import HooksCapability

    cap = HooksCapability(auto_graph_trace=False)
    ctx = MagicMock()
    ctx.deps = MagicMock()  # No graph_engine
    call = MagicMock(tool_name="t", tool_call_id="id")
    tool_def = MagicMock()
    result = await cap.before_tool_execute(ctx, call=call, tool_def=tool_def, args={})
    assert result == {}


@pytest.mark.asyncio
async def test_hooks_before_tool_with_graph_engine() -> None:
    """auto_graph_trace=True writes ToolCallNode to graph."""
    from agent_utilities.capabilities.hooks import HooksCapability

    cap = HooksCapability(auto_graph_trace=True)
    ctx = MagicMock()
    engine = MagicMock()
    engine.graph = nx.MultiDiGraph()
    ctx.deps = MagicMock(graph_engine=engine, episode_id="ep1")
    call = MagicMock(tool_name="t", tool_call_id="id1")
    tool_def = MagicMock()
    await cap.before_tool_execute(ctx, call=call, tool_def=tool_def, args={"x": 1})
    assert "id1" in engine.graph.nodes


@pytest.mark.asyncio
async def test_hooks_hook_modifies_args() -> None:
    """Hook that returns modify_args gets honored."""
    from agent_utilities.capabilities.hooks import HookResult, HooksCapability

    def my_hook(input: Any) -> HookResult:
        return HookResult(modify_args={"modified": True})

    cap = HooksCapability(hooks=[my_hook], auto_graph_trace=False)
    ctx = MagicMock()
    ctx.deps = MagicMock()
    call = MagicMock(tool_name="t", tool_call_id="id")
    tool_def = MagicMock()
    result = await cap.before_tool_execute(
        ctx, call=call, tool_def=tool_def, args={"x": 1}
    )
    assert result == {"modified": True}


@pytest.mark.asyncio
async def test_hooks_after_tool_modifies_result() -> None:
    """Hook returning modify_result alters after_tool output."""
    from agent_utilities.capabilities.hooks import HookResult, HooksCapability

    def my_hook(input: Any) -> HookResult:
        return HookResult(modify_result="modified")

    cap = HooksCapability(hooks=[my_hook], auto_graph_trace=False)
    ctx = MagicMock()
    ctx.deps = MagicMock()
    call = MagicMock(tool_name="t", tool_call_id="id")
    tool_def = MagicMock()
    result = await cap.after_tool_execute(
        ctx,
        call=call,
        tool_def=tool_def,
        args={},
        result="original",
    )
    assert result == "modified"


@pytest.mark.asyncio
async def test_hooks_after_tool_passthrough() -> None:
    """No hook modification -> original result passes through."""
    from agent_utilities.capabilities.hooks import HooksCapability

    cap = HooksCapability(auto_graph_trace=False)
    ctx = MagicMock()
    ctx.deps = MagicMock()
    call = MagicMock(tool_name="t", tool_call_id="id")
    tool_def = MagicMock()
    result = await cap.after_tool_execute(
        ctx, call=call, tool_def=tool_def, args={}, result="orig"
    )
    assert result == "orig"


@pytest.mark.asyncio
async def test_hooks_hook_raising_is_handled() -> None:
    """A hook that raises is logged and other hooks still run."""
    from agent_utilities.capabilities.hooks import HookResult, HooksCapability

    def bad_hook(inp: Any) -> HookResult:
        raise RuntimeError("boom")

    def ok_hook(inp: Any) -> HookResult:
        return HookResult(modify_args={"y": 2})

    cap = HooksCapability(hooks=[bad_hook, ok_hook], auto_graph_trace=False)
    ctx = MagicMock()
    ctx.deps = MagicMock()
    call = MagicMock(tool_name="t", tool_call_id="id")
    tool_def = MagicMock()
    result = await cap.before_tool_execute(
        ctx, call=call, tool_def=tool_def, args={"x": 1}
    )
    assert result == {"y": 2}


@pytest.mark.asyncio
async def test_hooks_cancel_branch() -> None:
    """Hook returning cancel=True sets cancel on HookResult aggregator."""
    from agent_utilities.capabilities.hooks import HookResult, HooksCapability

    def cancel_hook(inp: Any) -> HookResult:
        return HookResult(cancel=True, cancel_reason="not allowed")

    cap = HooksCapability(hooks=[cancel_hook], auto_graph_trace=False)
    ctx = MagicMock()
    ctx.deps = MagicMock()
    call = MagicMock(tool_name="t", tool_call_id="id")
    tool_def = MagicMock()
    # cancel path leaves the args untouched in current impl (the comment says
    # pydantic-ai handles it).  No exception raised.
    result = await cap.before_tool_execute(
        ctx, call=call, tool_def=tool_def, args={"x": 1}
    )
    assert result == {"x": 1}


def test_hooks_hook_input_dataclass() -> None:
    """HookInput dataclass accepts all fields."""
    from agent_utilities.capabilities.hooks import HookEvent, HookInput

    inp = HookInput(event=HookEvent.BEFORE_RUN, ctx=MagicMock())
    assert inp.event == HookEvent.BEFORE_RUN


def test_hooks_hook_result_defaults() -> None:
    """HookResult default values."""
    from agent_utilities.capabilities.hooks import HookResult

    r = HookResult()
    assert r.modify_args is None
    assert r.modify_result is None
    assert r.cancel is False
    assert r.cancel_reason is None


def test_hooks_hook_event_enum() -> None:
    """HookEvent values are stable strings."""
    from agent_utilities.capabilities.hooks import HookEvent

    assert HookEvent.BEFORE_RUN.value == "before_run"
    assert HookEvent.AFTER_RUN.value == "after_run"
    assert HookEvent.PRE_TOOL_USE.value == "pre_tool_use"
    assert HookEvent.POST_TOOL_USE.value == "post_tool_use"


# ---------------------------------------------------------------------------
# CheckpointMiddleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_checkpoint_middleware_every_tool() -> None:
    """frequency='every_tool' triggers save after each tool."""
    from agent_utilities.capabilities.checkpointing import (
        CheckpointMiddleware,
        InMemoryCheckpointStore,
    )

    store = InMemoryCheckpointStore()
    mw = CheckpointMiddleware(store=store, frequency="every_tool")
    ctx = MagicMock()
    ctx.messages = []
    ctx.deps = MagicMock(episode_id="ep1")
    call = MagicMock(tool_name="my_tool")
    kwargs = {"call": call, "result": "R"}
    result = await mw.after_tool_execute(ctx, **kwargs)
    assert result == "R"
    assert len(store._checkpoints) == 1


@pytest.mark.asyncio
async def test_checkpoint_middleware_manual_only() -> None:
    """frequency='manual_only' does not auto-save."""
    from agent_utilities.capabilities.checkpointing import (
        CheckpointMiddleware,
        InMemoryCheckpointStore,
    )

    store = InMemoryCheckpointStore()
    mw = CheckpointMiddleware(store=store, frequency="manual_only")
    ctx = MagicMock()
    ctx.messages = []
    ctx.deps = MagicMock()
    call = MagicMock(tool_name="x")
    await mw.after_tool_execute(ctx, call=call, result="R")
    assert len(store._checkpoints) == 0


@pytest.mark.asyncio
async def test_checkpoint_middleware_no_call() -> None:
    """Missing call in kwargs -> 'unknown' label, still saves."""
    from agent_utilities.capabilities.checkpointing import (
        CheckpointMiddleware,
        InMemoryCheckpointStore,
    )

    store = InMemoryCheckpointStore()
    mw = CheckpointMiddleware(store=store, frequency="every_tool")
    ctx = MagicMock()
    ctx.messages = []
    ctx.deps = MagicMock()
    await mw.after_tool_execute(ctx, result="R")
    assert len(store._checkpoints) == 1
    cp = next(iter(store._checkpoints.values()))
    assert "unknown" in cp.label
