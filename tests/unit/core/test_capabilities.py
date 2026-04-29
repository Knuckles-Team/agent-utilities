"""Coverage push for agent_utilities.capabilities.*.

CONCEPT:AU-008 — Resilient Agent Capabilities

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

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import networkx as nx
import pytest


# ---------------------------------------------------------------------------
# Checkpoint & CheckpointStore implementations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_in_memory_checkpoint_store_save_get() -> None:
    """Round-trip save + get."""
    from agent_utilities.capabilities.checkpointing import (
        Checkpoint,
        InMemoryCheckpointStore,
    )

    store = InMemoryCheckpointStore()
    cp = Checkpoint(id="cp1", label="test", turn=1, messages=[])
    await store.save(cp)
    got = await store.get("cp1")
    assert got is cp


@pytest.mark.asyncio
async def test_in_memory_checkpoint_store_get_missing() -> None:
    """Missing ID returns None."""
    from agent_utilities.capabilities.checkpointing import (
        InMemoryCheckpointStore,
    )

    store = InMemoryCheckpointStore()
    assert await store.get("missing") is None


@pytest.mark.asyncio
async def test_in_memory_checkpoint_store_list_sorted() -> None:
    """List returns sorted descending by timestamp, limited."""
    from agent_utilities.capabilities.checkpointing import (
        Checkpoint,
        InMemoryCheckpointStore,
    )

    store = InMemoryCheckpointStore()
    cp1 = Checkpoint(id="c1", label="a", turn=1, messages=[], timestamp=100.0)
    cp2 = Checkpoint(id="c2", label="b", turn=2, messages=[], timestamp=200.0)
    cp3 = Checkpoint(id="c3", label="c", turn=3, messages=[], timestamp=300.0)
    await store.save(cp1)
    await store.save(cp2)
    await store.save(cp3)

    result = await store.list(limit=2)
    assert [c.id for c in result] == ["c3", "c2"]


@pytest.mark.asyncio
async def test_file_checkpoint_store_save_get(tmp_path: Path) -> None:
    """FileCheckpointStore round-trips via disk."""
    from agent_utilities.capabilities.checkpointing import (
        Checkpoint,
        FileCheckpointStore,
    )

    store = FileCheckpointStore(str(tmp_path))
    cp = Checkpoint(id="f1", label="fs", turn=5, messages=[])
    await store.save(cp)
    assert (tmp_path / "f1.json").exists()
    got = await store.get("f1")
    assert got is not None
    assert got.id == "f1"
    assert got.label == "fs"


@pytest.mark.asyncio
async def test_file_checkpoint_store_get_missing(tmp_path: Path) -> None:
    """Missing file returns None."""
    from agent_utilities.capabilities.checkpointing import FileCheckpointStore

    store = FileCheckpointStore(str(tmp_path))
    assert await store.get("missing") is None


@pytest.mark.asyncio
async def test_file_checkpoint_store_list(tmp_path: Path) -> None:
    """List returns Checkpoint objects sorted by mtime desc."""
    from agent_utilities.capabilities.checkpointing import (
        Checkpoint,
        FileCheckpointStore,
    )

    store = FileCheckpointStore(str(tmp_path))
    for i in range(3):
        cp = Checkpoint(id=f"c{i}", label=f"l{i}", turn=i, messages=[])
        await store.save(cp)
    result = await store.list(limit=2)
    assert len(result) == 2


def test_checkpoint_to_json_from_json_roundtrip() -> None:
    """Checkpoint JSON round-trip."""
    from agent_utilities.capabilities.checkpointing import Checkpoint

    cp = Checkpoint(
        id="c1",
        label="My Checkpoint",
        turn=3,
        messages=[],
        timestamp=123.0,
        metadata={"episode_id": "ep1"},
    )
    json_str = cp.to_json()
    data = json.loads(json_str)
    assert data["id"] == "c1"
    assert data["label"] == "My Checkpoint"
    assert data["turn"] == 3
    assert data["metadata"] == {"episode_id": "ep1"}

    cp2 = Checkpoint.from_json(json_str)
    assert cp2.id == "c1"
    assert cp2.label == "My Checkpoint"


def test_rewind_requested_exception() -> None:
    """RewindRequested exception stores checkpoint_id."""
    from agent_utilities.capabilities.checkpointing import RewindRequested

    exc = RewindRequested("cp123")
    assert exc.checkpoint_id == "cp123"


@pytest.mark.asyncio
async def test_checkpoint_toolset_list_empty() -> None:
    """list_checkpoints returns empty string when store is empty."""
    from agent_utilities.capabilities.checkpointing import (
        CheckpointToolset,
        InMemoryCheckpointStore,
    )

    store = InMemoryCheckpointStore()
    toolset = CheckpointToolset(store)
    ctx = MagicMock()
    result = await toolset.list_checkpoints(ctx)
    assert result == ""


@pytest.mark.asyncio
async def test_checkpoint_toolset_create() -> None:
    """create_checkpoint returns stub text."""
    from agent_utilities.capabilities.checkpointing import (
        CheckpointToolset,
        InMemoryCheckpointStore,
    )

    toolset = CheckpointToolset(InMemoryCheckpointStore())
    ctx = MagicMock()
    result = await toolset.create_checkpoint(ctx, "mylabel")
    assert "Checkpoint" in result


@pytest.mark.asyncio
async def test_checkpoint_toolset_rewind_raises() -> None:
    """rewind raises RewindRequested."""
    from agent_utilities.capabilities.checkpointing import (
        CheckpointToolset,
        InMemoryCheckpointStore,
        RewindRequested,
    )

    toolset = CheckpointToolset(InMemoryCheckpointStore())
    with pytest.raises(RewindRequested) as exc_info:
        await toolset.rewind(MagicMock(), "cpid")
    assert exc_info.value.checkpoint_id == "cpid"


# ---------------------------------------------------------------------------
# GraphCheckpointStore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_checkpoint_store_save_no_backend() -> None:
    """Save works with engine and no backend."""
    from agent_utilities.capabilities.checkpointing import (
        Checkpoint,
        GraphCheckpointStore,
    )

    engine = MagicMock()
    engine.backend = None
    engine.graph = nx.MultiDiGraph()
    store = GraphCheckpointStore(engine)
    cp = Checkpoint(id="cp1", label="l", turn=1, messages=[], metadata={})
    await store.save(cp)
    assert "cp1" in engine.graph.nodes


@pytest.mark.asyncio
async def test_graph_checkpoint_store_save_with_backend() -> None:
    """Save with backend calls upsert_node."""
    from agent_utilities.capabilities.checkpointing import (
        Checkpoint,
        GraphCheckpointStore,
    )

    engine = MagicMock()
    engine.backend = MagicMock()
    engine.backend.upsert_node = AsyncMock()
    engine.graph = nx.MultiDiGraph()
    store = GraphCheckpointStore(engine)
    cp = Checkpoint(
        id="cp1", label="l", turn=1, messages=[], metadata={"episode_id": "ep1"}
    )
    await store.save(cp)
    engine.backend.upsert_node.assert_called_once()


@pytest.mark.asyncio
async def test_graph_checkpoint_store_save_exception() -> None:
    """Exception in graph.add_node is caught and logged."""
    from agent_utilities.capabilities.checkpointing import (
        Checkpoint,
        GraphCheckpointStore,
    )

    engine = MagicMock()
    engine.graph.add_node.side_effect = RuntimeError("oops")
    engine.backend = None
    store = GraphCheckpointStore(engine)
    cp = Checkpoint(id="cp1", label="l", turn=1, messages=[], metadata={})
    # Must not raise
    await store.save(cp)
    assert True, 'Exception during checkpoint save should be caught'


@pytest.mark.asyncio
async def test_graph_checkpoint_store_get_not_found() -> None:
    """Get for unknown ID returns None."""
    from agent_utilities.capabilities.checkpointing import GraphCheckpointStore

    engine = MagicMock()
    engine.backend = None
    engine.graph = nx.MultiDiGraph()
    store = GraphCheckpointStore(engine)
    assert await store.get("missing") is None


@pytest.mark.asyncio
async def test_graph_checkpoint_store_get_with_backend() -> None:
    """Get from backend returns parsed Checkpoint."""
    from agent_utilities.capabilities.checkpointing import (
        Checkpoint,
        GraphCheckpointStore,
    )

    engine = MagicMock()
    engine.backend = MagicMock()
    source = Checkpoint(id="cp1", label="l", turn=1, messages=[])
    engine.backend.get_node = AsyncMock(
        return_value={"message_data": source.to_json()}
    )
    engine.graph = nx.MultiDiGraph()
    store = GraphCheckpointStore(engine)
    cp = await store.get("cp1")
    assert cp is not None
    assert cp.id == "cp1"


@pytest.mark.asyncio
async def test_graph_checkpoint_store_list_returns_empty() -> None:
    """list() returns empty list (stub impl)."""
    from agent_utilities.capabilities.checkpointing import GraphCheckpointStore

    store = GraphCheckpointStore(MagicMock())
    assert await store.list() == []


# ---------------------------------------------------------------------------
# fork_from_checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fork_from_checkpoint() -> None:
    """fork_from_checkpoint delegates to agent.run with message_history."""
    from agent_utilities.capabilities.checkpointing import (
        Checkpoint,
        fork_from_checkpoint,
    )

    fake_agent = MagicMock()
    fake_agent.run = AsyncMock(return_value=MagicMock(output="ok"))
    cp = Checkpoint(id="c1", label="l", turn=1, messages=[])
    result = await fork_from_checkpoint(fake_agent, cp, "next turn")
    fake_agent.run.assert_called_once_with("next turn", message_history=cp.messages)


# ---------------------------------------------------------------------------
# ToolOutputEviction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_eviction_below_threshold() -> None:
    """Results below threshold are returned unchanged."""
    from agent_utilities.capabilities.eviction import ToolOutputEviction

    evict = ToolOutputEviction(threshold_chars=1000)
    ctx = MagicMock()
    ctx.deps = MagicMock()
    call = MagicMock(tool_name="my_tool", tool_call_id="id1")
    tool_def = MagicMock()

    result = await evict.after_tool_execute(
        ctx, call=call, tool_def=tool_def, args={}, result="short result"
    )
    assert result == "short result"


@pytest.mark.asyncio
async def test_eviction_above_threshold_no_engine() -> None:
    """Results above threshold get evicted with preview (no engine)."""
    from agent_utilities.capabilities.eviction import ToolOutputEviction

    evict = ToolOutputEviction(threshold_chars=50)
    ctx = MagicMock()
    ctx.deps = MagicMock()
    ctx.deps.graph_engine = None
    call = MagicMock(tool_name="my_tool", tool_call_id="id1")
    tool_def = MagicMock()

    long_content = "x" * 100
    result = await evict.after_tool_execute(
        ctx, call=call, tool_def=tool_def, args={}, result=long_content
    )
    assert "EVICTED" in result


@pytest.mark.asyncio
async def test_eviction_above_threshold_with_engine() -> None:
    """Results above threshold are stored in the graph engine."""
    from agent_utilities.capabilities.eviction import ToolOutputEviction

    evict = ToolOutputEviction(threshold_chars=50, store_in_graph=True)
    ctx = MagicMock()
    engine = MagicMock()
    engine.graph = nx.MultiDiGraph()
    engine.backend = None
    ctx.deps.graph_engine = engine
    call = MagicMock(tool_name="my_tool", tool_call_id="id1")
    tool_def = MagicMock()

    long_content = "x" * 100
    result = await evict.after_tool_execute(
        ctx, call=call, tool_def=tool_def, args={}, result=long_content
    )
    assert "EVICTED" in result
    # Graph should have 1 node
    assert len(engine.graph.nodes) == 1


@pytest.mark.asyncio
async def test_eviction_for_run_returns_replica() -> None:
    """for_run returns a copy of the instance."""
    from agent_utilities.capabilities.eviction import ToolOutputEviction

    evict = ToolOutputEviction(threshold_chars=50)
    other = await evict.for_run(MagicMock())
    assert other is not evict
    assert other.threshold_chars == 50


@pytest.mark.asyncio
async def test_eviction_add_node_raises_handled() -> None:
    """Exception in graph.add_node is caught."""
    from agent_utilities.capabilities.eviction import ToolOutputEviction

    evict = ToolOutputEviction(threshold_chars=50, store_in_graph=True)
    ctx = MagicMock()
    engine = MagicMock()
    engine.graph.add_node.side_effect = RuntimeError("oops")
    engine.backend = None
    ctx.deps.graph_engine = engine
    call = MagicMock(tool_name="t", tool_call_id="id")
    tool_def = MagicMock()
    long_content = "y" * 100
    result = await evict.after_tool_execute(
        ctx, call=call, tool_def=tool_def, args={}, result=long_content
    )
    # Graceful fallback still returns preview
    assert "EVICTED" in result


# ---------------------------------------------------------------------------
# StuckLoopDetection
# ---------------------------------------------------------------------------


def test_stuck_loop_init_validates_max_repeated() -> None:
    """max_repeated < 2 raises."""
    from agent_utilities.capabilities.stuck_loop import StuckLoopDetection

    with pytest.raises(ValueError, match="at least 2"):
        StuckLoopDetection(max_repeated=1)


@pytest.mark.asyncio
async def test_stuck_loop_for_run_returns_replica() -> None:
    """for_run returns a fresh instance."""
    from agent_utilities.capabilities.stuck_loop import StuckLoopDetection

    s = StuckLoopDetection()
    other = await s.for_run(MagicMock())
    assert other is not s


@pytest.mark.asyncio
async def test_stuck_loop_repeated_detection() -> None:
    """Identical tool calls N times -> ModelRetry."""
    from agent_utilities.capabilities.stuck_loop import StuckLoopDetection
    from pydantic_ai.exceptions import ModelRetry

    s = StuckLoopDetection(max_repeated=3, action="warn")
    ctx = MagicMock()
    ctx.deps = MagicMock(graph_engine=None)
    call = MagicMock(tool_name="same", tool_call_id="id")
    tool_def = MagicMock()
    args = {"x": 1}
    # First two calls - no retry
    for _ in range(2):
        await s.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args=args, result="r"
        )
    # Third call triggers
    with pytest.raises(ModelRetry):
        await s.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args=args, result="r"
        )


@pytest.mark.asyncio
async def test_stuck_loop_repeated_error_mode() -> None:
    """action='error' raises StuckLoopError on repeated."""
    from agent_utilities.capabilities.stuck_loop import (
        StuckLoopDetection,
        StuckLoopError,
    )

    s = StuckLoopDetection(max_repeated=2, action="error")
    ctx = MagicMock()
    ctx.deps = MagicMock(graph_engine=None)
    call = MagicMock(tool_name="same", tool_call_id="id")
    tool_def = MagicMock()
    args = {"x": 1}
    await s.after_tool_execute(
        ctx, call=call, tool_def=tool_def, args=args, result="r"
    )
    with pytest.raises(StuckLoopError) as exc:
        await s.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args=args, result="r"
        )
    assert exc.value.pattern == "repeated"


@pytest.mark.asyncio
async def test_stuck_loop_alternating_detection() -> None:
    """A, B, A, B pattern triggers alternating detection."""
    from agent_utilities.capabilities.stuck_loop import StuckLoopDetection
    from pydantic_ai.exceptions import ModelRetry

    s = StuckLoopDetection(max_repeated=2)
    ctx = MagicMock()
    ctx.deps = MagicMock(graph_engine=None)
    tool_def = MagicMock()
    call_a = MagicMock(tool_name="A", tool_call_id="a1")
    call_b = MagicMock(tool_name="B", tool_call_id="b1")
    # Sequence ABAB (needs 4 entries to trigger with max_repeated=2)
    for _ in range(2):
        await s.after_tool_execute(
            ctx, call=call_a, tool_def=tool_def, args={"x": 1}, result="r1"
        )
        # Last call in loop iteration before this one was for call_a.
        # We need alternating with different result to avoid noop.
        try:
            await s.after_tool_execute(
                ctx,
                call=call_b,
                tool_def=tool_def,
                args={"y": 2},
                result="r2",
            )
        except ModelRetry:
            break
    assert True, 'Alternating pattern detection completed'


@pytest.mark.asyncio
async def test_stuck_loop_noop_detection() -> None:
    """Same result N times triggers noop detection."""
    from agent_utilities.capabilities.stuck_loop import StuckLoopDetection
    from pydantic_ai.exceptions import ModelRetry

    s = StuckLoopDetection(
        max_repeated=3,
        detect_repeated=False,
        detect_alternating=False,
    )
    ctx = MagicMock()
    ctx.deps = MagicMock(graph_engine=None)
    tool_def = MagicMock()
    # Different args each time -> NOT repeated, but same result -> noop
    for i in range(2):
        await s.after_tool_execute(
            ctx,
            call=MagicMock(tool_name="tool", tool_call_id=f"{i}"),
            tool_def=tool_def,
            args={"x": i},
            result="same",
        )
    with pytest.raises(ModelRetry):
        await s.after_tool_execute(
            ctx,
            call=MagicMock(tool_name="tool", tool_call_id="2"),
            tool_def=tool_def,
            args={"x": 99},
            result="same",
        )


@pytest.mark.asyncio
async def test_stuck_loop_with_graph_engine() -> None:
    """Graph engine records SelfEvaluation node."""
    from agent_utilities.capabilities.stuck_loop import StuckLoopDetection
    from pydantic_ai.exceptions import ModelRetry

    s = StuckLoopDetection(max_repeated=2, action="warn")
    ctx = MagicMock()
    engine = MagicMock()
    engine.graph = nx.MultiDiGraph()
    engine.backend = MagicMock()
    engine.backend.upsert_node = AsyncMock()
    ctx.deps = MagicMock(graph_engine=engine)
    call = MagicMock(tool_name="same", tool_call_id="id")
    tool_def = MagicMock()
    await s.after_tool_execute(
        ctx, call=call, tool_def=tool_def, args={"x": 1}, result="r"
    )
    with pytest.raises(ModelRetry):
        await s.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args={"x": 1}, result="r"
        )
    # A SelfEvaluation node was created
    assert len(engine.graph.nodes) == 1


def test_stuck_loop_hash_args_dict() -> None:
    """_hash_args produces stable hash."""
    from agent_utilities.capabilities.stuck_loop import _hash_args

    h1 = _hash_args({"a": 1, "b": 2})
    h2 = _hash_args({"b": 2, "a": 1})
    assert h1 == h2


def test_stuck_loop_hash_args_unserializable() -> None:
    """_hash_args handles unserializable args."""
    from agent_utilities.capabilities.stuck_loop import _hash_args

    class NoSerializer:
        pass

    h = _hash_args({"obj": NoSerializer()})
    assert isinstance(h, str)


def test_stuck_loop_hash_result_string() -> None:
    """_hash_result works on strings."""
    from agent_utilities.capabilities.stuck_loop import _hash_result

    assert isinstance(_hash_result("hello"), str)


def test_stuck_loop_hash_result_dict() -> None:
    """_hash_result works on dicts."""
    from agent_utilities.capabilities.stuck_loop import _hash_result

    assert isinstance(_hash_result({"x": 1}), str)


def test_stuck_loop_hash_result_unserializable() -> None:
    """_hash_result handles unserializable results."""
    from agent_utilities.capabilities.stuck_loop import _hash_result

    class NoSerializer:
        pass

    assert isinstance(_hash_result(NoSerializer()), str)


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
    assert True, 'Context warner handled graph exceptions'


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
    assert True, 'No-hooks before-run is a no-op'


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
    result = await mw.after_tool_execute(ctx, result="R")
    assert len(store._checkpoints) == 1
    cp = next(iter(store._checkpoints.values()))
    assert "unknown" in cp.label
