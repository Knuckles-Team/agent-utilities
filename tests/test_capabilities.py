#!/usr/bin/python
# coding: utf-8
"""Tests for agent capabilities (reliability, session resilience, teams)."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, ANY
from pydantic_ai import RunContext, Agent
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

from agent_utilities.capabilities import (
    StuckLoopDetection, StuckLoopError,
    HooksCapability, HookEvent,
    ContextLimitWarner,
    ToolOutputEviction,
    CheckpointMiddleware, InMemoryCheckpointStore,
    TeamCapability
)
from agent_utilities.models.knowledge_graph import RegistryNodeType

@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.graph = MagicMock()
    engine.graph.nodes = {}
    engine.graph.__contains__.side_effect = lambda x: x in engine.graph.nodes
    def add_node(node_id, **kwargs):
        engine.graph.nodes[node_id] = kwargs
    engine.graph.add_node.side_effect = add_node
    engine.backend = AsyncMock()
    return engine

@pytest.fixture
def mock_deps(mock_engine):
    deps = MagicMock()
    deps.graph_engine = mock_engine
    deps.episode_id = "test_episode"
    deps.agent_id = "test_agent"
    return deps

@pytest.mark.asyncio
async def test_stuck_loop_repeated(mock_deps):
    cap = StuckLoopDetection(max_repeated=2, action="error")
    ctx = MagicMock(spec=RunContext)
    ctx.deps = mock_deps

    tool_def = ToolDefinition(name="test_tool", description="test")
    call = ToolCallPart(tool_name="test_tool", args={"a": 1}, tool_call_id="1")

    # First call
    await cap.after_tool_execute(ctx, call=call, tool_def=tool_def, args={"a": 1}, result="ok")

    # Second call (identical) -> should raise
    with pytest.raises(StuckLoopError) as exc:
        await cap.after_tool_execute(ctx, call=call, tool_def=tool_def, args={"a": 1}, result="ok")

    assert "identical arguments" in str(exc.value)
    # Verify graph write
    mock_deps.graph_engine.graph.add_node.assert_called()

@pytest.mark.asyncio
async def test_hooks_tracing(mock_deps):
    cap = HooksCapability(auto_graph_trace=True)
    ctx = MagicMock(spec=RunContext)
    ctx.deps = mock_deps

    tool_def = ToolDefinition(name="test_tool", description="test")
    call = ToolCallPart(tool_name="test_tool", args={"a": 1}, tool_call_id="1")

    # Before
    await cap.before_tool_execute(ctx, call=call, tool_def=tool_def, args={"a": 1})
    mock_deps.graph_engine.graph.add_node.assert_called()
    assert mock_deps.graph_engine.graph.add_node.call_args[0][0] == "1"

    # After
    await cap.after_tool_execute(ctx, call=call, tool_def=tool_def, args={"a": 1}, result="done")
    # Verify result update
    assert mock_deps.graph_engine.graph.nodes["1"]["result"] == "done"

@pytest.mark.asyncio
async def test_context_warner(mock_deps):
    cap = ContextLimitWarner(warn_at=0.5, critical_at=0.8, max_tokens=1000)

    # Mock usage
    ctx = MagicMock()
    ctx.usage.total_tokens = 600 # 60%
    ctx.deps = mock_deps
    ctx.model.max_input_tokens = 1000

    from pydantic_ai.messages import ModelRequest, SystemPromptPart
    req = ModelRequest(parts=[])

    # Should inject URGENT
    new_req = await cap.before_model_run(ctx, req)
    assert isinstance(new_req.parts[0], SystemPromptPart)
    assert "URGENT" in new_req.parts[0].content

    # Should inject CRITICAL and write to graph
    ctx.usage.total_tokens = 900 # 90%
    new_req = await cap.before_model_run(ctx, req)
    assert isinstance(new_req.parts[0], SystemPromptPart)
    assert "CRITICAL" in new_req.parts[0].content
    mock_deps.graph_engine.graph.add_node.assert_called()

@pytest.mark.asyncio
async def test_output_eviction(mock_deps):
    cap = ToolOutputEviction(threshold_chars=10)
    ctx = MagicMock(spec=RunContext)
    ctx.deps = mock_deps

    tool_def = ToolDefinition(name="test_tool", description="test")
    call = ToolCallPart(tool_name="test_tool", args={}, tool_call_id="1")

    large_result = "This is a very long result " * 1000

    final_result = await cap.after_tool_execute(ctx, call=call, tool_def=tool_def, args={}, result=large_result)

    assert "EVICTED" in final_result
    assert len(final_result) < len(large_result)
    mock_deps.graph_engine.graph.add_node.assert_called()

@pytest.mark.asyncio
async def test_checkpointing(mock_deps):
    store = InMemoryCheckpointStore()
    cap = CheckpointMiddleware(store=store, frequency="every_tool")

    ctx = MagicMock()
    ctx.deps = mock_deps
    ctx.all_messages.return_value = []

    tool_def = ToolDefinition(name="test_tool", description="test")
    call = ToolCallPart(tool_name="test_tool", args={}, tool_call_id="1")

    await cap.after_tool_execute(ctx, call=call, tool_def=tool_def, args={}, result="ok")

    ckpts = await store.list()
    assert len(ckpts) == 1
    assert ckpts[0].label == "After tool: test_tool"

@pytest.mark.asyncio
async def test_teams(mock_deps):
    cap = TeamCapability()
    ctx = MagicMock(spec=RunContext)
    ctx.deps = mock_deps

    team_id = await cap.create_team(ctx, "alpha", ["agent1", "agent2"])
    assert team_id.startswith("team_")

    task_id = await cap.add_task(ctx, "do work", assigned_to="agent1")
    assert task_id.startswith("task_")

    # Verify graph edges
    mock_deps.graph_engine.graph.add_edge.assert_called()
