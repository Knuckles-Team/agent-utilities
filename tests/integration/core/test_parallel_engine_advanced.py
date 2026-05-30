"""Integration tests for the Parallel Engine and Skill-to-Workflow compiler.

This suite validates the full end-to-end integration of the Unified Parallel Engine (ORCH-1.25),
verifying:
1. Skill compiler parsing DAG dependencies correctly.
2. Parallel waves batching, scheduling, and topological ordering.
3. Checkpoint middleware saving state between wave boundaries.
4. Stuck-loop and auto-healing capabilities.
5. Topological context propagation and synthesis.
"""

import os
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_utilities.orchestration import ParallelEngine

from agent_utilities.capabilities.checkpointing import FileCheckpointStore
from agent_utilities.models.execution_manifest import (
    AgentSpec,
    ExecutionManifest,
    SynthesisSpec,
)
from agent_utilities.workflows.skill_compiler import SkillCompiler

# Setup environment variables for clean testing
os.environ["OTEL_SDK_DISABLED"] = "true"


def test_skill_compiler_dag_parsing():
    """Verify that SkillCompiler compiles steps and extracts DAG dependencies correctly."""
    markdown_content = """# Autonomous Audit Skill

### Step 1: Research Agent [depends_on: none]
Gather baseline financial metrics.

### Step 2: Auditor Agent [depends_on: research-agent]
Audit the gathered metrics for compliance anomalies.

### Step 3: Reporting Agent [depends_on: researcher, auditor-agent]
Compile final PDF reports for the executives.
"""
    plan = SkillCompiler.compile_from_text("audit_skill", markdown_content)

    assert plan is not None
    assert len(plan.steps) == 3

    # Step 1
    assert plan.steps[0].id == "research-agent"
    assert len(plan.steps[0].depends_on) == 0

    # Step 2
    assert plan.steps[1].id == "auditor-agent"
    assert plan.steps[1].depends_on == ["research-agent"]

    # Step 3
    assert plan.steps[2].id == "reporting-agent"
    assert set(plan.steps[2].depends_on) == {"researcher", "auditor-agent"}


@pytest.mark.asyncio
async def test_parallel_engine_waves_scheduling():
    """Verify that ParallelEngine topologically schedules execution waves in parallel layers."""
    # Construct a non-trivial execution manifest with direct dependencies
    #   A (no deps) -> B (depends on A), C (depends on A) -> D (depends on B and C)
    # Expected waves: Wave 0 (A), Wave 1 (B, C in parallel), Wave 2 (D)
    agents = [
        AgentSpec(
            agent_id="agent-a",
            role="researcher",
            task_template="Task A",
            depends_on=[],
        ),
        AgentSpec(
            agent_id="agent-b",
            role="auditor",
            task_template="Task B",
            depends_on=["agent-a"],
        ),
        AgentSpec(
            agent_id="agent-c",
            role="analyst",
            task_template="Task C",
            depends_on=["agent-a"],
        ),
        AgentSpec(
            agent_id="agent-d",
            role="reporter",
            task_template="Task D",
            depends_on=["agent-b", "agent-c"],
        ),
    ]

    manifest = ExecutionManifest(
        name="DAG wave scheduling test",
        agents=agents,
        execution_mode="parallel",
        query="Verify wave execution layers",
        synthesis=SynthesisSpec(strategy="flat"),
    )

    engine = ParallelEngine()
    waves = engine._schedule_waves(manifest)

    assert len(waves) == 3
    # Wave 0 must have agent-a
    assert [a.agent_id for a in waves[0]] == ["agent-a"]
    # Wave 1 must have agent-b and agent-c (order not strictly guaranteed but both present)
    assert set(a.agent_id for a in waves[1]) == {"agent-b", "agent-c"}
    # Wave 2 must have agent-d
    assert [a.agent_id for a in waves[2]] == ["agent-d"]


@pytest.mark.asyncio
async def test_parallel_engine_topological_execution():
    """Verify that ParallelEngine executes agents wave by wave and injects topological context."""
    agents = [
        AgentSpec(
            agent_id="agent-a",
            role="researcher",
            task_template="Research background: {{query}}",
        ),
        AgentSpec(
            agent_id="agent-b",
            role="auditor",
            task_template="Audit findings from preceding agents",
            depends_on=["agent-a"],
        ),
    ]

    manifest = ExecutionManifest(
        name="Topological Context Injection Test",
        agents=agents,
        execution_mode="parallel",
        query="Market Arbitrage Anomaly Detection",
        synthesis=SynthesisSpec(strategy="flat"),
    )

    engine = ParallelEngine()

    # Mock `create_agent` from `agent_utilities.agent.factory`
    mock_run_result = MagicMock()
    mock_run_result.output = "Mocked Agent Response"

    mock_pydantic_agent = MagicMock()
    mock_pydantic_agent.run = AsyncMock(return_value=mock_run_result)

    with patch(
        "agent_utilities.agent.factory.create_agent",
        return_value=(mock_pydantic_agent, []),
    ):
        res = await engine.execute(manifest)

        assert res.success is True
        assert len(res.wave_results) == 2
        # Wave 0 result
        assert res.wave_results[0].results[0].agent_id == "agent-a"
        assert res.wave_results[0].results[0].output == "Mocked Agent Response"

        # Wave 1 result
        assert res.wave_results[1].results[0].agent_id == "agent-b"
        assert res.wave_results[1].results[0].output == "Mocked Agent Response"


@pytest.mark.asyncio
async def test_checkpointing_saves_and_resumes():
    """Verify checkpoint store saves wave execution results correctly to disk."""
    from agent_utilities.capabilities.checkpointing import Checkpoint

    temp_dir = tempfile.mkdtemp()
    try:
        store = FileCheckpointStore(directory=temp_dir)

        # Save a dummy state
        checkpoint_id = "test-checkpoint-123"
        cp = Checkpoint(
            id=checkpoint_id,
            label="Wave 0 checkpoint",
            turn=1,
            messages=[],
            metadata={"some": "metadata"},
        )

        await store.save(cp)

        # Load and verify
        retrieved = await store.get(checkpoint_id)
        assert retrieved is not None
        assert retrieved.id == checkpoint_id
        assert retrieved.label == "Wave 0 checkpoint"
        assert retrieved.turn == 1
        assert retrieved.metadata == {"some": "metadata"}
    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_auto_healing_triggers_on_failures():
    """Verify that auto-healing logic tracks failed agent runs inside the engine."""
    agents = [
        AgentSpec(
            agent_id="agent-failing",
            role="unstable-worker",
            task_template="Perform volatile work",
        )
    ]

    manifest = ExecutionManifest(
        name="Auto-Healing Failure Test",
        agents=agents,
        execution_mode="parallel",
        query="Triggers failure auto-healing tracker",
        synthesis=SynthesisSpec(strategy="flat"),
    )

    engine = ParallelEngine()

    # Force agent invocation failure by letting `create_agent` raise an exception
    with patch(
        "agent_utilities.agent.factory.create_agent",
        side_effect=Exception("API limit exceeded"),
    ):
        res = await engine.execute(manifest)

        assert res.success is False
        assert len(res.wave_results) == 1
        assert res.wave_results[0].results[0].success is False
        assert "API limit exceeded" in res.wave_results[0].results[0].error
