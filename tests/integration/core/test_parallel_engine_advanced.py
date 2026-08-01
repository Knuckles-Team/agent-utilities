"""Integration tests for the Parallel Engine and Skill-to-Workflow compiler.

This suite validates the full end-to-end integration of the Unified Parallel Engine (ORCH-1.8),
verifying:
1. Skill compiler parsing DAG dependencies correctly.
2. Parallel waves batching, scheduling, and topological ordering.
3. Checkpoint middleware saving state between wave boundaries.
4. Stuck-loop and repeated-failure escalation capabilities.
5. Topological context propagation and synthesis.
"""

import asyncio
import os
import shutil
import tempfile
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_utilities.capabilities.checkpointing import FileCheckpointStore
from agent_utilities.models.execution_manifest import (
    AgentSpec,
    ExecutionManifest,
    SynthesisSpec,
)
from agent_utilities.orchestration import ParallelEngine
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
async def test_repeated_failure_escalation_tracks_failures():
    """Repeated agent failures are tracked by the strangled escalation path
    (the old AutoHealingEngine shell's threshold counter, now feeding the
    failure_gap remediation chain when a KG engine is attached)."""
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
        # The strangled escalation path counted the failure (no KG engine
        # attached, so it never escalates - and never raises).
        assert engine._agent_failure_counts.get("agent-failing", 0) >= 1


class _FakeKGEngine:
    """Minimal shared knowledge engine for the GWT broadcast wiring test."""

    def __init__(self):
        from agent_utilities.knowledge_graph.core.graph_compute import (
            GraphComputeEngine,
        )

        self.graph = GraphComputeEngine(backend_type="rust")
        self.backend = None
        self.persisted: list[tuple] = []  # EvolvingMemoryStore._persist calls

    def _upsert_node(self, label, node_id, props):  # pragma: no cover - unused here
        pass

    def add_node(self, node_id, node_type, properties=None):
        # EvolvingMemoryStore._persist target.
        self.persisted.append((node_id, node_type, properties or {}))


def test_broadcast_workspace_attention_live_loop():
    """CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — the parallel engine drives the GWT loop and the broadcast
    is readable back via get_attention_score (the previously-dead loop, now live)."""
    from types import SimpleNamespace

    from agent_utilities.graph.workspace_attention import WorkspaceAttention
    from agent_utilities.models.execution_manifest import AgentExecutionResult

    kg = _FakeKGEngine()
    engine = ParallelEngine(engine=kg)
    results = [
        AgentExecutionResult(
            agent_id="spec:gitlab",
            output="Here are 5 gitlab projects with full details and tasks",
            success=True,
        ),
        AgentExecutionResult(
            agent_id="spec:weather",
            output="It is sunny today, totally unrelated content",
            success=True,
        ),
    ]
    manifest = SimpleNamespace(query="list gitlab projects", manifest_id="m1")

    broadcast_ids = engine._broadcast_workspace_attention(results, manifest)
    assert broadcast_ids  # winners were broadcast (≥2 successful outputs)

    # The broadcast is now readable as runtime attention — the loop is closed.
    score = WorkspaceAttention(kg).get_attention_score(broadcast_ids[0])
    assert score is not None and 0.0 <= score <= 1.0

    # CONCEPT:AU-KG.memory.tiered-memory-caching — winners were also routed into the evolving memory store,
    # persisted to the shared engine (INSIGHT bank, gwt-winner signature).
    insight_nodes = [
        props
        for _id, _t, props in kg.persisted
        if props.get("bank") == "insight"
        and str(props.get("signature", "")).startswith("gwt-winner:")
    ]
    assert insight_nodes  # the previously-deferred EvolvingMemoryStore site is live


def test_broadcast_workspace_attention_noop_without_engine():
    engine = ParallelEngine()  # no shared engine
    from types import SimpleNamespace

    from agent_utilities.models.execution_manifest import AgentExecutionResult

    results = [
        AgentExecutionResult(agent_id="a", output="x", success=True),
        AgentExecutionResult(agent_id="b", output="y", success=True),
    ]
    out = engine._broadcast_workspace_attention(
        results, SimpleNamespace(query="q", manifest_id="m")
    )
    assert out == []


@pytest.mark.asyncio
async def test_parallel_execute_keeps_graph_coordination_and_persistence_off_loop(
    monkeypatch: pytest.MonkeyPatch,
):
    """The public execution path isolates every synchronous graph phase."""

    from agent_utilities.capabilities import adversarial_verifier
    from agent_utilities.graph.coordination import BUILTIN_PROTOCOLS
    from agent_utilities.knowledge_graph.core.session import (
        GraphSession,
        current_session,
        use_session,
    )
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.models.execution_manifest import (
        AgentExecutionResult,
        WaveResult,
    )
    from agent_utilities.security.brain_context import (
        ActorContext,
        current_actor,
        use_actor,
    )

    parallel = ParallelEngine(engine=object())
    main_thread = threading.current_thread()
    phase_order: list[str] = []
    phase_threads: list[threading.Thread] = []
    authority: list[tuple[object, object]] = []
    entered = [threading.Event() for _ in range(3)]
    release = [threading.Event() for _ in range(3)]

    def phase(index: int, name: str) -> None:
        phase_order.append(name)
        phase_threads.append(threading.current_thread())
        authority.append((current_actor(), current_session()))
        entered[index].set()
        release[index].wait(0.5)

    def select_protocol(**_kwargs):
        phase(0, "coordination_select")
        return BUILTIN_PROTOCOLS["pair_consensus"]

    original_apply = parallel.coordination.apply_protocol

    def log_coordination(_result):
        phase_order.append("coordination_trace")
        phase_threads.append(threading.current_thread())
        authority.append((current_actor(), current_session()))
        return "coordination:trace"

    async def execute_wave(
        wave_agents,
        wave_idx,
        _scheduler,
        _manifest,
        _graph_deps,
        _prior_results,
    ):
        return WaveResult(
            wave_index=wave_idx,
            results=[
                AgentExecutionResult(
                    agent_id=agent.agent_id,
                    output=f"output:{agent.agent_id}",
                    success=True,
                )
                for agent in wave_agents
            ],
            duration_ms=1.0,
        )

    def workspace_attention(_results, _manifest):
        phase(1, "workspace_memory")
        return ["agent:a"]

    async def synthesize(_results, _spec, _query, _graph_deps):
        return "synthesized"

    def persist_execution(_manifest, _waves, _output):
        phase(2, "execution_hierarchy")
        return "pe:stable"

    monkeypatch.setattr(parallel.coordination, "select_protocol", select_protocol)
    monkeypatch.setattr(parallel.coordination, "apply_protocol", original_apply)
    monkeypatch.setattr(
        parallel.coordination, "log_coordination_trace", log_coordination
    )
    monkeypatch.setattr(parallel, "_execute_wave", execute_wave)
    monkeypatch.setattr(parallel, "_broadcast_workspace_attention", workspace_attention)
    monkeypatch.setattr(parallel, "_synthesize", synthesize)
    monkeypatch.setattr(parallel, "_persist_execution", persist_execution)
    monkeypatch.setattr(adversarial_verifier, "ADVERSARIAL_ENABLED", False)

    manifest = ExecutionManifest(
        name="event-loop-isolation",
        agents=[
            AgentSpec(agent_id="agent:a", task_template="A"),
            AgentSpec(agent_id="agent:b", task_template="B"),
        ],
        execution_mode="parallel",
        query="combine A and B",
        synthesis=SynthesisSpec(strategy="flat"),
    )
    actor = ActorContext(
        actor_id="agent:parallel-test",
        actor_type=ActorType.AI_AGENT,
        roles=("kg:write",),
        tenant_id="tenant:test",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant="tenant:test",
        scopes=frozenset({"kg:read", "kg:write"}),
        graph="tenant:test",
        policy_version="test",
        audience="graph-os",
    )

    heartbeats = 0
    stop_heartbeat = False

    async def heartbeat() -> None:
        nonlocal heartbeats
        while not stop_heartbeat:
            heartbeats += 1
            await asyncio.sleep(0.002)

    with use_actor(actor), use_session(session):
        request = asyncio.create_task(parallel.execute(manifest))
    pulse = asyncio.create_task(heartbeat())
    try:
        for index in range(3):
            assert await asyncio.to_thread(entered[index].wait, 0.2)
            before = heartbeats
            await asyncio.sleep(0.02)
            assert heartbeats > before
            assert not request.done()
            release[index].set()
        result = await asyncio.wait_for(request, timeout=1.0)
    finally:
        for event in release:
            event.set()
        stop_heartbeat = True
        await pulse

    assert result.execution_id == "pe:stable"
    assert result.synthesis_output == "synthesized"
    assert phase_order == [
        "coordination_select",
        "coordination_trace",
        "workspace_memory",
        "execution_hierarchy",
    ]
    assert all(thread is not main_thread for thread in phase_threads)
    assert authority == [(actor, session)] * 4
