"""Truthful Pydantic Graph topology, transition, and checkpoint evidence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic_graph import GraphBuilder

from agent_utilities.core.checkpoint.manager import KGBackend
from agent_utilities.graph import _router_impl
from agent_utilities.graph.state import GraphDeps, GraphState
from agent_utilities.orchestration import AgentOrchestrationEngine
from agent_utilities.orchestration.graph_execution_evidence import (
    GraphExecutionEvidenceCollector,
    run_with_execution_evidence,
)

pytestmark = pytest.mark.concept("AU-OS.observability.telemetry-observability")


def _build_fixture_graph():
    graph_builder = GraphBuilder(
        name="execution-evidence-fixture",
        state_type=GraphState,
        deps_type=GraphDeps,
        output_type=str,
    )

    async def route(_ctx):
        return "routed"

    async def execute(ctx):
        return f"{ctx.inputs}:executed"

    route_node = graph_builder.step(route, node_id="route")
    execute_node = graph_builder.step(execute, node_id="execute")
    graph_builder.add(
        graph_builder.edge_from(graph_builder.start_node).to(route_node),
        graph_builder.edge_from(route_node).to(execute_node),
        graph_builder.edge_from(execute_node).to(graph_builder.end_node),
    )
    return graph_builder.build()


def _deps() -> GraphDeps:
    return GraphDeps(
        tag_prompts={},
        tag_env_vars={},
        mcp_toolsets=[],
        router_model=None,
        agent_model=None,
    )


@pytest.mark.asyncio
async def test_real_graph_run_records_every_scheduler_transition() -> None:
    graph = _build_fixture_graph()
    state = GraphState(query="synthetic", session_id="run:evidence")
    collector = GraphExecutionEvidenceCollector(graph, topology="fixture")
    collector.bind_state(state)

    output = await run_with_execution_evidence(
        graph,
        state=state,
        deps=_deps(),
        collector=collector,
    )
    evidence = collector.evidence(state=state)

    assert output == "routed:executed"
    assert evidence.runtime_version
    assert evidence.topology_digest.startswith("sha256:")
    assert evidence.version_digest.startswith("sha256:")
    assert evidence.node_sequence == ["route", "execute", "__end__"]
    assert [
        [task.node_id for task in transition.scheduled_tasks]
        for transition in evidence.transitions
    ] == [["route"], ["execute"], ["__end__"]]
    assert [transition.sequence for transition in evidence.transitions] == [1, 2, 3]
    assert state.graph_node_sequence == evidence.node_sequence
    assert state.graph_transition_sequence == [
        transition.model_dump() for transition in evidence.transitions
    ]
    assert evidence.checkpoint_ids == []
    assert evidence.resume_supported is False


@pytest.mark.asyncio
async def test_real_graph_run_keeps_parallel_tasks_in_one_transition() -> None:
    graph_builder = GraphBuilder(
        name="parallel-evidence-fixture",
        state_type=GraphState,
        deps_type=GraphDeps,
        output_type=str,
    )

    async def root(_ctx):
        return "root"

    async def left(_ctx):
        return "left"

    async def right(_ctx):
        return "right"

    root_node = graph_builder.step(root, node_id="root")
    left_node = graph_builder.step(left, node_id="left")
    right_node = graph_builder.step(right, node_id="right")
    graph_builder.add(
        graph_builder.edge_from(graph_builder.start_node).to(root_node),
        graph_builder.edge_from(root_node).to(left_node, right_node),
        graph_builder.edge_from(left_node).to(graph_builder.end_node),
        graph_builder.edge_from(right_node).to(graph_builder.end_node),
    )
    graph = graph_builder.build()
    state = GraphState(query="synthetic", session_id="run:parallel")
    collector = GraphExecutionEvidenceCollector(graph, topology="parallel")
    collector.bind_state(state)

    await run_with_execution_evidence(
        graph,
        state=state,
        deps=_deps(),
        collector=collector,
    )

    fanout = next(
        transition
        for transition in collector.transitions
        if len(transition.scheduled_tasks) == 2
    )
    assert {task.node_id for task in fanout.scheduled_tasks} == {"left", "right"}
    assert len({task.task_id for task in fanout.scheduled_tasks}) == 2


def test_topology_and_version_digests_are_deterministic() -> None:
    first = GraphExecutionEvidenceCollector(_build_fixture_graph(), topology="fixture")
    second = GraphExecutionEvidenceCollector(_build_fixture_graph(), topology="fixture")

    assert first.topology_digest == second.topology_digest
    assert first.version_digest == second.version_digest


@pytest.mark.asyncio
async def test_execute_graph_live_path_surfaces_typed_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.core.registry.service_adapter import ServiceRegistry
    from agent_utilities.orchestration import engine as engine_module

    monkeypatch.setattr(engine_module, "create_model", lambda **_kwargs: None)
    monkeypatch.setattr(
        engine_module,
        "get_discovery_registry",
        lambda: SimpleNamespace(agents=[]),
    )
    monkeypatch.setattr(
        ServiceRegistry,
        "instance",
        classmethod(lambda _cls: SimpleNamespace(initialize=lambda: 0)),
    )

    response = await AgentOrchestrationEngine().execute_graph(
        _build_fixture_graph(),
        {
            "tag_prompts": {},
            "tag_env_vars": {},
            "mcp_toolsets": [],
            "router_model": None,
            "agent_model": None,
        },
        query="synthetic",
        run_id="run:live-evidence",
        streamdown=False,
    )

    evidence = response["execution_evidence"]
    # This minimal graph returns a string rather than the platform's terminal
    # GraphResponse, so the existing guard truthfully classifies it as partial.
    assert response["status"] == "partial"
    assert evidence["node_sequence"] == ["route", "execute", "__end__"]
    assert evidence["resume_supported"] is False


@pytest.mark.asyncio
async def test_dispatcher_records_only_checkpoint_backend_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = SimpleNamespace(save=lambda _state, **_kwargs: "ckpt:run:evidence:123")
    monkeypatch.setattr(
        "agent_utilities.core.checkpoint.manager.CheckpointManager.create",
        lambda **_kwargs: checkpoint,
    )
    state = GraphState(query="synthetic", session_id="run:evidence")
    deps = SimpleNamespace(
        knowledge_engine=object(),
        event_queue=None,
        execution_shape=SimpleNamespace(run_discovery=True),
        plan_sync=None,
    )

    next_node = await _router_impl.dispatcher_step(
        SimpleNamespace(state=state, deps=deps)
    )

    assert next_node == "memory_selection"
    assert state.checkpoint_ids == ["ckpt:run:evidence:123"]
    assert state.checkpoint_ts > 0


def test_kg_checkpoint_without_engine_returns_no_synthetic_identifier() -> None:
    state = GraphState(query="synthetic", session_id="run:evidence")

    assert KGBackend(engine=None).checkpoint(state, session_id=state.session_id) is None


def test_kg_checkpoint_write_failure_returns_no_synthetic_identifier() -> None:
    class _FailedEngine:
        backend_type = "rust"

        def add_node(self, *_args, **_kwargs) -> None:
            raise OSError("fixture write failed")

    state = GraphState(query="synthetic", session_id="run:evidence")

    assert (
        KGBackend(engine=_FailedEngine()).checkpoint(state, session_id=state.session_id)
        is None
    )


def test_kg_checkpoint_backend_failure_is_not_rescued_by_memory_mirror() -> None:
    class _Mirror:
        def __init__(self) -> None:
            self.nodes: dict[str, dict] = {}

        def add_node(self, node_id: str, **properties) -> None:
            self.nodes[node_id] = properties

    class _FailedBackendEngine:
        backend = object()

        def __init__(self) -> None:
            self.graph = _Mirror()

        def _upsert_node(self, *_args, **_kwargs) -> None:
            raise OSError("fixture authority write failed")

    engine = _FailedBackendEngine()
    state = GraphState(query="synthetic", session_id="run:evidence")

    checkpoint_id = KGBackend(engine=engine).checkpoint(
        state, session_id=state.session_id
    )

    assert checkpoint_id is None
    assert engine.graph.nodes == {}


def test_kg_checkpoint_ids_are_unique_after_confirmed_writes() -> None:
    class _PersistedEngine:
        backend_type = "rust"

        def __init__(self) -> None:
            self.nodes: dict[str, dict] = {}

        def add_node(self, node_id: str, *, properties: dict) -> None:
            self.nodes[node_id] = properties

    engine = _PersistedEngine()
    backend = KGBackend(engine=engine)
    state = GraphState(query="synthetic", session_id="run:evidence")

    first = backend.checkpoint(state, session_id=state.session_id)
    second = backend.checkpoint(state, session_id=state.session_id)

    assert first is not None
    assert second is not None
    assert first != second
    assert first.startswith("ckpt:run:evidence:")
    assert second.startswith("ckpt:run:evidence:")
    assert set(engine.nodes) == {first, second}
