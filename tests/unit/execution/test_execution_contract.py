"""Plan 03 Step 5 — unified ExecutionEngine contract tests.

Asserts:
    (a) shared models import from ``core.execution``;
    (b) engine-local / back-compat re-exports point to the SAME class object
        (``is`` identity) — proving a single source of truth;
    (c) each engine is recognised as an ``ExecutionEngine`` (structural /
        runtime-checkable Protocol check);
    (d) constructing/using the canonical models works.
"""

from __future__ import annotations

import inspect


# ── (a) shared models import from core.execution ────────────────────


def test_shared_models_import_from_core_execution():
    from agent_utilities.core.execution import (
        ExecutionEngine,
        ExecutionManifest,
        ExecutionResult,
        ExecutionStep,
    )

    assert ExecutionEngine is not None
    assert ExecutionManifest is not None
    assert ExecutionResult is not None
    assert ExecutionStep is not None


# ── (b) single-source identity: re-exports are the SAME object ──────


def test_execution_manifest_single_source_identity():
    from agent_utilities.core.execution import ExecutionManifest as CoreManifest
    from agent_utilities.models.execution_manifest import (
        ExecutionManifest as CanonicalManifest,
    )

    assert CoreManifest is CanonicalManifest


def test_execution_result_single_source_identity():
    from agent_utilities.core.execution import ExecutionResult as CoreResult
    from agent_utilities.models.execution_manifest import (
        ExecutionResult as CanonicalResult,
    )

    assert CoreResult is CanonicalResult


def test_execution_step_single_source_identity():
    from agent_utilities.core.execution import ExecutionStep as CoreStep
    from agent_utilities.models import ExecutionStep as ModelsStep
    from agent_utilities.models.graph import ExecutionStep as GraphStep

    # All import paths resolve to the one canonical class object.
    assert CoreStep is ModelsStep
    assert CoreStep is GraphStep


def test_orchestration_engine_uses_canonical_models():
    """The orchestration engine imports the very same model objects."""
    from agent_utilities.core.execution import ExecutionManifest, ExecutionResult
    from agent_utilities.orchestration import engine as orch_engine

    assert orch_engine.ExecutionManifest is ExecutionManifest
    assert orch_engine.ExecutionResult is ExecutionResult


# ── (c) each engine conforms to the ExecutionEngine Protocol ────────


def _has_async_run(cls) -> bool:
    run = getattr(cls, "run", None)
    return run is not None and inspect.iscoroutinefunction(run)


def test_orchestration_engine_conforms():
    from agent_utilities.core.execution import ExecutionEngine
    from agent_utilities.orchestration.engine import AgentOrchestrationEngine

    # Structural: async run(manifest) exists.
    assert _has_async_run(AgentOrchestrationEngine)
    # runtime_checkable Protocol recognises the class structurally.
    assert issubclass(AgentOrchestrationEngine, ExecutionEngine)


def test_knowledge_graph_engine_conforms():
    from agent_utilities.core.execution import ExecutionEngine
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    assert _has_async_run(IntelligenceGraphEngine)
    assert issubclass(IntelligenceGraphEngine, ExecutionEngine)


def test_graph_executor_engine_conforms():
    from agent_utilities.core.execution import ExecutionEngine
    from agent_utilities.graph.executor import GraphExecutorEngine

    assert _has_async_run(GraphExecutorEngine)
    assert issubclass(GraphExecutorEngine, ExecutionEngine)


def test_unified_reference_engine_conforms():
    from agent_utilities.core.execution import ExecutionEngine, UnifiedExecutionEngine

    engine = UnifiedExecutionEngine()
    assert _has_async_run(UnifiedExecutionEngine)
    assert isinstance(engine, ExecutionEngine)


# ── (d) constructing / using the canonical models works ─────────────


def test_construct_execution_step():
    from agent_utilities.core.execution import ExecutionStep

    step = ExecutionStep(id="step-1", description="do a thing")
    # ``id``/``node_id`` aliasing is preserved by the canonical Task model.
    assert step.id == "step-1"
    assert step.node_id == "step-1"


def test_construct_execution_manifest_and_result():
    from agent_utilities.core.execution import (
        ExecutionManifest,
        ExecutionResult,
    )
    from agent_utilities.models.execution_manifest import AgentSpec

    manifest = ExecutionManifest(
        name="unit-test",
        agents=[AgentSpec(agent_id="a1", task_template="hello")],
        query="hello",
    )
    assert manifest.agent_count == 1
    assert manifest.is_trivial is True

    result = ExecutionResult(
        manifest_id=manifest.manifest_id,
        agent_count=manifest.agent_count,
        success=True,
    )
    assert result.success is True
    assert result.manifest_id == manifest.manifest_id


async def test_unified_engine_run_returns_result():
    from agent_utilities.core.execution import (
        ExecutionManifest,
        ExecutionResult,
        UnifiedExecutionEngine,
    )
    from agent_utilities.models.execution_manifest import AgentSpec

    manifest = ExecutionManifest(agents=[AgentSpec(agent_id="a1", task_template="hi")])
    result = await UnifiedExecutionEngine().run(manifest)
    assert isinstance(result, ExecutionResult)
    assert result.success is True
    assert result.manifest_id == manifest.manifest_id
