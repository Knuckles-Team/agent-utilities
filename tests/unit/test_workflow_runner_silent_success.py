"""D-FSR-1 — a zero-step ``WorkflowDefinition`` must never silently succeed.

CONCEPT:AU-ORCH.execution.no-silent-success-on-empty-workflow

``graph_workflows action=execute`` resolves a stored ``WorkflowDefinition`` by
name and runs its step-DAG via :class:`WorkflowRunner`. 415 of 625 stored
``WorkflowDefinition`` nodes in the live KG carry ``step_count=0`` — an
ingester bug (fixed on ``main``) unconditionally minted a ``WorkflowDefinition``
for every ``SKILL.md`` regardless of its declared ``skill_type``. Before this
fix, executing one of these zero-step definitions ran the empty wave loop in
:meth:`WorkflowRunner._execute_plan_via_agents`, produced zero
``StepResult``s, and computed ``status = "completed" if n_failed == 0 else
...`` — trivially true for an empty list — so the caller received a
``status="completed"`` response indistinguishable from a workflow that
actually ran and succeeded.

This does not apply only to the ``graph_workflows`` MCP tool: several other
callers (``ticket_playbooks._dispatch_workflow``,
``loop_controller._default_skill_runner``,
``weights_distillation._dispatch_train_workflow``, ``schedule_engine``) call
``Orchestrator.execute_workflow`` / ``WorkflowRunner.execute_by_name``
directly and never pass through the ``graph_workflows`` SHACL/ACL gate
(``workflow_gate.gate_workflow_execution``), so the fix belongs in the
runtime engine itself, not only at the MCP tool boundary.

A workflow with a legitimately empty step-DAG may still be *defined* (that is
not this bug) — only *executing* one must never report success.

@pytest.mark.concept("AU-ORCH.execution.no-silent-success-on-empty-workflow")
"""

from __future__ import annotations

import pytest

from agent_utilities.models.graph import GraphPlan
from agent_utilities.workflows.runner import WorkflowHasNoStepsError, WorkflowRunner

pytestmark = pytest.mark.concept(
    "AU-ORCH.execution.no-silent-success-on-empty-workflow"
)


class FakeEngine:
    """Minimal engine stand-in — never reached once the zero-step guard fires."""

    def __init__(self):
        self.graph = None
        self.backend = None


class TestZeroStepExecutionNeverSilentlySucceeds:
    async def test_execute_by_name_does_not_report_completed_for_zero_steps(
        self, monkeypatch
    ):
        """Reproduces the live bug: a step_count=0 WorkflowDefinition executed
        via ``execute_by_name`` (the ``graph_workflows action=execute`` path)
        must not return a ``WorkflowResult`` claiming ``status="completed"``
        having run zero steps.
        """
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

        monkeypatch.setattr(
            WorkflowStore,
            "load_workflow",
            lambda self, name: GraphPlan(steps=[], metadata={}),
        )

        engine = FakeEngine()
        runner = WorkflowRunner()

        with pytest.raises(WorkflowHasNoStepsError) as excinfo:
            await runner.execute_by_name("stale_skill_workflow", engine)

        # Must name the workflow and explain there was nothing to execute —
        # a bare/opaque exception would be its own silent-failure regression.
        message = str(excinfo.value).lower()
        assert "stale_skill_workflow" in message
        assert "step" in message

    async def test_error_hints_at_sibling_callable_resource_skill(self, monkeypatch):
        """When a repaired CallableResource(AGENT_SKILL) shares the stale
        workflow's name (the 267-node repair pass, D-FSR-1 context), the
        error should point the caller at it instead of just dead-ending.
        """
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

        monkeypatch.setattr(
            WorkflowStore,
            "load_workflow",
            lambda self, name: GraphPlan(steps=[], metadata={}),
        )

        class FakeBackend:
            def execute(self, query, params):
                assert params["name"] == "repaired_skill"
                return [{"rid": "skill:repaired_skill:abc"}]

        engine = FakeEngine()
        engine.backend = FakeBackend()
        runner = WorkflowRunner()

        with pytest.raises(WorkflowHasNoStepsError) as excinfo:
            await runner.execute_by_name("repaired_skill", engine)

        message = str(excinfo.value)
        assert "CallableResource" in message
        assert "graph_orchestrate" in message

    async def test_execute_parallel_engine_bridge_does_not_report_completed_for_zero_steps(
        self, monkeypatch
    ):
        """D-WS-7 — the OTHER execution chokepoint (``WorkflowRunner.execute()``,
        the ``ParallelEngine`` bridge) shared the same trivial-success-on-zero-
        steps shape as the ``execute_by_name`` path this module already fixed
        (D-FSR-1): an empty ``GraphPlan`` produces zero waves, and
        ``ParallelEngine``/``ExecutionManifest`` vacuously report nothing
        failed, so ``execute()`` would return ``status="completed"`` having run
        zero steps. Proves the real entrypoint — no mocked
        ``execute_via_parallel_engine`` here, the guard must fire BEFORE it is
        ever called.
        """
        called = False

        async def _fail_if_called(self_, plan, engine, workflow_name="", query=""):
            nonlocal called
            called = True
            raise AssertionError("execute_via_parallel_engine must not be reached")

        monkeypatch.setattr(
            WorkflowRunner, "execute_via_parallel_engine", _fail_if_called
        )

        engine = FakeEngine()
        runner = WorkflowRunner()
        plan = GraphPlan(steps=[], metadata={})

        with pytest.raises(WorkflowHasNoStepsError) as excinfo:
            await runner.execute(plan, engine, workflow_name="empty_bridge_flow")

        assert not called
        message = str(excinfo.value).lower()
        assert "empty_bridge_flow" in message
        assert "step" in message

    async def test_workflow_not_found_still_raises_as_before(self, monkeypatch):
        """Sanity check the existing not-found contract is untouched."""
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

        monkeypatch.setattr(WorkflowStore, "load_workflow", lambda self, name: None)

        engine = FakeEngine()
        runner = WorkflowRunner()

        with pytest.raises(ValueError, match="not found"):
            await runner.execute_by_name("nonexistent_workflow", engine)
