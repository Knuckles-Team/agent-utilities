"""BUG-014 — an invalid workflow DAG (cycle / dangling dependency) must never
execute the remaining steps as a "run them all as one wave" fallback.

CONCEPT:AU-ORCH.execution.workflow-dag-validation

``WorkflowRunner._execute_plan_via_agents``'s wave loop computes ``ready`` as
the steps whose ``depends_on`` are all satisfied. When a stored
``WorkflowDefinition`` carries a dependency cycle, or a step whose
``depends_on`` names an id that is not any step in the plan (a dangling
dependency), no step is ever "ready" and the loop falls into:

    if not ready:
        # A dependency cycle / dangling dep -- run the rest as one wave rather
        # than deadlock (the SHACL gate upstream guards malformed DAGs).
        ready = list(remaining)

That comment's premise is false. Two independent things make it false:

1. The execution-time shape gate (``workflow_gate.gate_workflow_execution``,
   ``agent_utilities/knowledge_graph/core/workflow_gate.py``) validates
   ``WorkflowDefinitionShape``/``WorkflowStepShape`` against
   ``governance.shapes.ttl`` -- and those shapes only check
   ``name``/``step_count``/``node_id`` presence. Neither shape declares any
   SPARQL constraint over ``depends_on`` edges, so a cycle or a dangling
   dependency passes the shape gate cleanly. (The cycle/dangling-dependency
   detector this codebase already has --
   ``agent_utilities.models.sdd.Tasks.detect_cycles``/``validate_dependencies``
   -- is never called from this path; ``git grep`` finds it referenced only
   from its own module and ``agent_utilities/sdd/__init__.py``.)

2. Even where the shape gate is consulted, ``WorkflowHasNoStepsError``'s own
   docstring (this same module, ``runner.py``) records that it "only guards
   the MCP boundary": ``ticket_playbooks._dispatch_workflow``,
   ``loop_controller._default_skill_runner``,
   ``weights_distillation._dispatch_train_workflow``, and
   ``schedule_engine``'s workflow/agent dispatch all call
   ``WorkflowRunner.execute_by_name``/``Orchestrator.execute_workflow``
   directly and never pass through it.

So an invalid DAG reaching ``execute_by_name`` today does not deadlock and
does not refuse -- it silently dispatches every remaining step through
``run_agent`` in one wave, and (if every step's task happens to complete) can
report ``status="completed"`` for a workflow whose declared dependency graph
was never honored.

These fixtures pin the fail-closed invariant a future validation pass must
satisfy: an invalid DAG must dispatch ZERO steps. They are written directly
against ``WorkflowRunner`` (bypassing the MCP-boundary-only shape gate, the
same way the four bypass callers above do) so they exercise the real
regression, not the one guarded surface. Per the ledger's Next-owned-action
for this bug ("Add fail-closed validation fixtures before any runner
refactor"), no runner change ships in this pass -- these fixtures are
expected to FAIL against current ``runner.py`` and are the deliverable.

@pytest.mark.concept("AU-ORCH.execution.workflow-dag-validation")
"""

from __future__ import annotations

import pytest

from agent_utilities.models.graph import GraphPlan
from agent_utilities.models.sdd import Task
from agent_utilities.workflows.runner import WorkflowRunner

pytestmark = pytest.mark.concept("AU-ORCH.execution.workflow-dag-validation")


class FakeEngine:
    """Minimal engine stand-in, matching test_workflow_runner_silent_success.py."""

    def __init__(self):
        self.graph = None
        self.backend = None


def _install_fake_run_agent(monkeypatch) -> list[str]:
    """Replace ``run_agent`` and return the list it appends dispatched step
    (agent) names to. An empty list after execution is the fail-closed proof:
    no step of an invalid DAG was ever dispatched."""
    import agent_utilities.orchestration.agent_runner as agent_runner_module

    dispatched: list[str] = []

    async def _fake_run_agent(*, agent_name, task, engine, max_steps, context, session_id, reasoning_effort):
        dispatched.append(agent_name)
        return f"ok:{agent_name}"

    monkeypatch.setattr(agent_runner_module, "run_agent", _fake_run_agent)
    return dispatched


class TestCyclicWorkflowDagNeverExecutesAsFallbackWave:
    async def test_two_step_cycle_dispatches_no_steps(self, monkeypatch):
        """A -> depends on B, B -> depends on A: no step can ever be 'ready'.

        Reproduces the live defect: today this does not refuse -- the
        no-ready-step fallback treats both steps as one wave and dispatches
        both via run_agent.
        """
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

        steps = [
            Task(id="step-a", depends_on=["step-b"]),
            Task(id="step-b", depends_on=["step-a"]),
        ]
        monkeypatch.setattr(
            WorkflowStore,
            "load_workflow",
            lambda self, name: GraphPlan(steps=steps, metadata={}),
        )
        dispatched = _install_fake_run_agent(monkeypatch)

        engine = FakeEngine()
        runner = WorkflowRunner()

        result = None
        raised: Exception | None = None
        try:
            result = await runner.execute_by_name("cyclic_workflow", engine)
        except Exception as exc:  # noqa: BLE001 -- any refusal is acceptable, only the dispatch invariant is asserted below
            raised = exc

        assert dispatched == [], (
            "a two-step dependency cycle must dispatch ZERO steps via "
            f"run_agent (fail closed); actually dispatched {dispatched!r} "
            "-- this is BUG-014's 'run the rest as one wave' fallback firing "
            "on a cyclic DAG"
        )
        # If it did not raise, it must not have reported a plain success either
        # -- a cycle silently "completing" would be its own separate defect.
        if raised is None:
            assert result is not None
            assert result.status != "completed", (
                f"a cyclic workflow DAG reported status={result.status!r}; "
                "an unvalidated cycle must never resolve to a success status"
            )

    async def test_three_step_cycle_with_a_clean_entrypoint_dispatches_no_steps(
        self, monkeypatch
    ):
        """entry -> ok (no deps); a -> b -> c -> a (a 3-cycle unreachable from
        entry). The presence of one legitimately-ready step must not let the
        fallback wave smuggle the cyclic steps through in the same pass --
        the ENTIRE definition is invalid, not just the cyclic subset."""
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

        steps = [
            Task(id="entry", depends_on=[]),
            Task(id="a", depends_on=["c"]),
            Task(id="b", depends_on=["a"]),
            Task(id="c", depends_on=["b"]),
        ]
        monkeypatch.setattr(
            WorkflowStore,
            "load_workflow",
            lambda self, name: GraphPlan(steps=steps, metadata={}),
        )
        dispatched = _install_fake_run_agent(monkeypatch)

        engine = FakeEngine()
        runner = WorkflowRunner()

        try:
            await runner.execute_by_name("partially_cyclic_workflow", engine)
        except Exception:  # noqa: BLE001 -- only the dispatch invariant matters here
            pass

        assert dispatched == [], (
            "a workflow DAG containing ANY cycle must dispatch ZERO steps, "
            f"including its legitimately-ready steps; actually dispatched "
            f"{dispatched!r}"
        )


class TestDanglingDependencyWorkflowDagNeverExecutesAsFallbackWave:
    async def test_dependency_on_unknown_step_id_dispatches_no_steps(
        self, monkeypatch
    ):
        """``step-a`` depends on ``step-ghost``, which is not any step in the
        plan. No step can ever become 'ready' -- reproduces the same
        no-ready-step fallback via a dangling reference instead of a cycle.
        """
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

        steps = [Task(id="step-a", depends_on=["step-ghost"])]
        monkeypatch.setattr(
            WorkflowStore,
            "load_workflow",
            lambda self, name: GraphPlan(steps=steps, metadata={}),
        )
        dispatched = _install_fake_run_agent(monkeypatch)

        engine = FakeEngine()
        runner = WorkflowRunner()

        try:
            await runner.execute_by_name("dangling_dep_workflow", engine)
        except Exception:  # noqa: BLE001 -- only the dispatch invariant matters here
            pass

        assert dispatched == [], (
            "a step depending on an unknown step id must dispatch ZERO "
            f"steps (fail closed); actually dispatched {dispatched!r}"
        )

    async def test_self_dependency_dispatches_no_steps(self, monkeypatch):
        """``step-a`` depends on itself -- the degenerate one-node cycle."""
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

        steps = [Task(id="step-a", depends_on=["step-a"])]
        monkeypatch.setattr(
            WorkflowStore,
            "load_workflow",
            lambda self, name: GraphPlan(steps=steps, metadata={}),
        )
        dispatched = _install_fake_run_agent(monkeypatch)

        engine = FakeEngine()
        runner = WorkflowRunner()

        try:
            await runner.execute_by_name("self_dependency_workflow", engine)
        except Exception:  # noqa: BLE001 -- only the dispatch invariant matters here
            pass

        assert dispatched == [], (
            "a self-dependent step must dispatch ZERO steps (fail closed); "
            f"actually dispatched {dispatched!r}"
        )
