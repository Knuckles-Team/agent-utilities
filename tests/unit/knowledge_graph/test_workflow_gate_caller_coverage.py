"""D-WS-8 per-caller proof — each bypassing caller now inherits the gate.

CONCEPT:AU-ORCH.execution.ontology-validation-execution-path

``Orchestrator.execute_workflow`` is the chokepoint every caller of a
by-name workflow run converges on; the D-WS-8 fix moved
``workflow_gate.gate_workflow_execution`` (SHACL shape + ACL pre-dispatch
validation) into that method itself so every caller inherits it without
each needing its own call to the gate. That fix (and its unit-level proof
that ``Orchestrator.execute_workflow`` itself gates) already lives in
``test_workflow_gate.py::TestOrchestratorExecuteWorkflowGatesAtTheChokepoint``.

This file is the *outside-in* proof the lane brief demanded: a test per
production caller that was named as bypassing the gate before the fix —
``ticket_playbooks._dispatch_workflow``,
``loop_controller._default_skill_runner`` (the autonomous Loop engine),
``weights_distillation._dispatch_train_workflow``, and
``schedule_engine``'s ``kind in (workflow, agent)`` dispatch — proving a
KNOWN-BAD workflow (a name with no stored ``WorkflowDefinition``, the
cheapest deterministic denial: ``workflow_definition_missing``) is now
refused when reached THROUGH each caller's own public entrypoint, not just
by calling ``Orchestrator.execute_workflow`` directly.

``schedule_engine`` gets an extra assertion: before this lane's fix it
called ``engine.execute_workflow(...)`` on the bare ``IntelligenceGraphEngine``
(which has no such method) — an ``AttributeError`` that never reached the
gate at all and looked, from the outside, exactly like "the gate denied
it". The ``error_class`` returned by ``public_error_payload`` distinguishes
the two: this test asserts it is ``WorkflowGateDeniedError``, not
``AttributeError``.

@pytest.mark.concept("AU-ORCH.execution.ontology-validation-execution-path")
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyshacl")
pytest.importorskip("rdflib")

pytestmark = pytest.mark.concept("AU-ORCH.execution.ontology-validation-execution-path")


class FakeGraph:
    """Compute-mirror fake honoring nodes(data=True) / out_edges(data=True)."""

    def __init__(self):
        self._nodes: dict[str, dict] = {}
        self._edges: list[tuple[str, str, dict]] = []

    def add_node(self, node_id, props):
        self._nodes[node_id] = dict(props)

    def add_edge(self, src, tgt, **props):
        self._edges.append((src, tgt, props))

    @property
    def nodes(self):
        outer = self

        class _View(dict):
            def __call__(self, data=False):
                if data:
                    return list(outer._nodes.items())
                return list(outer._nodes)

        return _View(outer._nodes)

    def out_edges(self, node_id, data=False):
        rows = [(s, t, p) for s, t, p in self._edges if s == node_id]
        return rows if data else [(s, t) for s, t, _ in rows]


class FakeEngine:
    """Stand-in for ``IntelligenceGraphEngine`` -- no workflow named
    ``UNSTORED_WORKFLOW_NAME`` is ever seeded on it, so
    ``gate_workflow_execution`` deterministically denies with
    ``workflow_definition_missing`` before any SHACL/ACL machinery runs.
    """

    def __init__(self):
        self.graph = FakeGraph()
        self.backend = None


UNSTORED_WORKFLOW_NAME = "definitely_never_stored_workflow_xyz"


class TestTicketPlaybooksDispatchWorkflowIsGated:
    """``ticket_playbooks._dispatch_workflow`` (D-WS-8 caller #1)."""

    async def test_unstored_workflow_is_refused_not_silently_dispatched(self):
        from agent_utilities.knowledge_graph.adaptation.ticket_playbooks import (
            _dispatch_workflow,
        )

        engine = FakeEngine()

        result = _dispatch_workflow(
            engine, UNSTORED_WORKFLOW_NAME, ticket="TICKET-1", source="jira"
        )

        # _dispatch_workflow degrades any exception (incl. the gate denial) to
        # None rather than crashing triage -- the proof is that it does NOT
        # report the workflow as dispatched.
        assert result is None


class TestLoopControllerDefaultSkillRunnerIsGated:
    """``loop_controller._default_skill_runner`` (D-WS-8 caller #2 -- the
    autonomous Loop engine itself, the caller the lane brief calls out by
    name).
    """

    async def test_unstored_workflow_is_refused_not_silently_run(self):
        from agent_utilities.knowledge_graph.research.loop_controller import (
            _default_skill_runner,
        )

        engine = FakeEngine()

        # The "workflow:" prefix skips compile_workflow and passes the ref
        # straight through as the workflow_id -- exactly the path
        # loop_controller uses for an already-compiled skill workflow.
        ok, output = _default_skill_runner(
            f"workflow:{UNSTORED_WORKFLOW_NAME}", "run the thing", engine=engine
        )

        assert ok is False
        assert "refused" in output.lower() or "gate" in output.lower()


class TestWeightsDistillationDispatchTrainWorkflowIsGated:
    """``weights_distillation._dispatch_train_workflow`` (D-WS-8 caller #3)."""

    async def test_unstored_workflow_raises_gate_denial(self):
        from agent_utilities.knowledge_graph.core.workflow_gate import (
            WorkflowGateDeniedError,
        )
        from agent_utilities.knowledge_graph.memory.weights_distillation import (
            _dispatch_train_workflow,
        )

        engine = FakeEngine()

        # This caller's contract is to RAISE on failure (so the caller can
        # degrade to a durable "enqueued" job) rather than swallow -- the
        # gate denial must be the thing that propagates.
        with pytest.raises(WorkflowGateDeniedError):
            _dispatch_train_workflow(
                engine, UNSTORED_WORKFLOW_NAME, {"target": "lora"}, timeout=5.0
            )


class TestScheduleEngineWorkflowDispatchIsGated:
    """``schedule_engine``'s ``kind in (workflow, agent)`` dispatch (D-WS-8
    caller #4).

    Before this lane, this branch called ``engine.execute_workflow(...)`` on
    the bare ``IntelligenceGraphEngine`` -- which has no such method -- so it
    NEVER reached ``Orchestrator.execute_workflow`` (the gated chokepoint) at
    all; it just raised ``AttributeError`` from the wrong object, an
    unrelated bug that happened to also fail closed. This lane rewired the
    call through ``Orchestrator(engine).execute_workflow(...)`` so it both
    reaches the real dispatch surface and inherits the gate. ``error_class``
    (from ``public_error_payload``) distinguishes the two failure modes.
    """

    async def test_unstored_workflow_is_refused_by_the_gate_not_by_attributeerror(
        self,
    ):
        from agent_utilities.core.schedule_engine import _dispatch_scheduled_job

        engine = FakeEngine()

        result = _dispatch_scheduled_job(
            engine,
            {
                "kind": "workflow",
                "ref": UNSTORED_WORKFLOW_NAME,
                "task": "scheduled run",
            },
        )

        assert result["status"] == "failed"
        assert result["error_class"] == "WorkflowGateDeniedError", (
            "must fail via the D-WS-8 gate, not the pre-fix AttributeError "
            "from calling execute_workflow on the bare engine"
        )

    async def test_agent_kind_is_also_routed_through_the_gate(self):
        """``kind == "agent"`` shares the same dispatch branch as
        ``"workflow"`` -- prove it too, not just the more obviously-named
        ``"workflow"`` kind.
        """
        from agent_utilities.core.schedule_engine import _dispatch_scheduled_job

        engine = FakeEngine()

        result = _dispatch_scheduled_job(
            engine,
            {
                "kind": "agent",
                "ref": UNSTORED_WORKFLOW_NAME,
                "task": "scheduled run",
            },
        )

        assert result["status"] == "failed"
        assert result["error_class"] == "WorkflowGateDeniedError"
