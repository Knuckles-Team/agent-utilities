"""BUG-015 (GOC-20) — a successful run can lack result or provenance.

Fault-injection at every provenance write boundary a ``run_agent``/WorkItem
completion touches, proving the specific defect the lane names: a run (or a
WorkItem) can be REPORTED as successful while its RunTrace/Outcome/ToolCall
provenance write silently failed. See
``plans/graph-os-completion-program/decisions/GOC-20-atomic-outcome-provenance.md``
for the fault-injection matrix these tests back and the chosen remediation.

Three ``agent_runner.py`` boundaries are fixed here (this lane owns that
file): the ``enterprise`` early-exit, the ``ServiceRegistry`` early-exit, and
the main dispatch path's run_summary/synthesis/done reporting.

Boundaries 4/5 (``agent_dispatch_worker.py`` WorkItem provenance) were
originally fault-injected-and-documented-only, deferred pending GOC-18's file
ownership per the program's then-current Phase-0 hard rule. GOC-18
(``merge(goc-18): no broker ack before durable terminal state``, BUG-002/003)
has since landed and its own lane doc states "GOC-20 owns terminal outcome" —
the deferral's premise (a live, in-flight GOC-18 branch on this same file) no
longer holds, so GOC-20 (this lane) applies the remediation specified in the
decision record directly. The two tests below are UN-xfailed now that
``_finalize_work_item``/``_write_work_item_provenance`` route the
OutcomeEvaluation and Observation/Claim/Action/Trace writes through the
engine's atomic ``batch_typed_mutations`` and surface a caller-visible
``"degraded"``/``"failed"``/``"unavailable"`` signal instead of silently
reporting ``"committed"``/``None``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_utilities.orchestration import agent_dispatch_worker as worker
from agent_utilities.orchestration import agent_runner

# ``asyncio_mode = "auto"`` (pyproject.toml/pytest.ini) discovers async tests
# without a marker; no blanket ``pytestmark`` here, since two tests below are
# deliberately synchronous (the agent_dispatch_worker fault injections).


# --------------------------------------------------------------------------- #
# Boundary 1 — agent_runner.py "enterprise" early exit (FIXED in this lane)
# --------------------------------------------------------------------------- #


async def test_enterprise_path_surfaces_provenance_write_failure() -> None:
    """A successful enterprise orchestration whose RunTrace write fails must
    not be reported as fully provenanced. Before the BUG-015 fix, this early
    exit discarded ``_record_execution_trace_ordered``'s return value
    entirely — the caller had ZERO signal the write failed."""

    class _FakePE:
        async def execute(self, _manifest):
            return {"answer": "the enterprise result"}

    with (
        patch.object(
            agent_runner,
            "_get_or_create_engine",
            return_value=MagicMock(),
        ),
        patch(
            "agent_utilities.graph.manifest_generators.manifest_for_enterprise",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "agent_utilities.graph.parallel_engine.ParallelEngine",
            return_value=_FakePE(),
        ),
        patch.object(
            agent_runner,
            "_record_execution_trace_ordered",
            new=AsyncMock(return_value=False),  # <-- fault injection: write failed
        ),
    ):
        out = await agent_runner.run_agent(
            agent_name="enterprise", task="run the company", return_mermaid=True
        )

    payload = json.loads(out)
    # The defect: pre-fix, this key did not exist at all and the caller could
    # not distinguish this run from a fully-provenanced one.
    assert payload["provenance_recorded"] is False


# --------------------------------------------------------------------------- #
# Boundary 2 — agent_runner.py ServiceRegistry early exit (FIXED in this lane)
# --------------------------------------------------------------------------- #


async def test_service_registry_path_surfaces_provenance_write_failure() -> None:
    class _Capability:
        def run(self, task: str) -> str:
            return f"handled: {task}"

    class _Service:
        @staticmethod
        def get_class():
            return _Capability

    class _Registry:
        @staticmethod
        def get(_name: str):
            return _Service()

    with (
        patch(
            "agent_utilities.core.registry.service_adapter.ServiceRegistry.instance",
            return_value=_Registry(),
        ),
        patch.object(
            agent_runner,
            "_record_execution_trace_ordered",
            new=AsyncMock(return_value=False),  # <-- fault injection: write failed
        ),
    ):
        out = await agent_runner.run_agent(
            "native-capability",
            "fixture task",
            engine=object(),
            return_mermaid=True,
        )

    payload = json.loads(out)
    assert payload["output"] == "handled: fixture task"
    assert payload["provenance_recorded"] is False


# --------------------------------------------------------------------------- #
# Boundary 3 — agent_runner.py main dispatch path (FIXED in this lane)
# --------------------------------------------------------------------------- #


async def test_main_path_run_summary_reports_degraded_when_trace_write_fails() -> None:
    """A GENUINELY successful delegation (real, non-sentinel output) whose
    durable RunTrace write fails must not report ``run_summary.outcome ==
    "ok"``. This is the exact BUG-015 scenario: content succeeded, durable
    provenance did not, and the two must not be conflated into "ok"."""
    fake_engine = MagicMock()
    fake_engine.backend = None

    real_result = {"results": {"output": "Found 3 running containers: web, db, cache."}}

    with (
        patch.object(agent_runner, "_get_or_create_engine", return_value=fake_engine),
        patch.object(
            agent_runner, "_resolve_agent_from_kg", return_value={"type": "unknown"}
        ),
        patch.object(
            agent_runner,
            "_build_execution_config",
            return_value={"mcp_toolsets": []},
        ),
        patch.object(
            agent_runner,
            "_execute_graph",
            new=AsyncMock(return_value=real_result),
        ),
        patch.object(
            agent_runner,
            "_record_execution_trace_ordered",
            new=AsyncMock(return_value=False),  # <-- fault injection: write failed
        ),
        patch.object(agent_runner, "_write_step_credit"),
    ):
        out = await agent_runner.run_agent(
            agent_name="some-agent", task="list containers", include_run_summary=True
        )

    payload = json.loads(out)
    assert payload["output"] == "Found 3 running containers: web, db, cache."
    assert payload["provenance_recorded"] is False
    run_summary = payload["run_summary"]
    assert run_summary["provenance_recorded"] is False
    # The defect: pre-fix, a content-successful run always reported "ok" here
    # regardless of whether its provenance write actually committed.
    assert run_summary["outcome"] == "degraded"
    assert "failure" in run_summary
    assert "provenance" in run_summary["failure"]["raw"].lower()


async def test_main_path_done_event_is_consistent_with_checkpoint_event() -> None:
    """The "checkpoint" progress event was already gated on the trace write
    (D-DST-6); "synthesis"/"done" were not. A caller reading only the FINAL
    ("done") event of the stream — the one most systems act on — must not see
    a different, more optimistic verdict than "checkpoint" already gave for
    the exact same run."""
    fake_engine = MagicMock()
    fake_engine.backend = None
    real_result = {"results": {"output": "Found 3 running containers."}}
    events: list[dict] = []

    async def _sink(event) -> None:
        events.append(
            {
                "stage": getattr(event, "stage", None),
                "status": getattr(event, "status", None),
            }
        )

    with (
        patch.object(agent_runner, "_get_or_create_engine", return_value=fake_engine),
        patch.object(
            agent_runner, "_resolve_agent_from_kg", return_value={"type": "unknown"}
        ),
        patch.object(
            agent_runner,
            "_build_execution_config",
            return_value={"mcp_toolsets": []},
        ),
        patch.object(
            agent_runner,
            "_execute_graph",
            new=AsyncMock(return_value=real_result),
        ),
        patch.object(
            agent_runner,
            "_record_execution_trace_ordered",
            new=AsyncMock(return_value=False),  # <-- fault injection: write failed
        ),
        patch.object(agent_runner, "_write_step_credit"),
    ):
        await agent_runner.run_agent(
            agent_name="some-agent",
            task="list containers",
            progress_sink=_sink,
        )

    by_stage = {e["stage"]: e["status"] for e in events if e["stage"]}
    assert "checkpoint" in by_stage and "done" in by_stage
    # Pre-fix: checkpoint == "degraded" but done == "ok" for the SAME run —
    # an internally inconsistent stream. Both must agree now.
    assert by_stage["checkpoint"] == by_stage["done"] == "degraded"
    assert by_stage.get("synthesis") == "degraded"


# --------------------------------------------------------------------------- #
# Boundary 4/5 — agent_dispatch_worker.py WorkItem provenance (FIXED in this
# lane — see the module docstring for why the GOC-18 deferral no longer
# applies).
# --------------------------------------------------------------------------- #


def _policy_decision() -> SimpleNamespace:
    return SimpleNamespace(
        kind="work_item.execute",
        target="workitem:fixture-1",
        decision="allow",
        reason="fixture",
        allowed=True,
    )


class _RaisingOutcomeEngine:
    """``batch_typed_mutations`` fails for the OutcomeEvaluation append
    (boundary 4) — the real write path since this lane's fix. ``add_node`` is
    also defined (and would likewise raise for ``OutcomeEvaluation``) so a
    regression back to the pre-fix per-node write path fails loudly too,
    instead of silently passing through an unexercised method."""

    def __init__(self) -> None:
        self.nodes: list[tuple[str, str]] = []

    def add_node(self, node_id, node_type, properties=None):
        if node_type == "OutcomeEvaluation":
            raise RuntimeError("engine unavailable: OutcomeEvaluation write failed")
        self.nodes.append((node_id, node_type))

    def batch_typed_mutations(self, mutations, *, upsert: bool = True) -> bool:
        if any(m.get("node_type") == "OutcomeEvaluation" for m in mutations):
            raise RuntimeError("engine unavailable: OutcomeEvaluation batch failed")
        self.nodes.extend((m["id"], m["node_type"]) for m in mutations)
        return True


def test_finalize_work_item_surfaces_outcome_evaluation_write_failure(monkeypatch):
    engine = _RaisingOutcomeEngine()
    monkeypatch.setattr(
        "agent_utilities.orchestration.work_item.commit_execution_work_item",
        lambda *_a, **_k: "committed",
    )

    result = worker._finalize_work_item(
        engine,
        "workitem:fixture-1",
        {"work_item_id": "workitem:fixture-1", "lease_id": "lease-1"},
        status="completed",
        reward=1.0,
        feedback_text="ok",
    )

    # A WorkItem cannot report "committed" while its required
    # OutcomeEvaluation node is absent (BUG-015/GOC-20 B7).
    assert result != "committed"
    assert result == "degraded"


def test_finalize_work_item_reports_committed_when_outcome_evaluation_lands(
    monkeypatch,
):
    """Known-good counterpart: a batch that actually SUCCEEDS still reports
    "committed" — proves the "degraded" result above is caused by the
    injected batch failure, not by some unconditional downgrade."""

    class _SucceedingOutcomeEngine:
        def __init__(self) -> None:
            self.nodes: list[tuple[str, str]] = []

        def batch_typed_mutations(self, mutations, *, upsert: bool = True) -> bool:
            self.nodes.extend((m["id"], m["node_type"]) for m in mutations)
            return True

    engine = _SucceedingOutcomeEngine()
    monkeypatch.setattr(
        "agent_utilities.orchestration.work_item.commit_execution_work_item",
        lambda *_a, **_k: "committed",
    )

    result = worker._finalize_work_item(
        engine,
        "workitem:fixture-1",
        {"work_item_id": "workitem:fixture-1", "lease_id": "lease-1"},
        status="completed",
        reward=1.0,
        feedback_text="ok",
    )
    assert result == "committed"
    assert any(node_type == "OutcomeEvaluation" for _id, node_type in engine.nodes)


class _RaisingProvenanceEngine:
    """``batch_typed_mutations`` fails for the provenance batch (boundary 5)
    — the real write path since this lane's fix."""

    def batch_typed_mutations(self, mutations, *, upsert: bool = True) -> bool:
        raise RuntimeError("engine unavailable: provenance batch failed")


def test_write_work_item_provenance_surfaces_total_batch_failure():
    engine = _RaisingProvenanceEngine()

    result = worker._write_work_item_provenance(
        engine,
        work_item_id="workitem:fixture-1",
        claim={"work_item_id": "workitem:fixture-1", "lease_id": "lease-1"},
        agent_id="agent:fixture",
        status="completed",
        result="the result",
        evidence=SimpleNamespace(confidence=1.0),
        policy_decision_node=_policy_decision(),
        grant_id=None,
    )

    # Previously this returned None on both success AND total failure --
    # there was no way for a caller to tell them apart. It must now return a
    # signal distinguishing "batch raised" from "written" (BUG-015/GOC-20 B8).
    assert result is not None
    assert result == "failed"


def test_write_work_item_provenance_reports_unavailable_with_no_batch_capability():
    """A backend exposing no ``batch_typed_mutations`` at all must report
    ``"unavailable"`` — not silently attempt a per-node fallback that could
    partially succeed (no serial fallback per the lane's design)."""

    class _NoBatchEngine:
        pass

    result = worker._write_work_item_provenance(
        _NoBatchEngine(),
        work_item_id="workitem:fixture-1",
        claim={"work_item_id": "workitem:fixture-1", "lease_id": "lease-1"},
        agent_id="agent:fixture",
        status="completed",
        result="the result",
        evidence=SimpleNamespace(confidence=1.0),
        policy_decision_node=_policy_decision(),
        grant_id=None,
    )
    assert result == "unavailable"


def test_write_work_item_provenance_reports_written_on_success():
    """Known-good counterpart: a real one-shot atomic batch commits all four
    provenance nodes together and the caller is told so explicitly."""

    class _AtomicEngine:
        def __init__(self) -> None:
            self.batches: list[list[dict]] = []

        def batch_typed_mutations(self, mutations, *, upsert: bool = True) -> bool:
            self.batches.append(mutations)
            return True

    engine = _AtomicEngine()
    result = worker._write_work_item_provenance(
        engine,
        work_item_id="workitem:fixture-1",
        claim={"work_item_id": "workitem:fixture-1", "lease_id": "lease-1"},
        agent_id="agent:fixture",
        status="completed",
        result="the result",
        evidence=SimpleNamespace(confidence=1.0),
        policy_decision_node=_policy_decision(),
        grant_id=None,
    )
    assert result == "written"
    # ONE atomic call, all four nodes -- not four independent add_node calls.
    assert len(engine.batches) == 1
    assert {m["node_type"] for m in engine.batches[0]} == {
        "Observation",
        "Claim",
        "Action",
        "Trace",
    }
