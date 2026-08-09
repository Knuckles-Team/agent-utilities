"""BUG-015 (GOC-20) — a successful run can lack result or provenance.

Fault-injection at every provenance write boundary a ``run_agent``/WorkItem
completion touches, proving the specific defect the lane names: a run (or a
WorkItem) can be REPORTED as successful while its RunTrace/Outcome/ToolCall
provenance write silently failed. See
``plans/graph-os-completion-program/decisions/GOC-20-atomic-outcome-provenance.md``
for the fault-injection matrix these tests back and the chosen remediation.

Three ``agent_runner.py`` boundaries are fixed here (this lane owns that
file): the ``enterprise`` early-exit, the ``ServiceRegistry`` early-exit, and
the main dispatch path's run_summary/synthesis/done reporting. Two
``agent_dispatch_worker.py`` boundaries are FAULT-INJECTED AND DOCUMENTED
ONLY — GOC-18 owns that file (see the program's Phase-0 hard rule); those
tests demonstrate the defect and are expected to keep failing red until
GOC-18 applies the change specified in the decision record.
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

    real_result = {
        "results": {"output": "Found 3 running containers: web, db, cache."}
    }

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
# Boundary 4/5 — agent_dispatch_worker.py WorkItem provenance (GOC-18 owns
# this file; NOT edited here per the Phase-0 worker rules). These document
# the defect via direct fault injection at the write boundary and are
# expected to stay RED until GOC-18 applies the change specified in
# decisions/GOC-20-atomic-outcome-provenance.md.
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
    """``add_node`` fails ONLY for the OutcomeEvaluation append (boundary 4)."""

    def __init__(self) -> None:
        self.nodes: list[tuple[str, str]] = []

    def add_node(self, node_id, node_type, properties=None):
        if node_type == "OutcomeEvaluation":
            raise RuntimeError("engine unavailable: OutcomeEvaluation write failed")
        self.nodes.append((node_id, node_type))


@pytest.mark.xfail(
    reason=(
        "BUG-015 boundary 4 (agent_dispatch_worker._finalize_work_item): the "
        "OutcomeEvaluation append failure is swallowed and does not change "
        "the returned commit status. GOC-18 owns this file; fix specified in "
        "decisions/GOC-20-atomic-outcome-provenance.md, not applied here."
    ),
    strict=True,
)
def test_finalize_work_item_hides_outcome_evaluation_write_failure(monkeypatch):
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

    # This is the ASSERTION THAT SHOULD HOLD once fixed: a WorkItem cannot
    # report "committed" while its required OutcomeEvaluation node is absent.
    # Today it returns "committed" unconditionally (xfail proves the gap).
    assert result != "committed"


class _RaisingProvenanceEngine:
    """``add_node`` fails for every provenance node (boundary 5)."""

    def add_node(self, *_a, **_k):
        raise RuntimeError("engine unavailable: provenance write failed")


@pytest.mark.xfail(
    reason=(
        "BUG-015 boundary 5 (agent_dispatch_worker._write_work_item_provenance): "
        "a total provenance-write failure is caught, logged, and silently "
        "returns None -- the caller (execute_work_item_turn) never checks a "
        "return value at all, so 'completed' commits with zero Observation/"
        "Claim/Action/Trace nodes. GOC-18 owns this file; fix specified in "
        "decisions/GOC-20-atomic-outcome-provenance.md, not applied here."
    ),
    strict=True,
)
def test_write_work_item_provenance_gives_no_caller_visible_signal_on_total_failure():
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

    # The function returns None on both success AND total failure today --
    # there is no way for a caller to tell them apart. Once fixed it must
    # return a signal execute_work_item_turn can act on.
    assert result is not None
