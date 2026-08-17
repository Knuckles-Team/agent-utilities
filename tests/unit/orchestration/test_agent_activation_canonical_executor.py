"""GOC-17 — the canonical executor is REALLY invocable, not merely reported ready.

``activation_worker_readiness()`` returning ``True`` after
:func:`agent_activation.set_activation_executor(agent_activation.canonical_activation_executor)`
is itself a status signal — this program's invariant is that a status signal must never
disagree with reality (BUG-001's whole point). These tests are the known-bad/known-good
proof pair the lane report requires:

1. ``canonical_activation_executor`` genuinely calls
   ``orchestration.manager.Orchestrator.execute_agent`` (the ONE gateway every other
   delegation entrypoint converges on) — not a stub, not a local acknowledgement. Proven
   by patching the *class* ``canonical_activation_executor`` imports at call time and
   asserting the exact ``agent_name``/``task``/``run_id``/``allowed_tools`` it receives.
2. A canonical-executor FAILURE (the gateway raises) is committed ``failed``/retryable
   and tagged ``executor_status=failed`` — never silently swallowed into a false
   ``succeeded`` and never conflated with ``unavailable`` (no attempt made).
3. A canonical-executor SUCCESS runs through :func:`process_one_activation` end to end —
   real WorkItem claim → real statechart activate/deactivate → the bound executor called
   with the drained mailbox → a ``:RunTrace``/``:ToolCall`` (never a receipt) → WorkItem
   committed ``succeeded`` — proving the "available capability" the readiness probe
   reports is the SAME capability that just ran, not a parallel unverified claim.
"""

from __future__ import annotations

import threading
from unittest import mock

import pytest

from agent_utilities.orchestration import agent_activation as aa
from agent_utilities.orchestration import work_item as wi
from tests.unit.orchestration.test_agent_activation import ActivationEngine


@pytest.fixture
def engine() -> ActivationEngine:
    return ActivationEngine()


@pytest.fixture(autouse=True)
def _clean_activation_globals():
    aa.set_activation_executor(None)
    aa.set_activation_diagnostic_mode(False)
    yield
    aa.set_activation_executor(None)
    aa.set_activation_diagnostic_mode(False)


def _fake_orchestrator_class(execute_agent_mock: mock.AsyncMock) -> type:
    """A stand-in for ``orchestration.manager.Orchestrator`` that records the call
    without touching a live engine/session — ``canonical_activation_executor`` only
    needs ``Orchestrator(engine).execute_agent(...)`` to resolve."""

    class _FakeOrchestrator:
        def __init__(self, engine: object) -> None:
            self.engine = engine
            self.execute_agent = execute_agent_mock

    return _FakeOrchestrator


# ── 1 + 2. the bound capability genuinely reaches the canonical gateway ──────────────


def test_canonical_executor_actually_calls_orchestrator_execute_agent(
    engine: ActivationEngine,
) -> None:
    """A known-good proof: the SAME executor ``main()`` binds in production really
    invokes ``Orchestrator.execute_agent`` with the activation's real agent name, task,
    run id, and allowed tools — not a local mailbox acknowledgement."""
    execute_agent_mock = mock.AsyncMock(return_value="the agent's final answer")
    fake_cls = _fake_orchestrator_class(execute_agent_mock)

    ctx = aa.ActivationContext(
        engine=engine,
        instance_id="inst-1",
        agent_name="researcher",
        tenant="tenant-a",
        work_item_id="wi-1",
        run_id="run-1",
        messages=({"source": "direct", "message_ref": "msg:do-the-thing"},),
        priority_class=aa.PriorityClass.INTERACTIVE,
        delegation=None,
        model_id="",
        tool_ids=("search",),
    )

    with mock.patch("agent_utilities.orchestration.manager.Orchestrator", fake_cls):
        result = aa.canonical_activation_executor(ctx)

    assert execute_agent_mock.await_count == 1
    _args, kwargs = execute_agent_mock.await_args
    assert kwargs["agent_name"] == "researcher"
    assert kwargs["run_id"] == "run-1"
    assert kwargs["allowed_tools"] == ["search"]
    assert "msg:do-the-thing" in kwargs["task"]

    assert result.outcome == "succeeded"
    assert result.result_ref == "activation-output:run-1"
    assert result.tool_calls[0]["tool"] == "orchestration.execute_agent"


def test_canonical_executor_gateway_failure_is_reported_failed_not_success(
    engine: ActivationEngine,
) -> None:
    """Known-bad-input proof: the gateway itself raising must surface as a truthful
    ``failed``/retryable outcome — never swallowed, never reported ``succeeded``."""
    execute_agent_mock = mock.AsyncMock(side_effect=RuntimeError("gateway exploded"))
    fake_cls = _fake_orchestrator_class(execute_agent_mock)

    ctx = aa.ActivationContext(
        engine=engine,
        instance_id="inst-2",
        agent_name="researcher",
        tenant="tenant-a",
        work_item_id="wi-2",
        run_id="run-2",
        messages=(),
        priority_class=aa.PriorityClass.INTERACTIVE,
        delegation=None,
    )

    with mock.patch("agent_utilities.orchestration.manager.Orchestrator", fake_cls):
        result = aa.canonical_activation_executor(ctx)

    assert result.outcome == "failed"
    assert result.retryable is True
    assert result.error_ref is not None and "RuntimeError" in result.error_ref
    # A genuine attempt was made; the chokepoint (process_one_activation) is
    # responsible for tagging FAILED vs UNAVAILABLE — verified end to end below.


# ── 3. end-to-end: the readiness-reported capability is the one that actually ran ────


def test_bound_canonical_executor_runs_end_to_end_through_process_one_activation(
    engine: ActivationEngine,
) -> None:
    """The full loop: bind the canonical executor exactly as ``main()`` does, claim a
    real WorkItem, run ``process_one_activation``, and prove the gateway was invoked
    with the drained mailbox AND the WorkItem/provenance reflect a real execution —
    never the receipt-only shape (no ``:ActivationReceipt``, a real ``:ToolCall``
    naming ``orchestration.execute_agent``, ``executor_status=executed``)."""
    execute_agent_mock = mock.AsyncMock(return_value="ok")
    fake_cls = _fake_orchestrator_class(execute_agent_mock)

    aa.set_activation_executor(aa.canonical_activation_executor)
    ready, reason = aa.activation_worker_readiness()
    assert ready is True, f"readiness must report True once bound: {reason}"

    instance_id = aa.register_agent_instance(
        engine, agent_name="researcher", tenant="t"
    )
    wid = aa.deliver_activation(
        engine, instance_id, message_ref="msg:real-task", source="direct"
    )
    claim = wi.claim_next(
        engine, resource_class=aa.WORK_ITEM_KIND, queue=aa.WORK_ITEM_KIND
    )
    assert claim is not None

    with mock.patch("agent_utilities.orchestration.manager.Orchestrator", fake_cls):
        aa.process_one_activation(engine, claim, token=str(claim["lease_owner"]))

    # The readiness signal was truthful: the capability it reported as available is
    # the exact one that ran.
    assert execute_agent_mock.await_count == 1
    _args, kwargs = execute_agent_mock.await_args
    assert "msg:real-task" in kwargs["task"]

    item = wi.get_work_item(engine, wid)
    assert item is not None
    assert item["status"] == "succeeded"

    tool_calls = engine.by_label(aa._TOOLCALL_LABEL)
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool"] == "orchestration.execute_agent"
    traces = engine.by_label(aa._RUNTRACE_LABEL)
    assert len(traces) == 1
    assert traces[0]["executor_status"] == aa.ActivationExecutorStatus.EXECUTED.value
    # Never the receipt-only shape on a real-executor path.
    assert engine.by_label(aa._RECEIPT_LABEL) == []


def test_bound_canonical_executor_failure_ends_up_failed_never_succeeded(
    engine: ActivationEngine,
) -> None:
    """Same end-to-end path, but the gateway fails — the WorkItem must NOT be
    committed succeeded and the run trace must carry ``executor_status=failed``."""
    execute_agent_mock = mock.AsyncMock(side_effect=RuntimeError("boom"))
    fake_cls = _fake_orchestrator_class(execute_agent_mock)

    aa.set_activation_executor(aa.canonical_activation_executor)

    instance_id = aa.register_agent_instance(
        engine, agent_name="researcher", tenant="t"
    )
    wid = aa.deliver_activation(
        engine, instance_id, message_ref="msg:will-fail", source="direct"
    )
    claim = wi.claim_next(
        engine, resource_class=aa.WORK_ITEM_KIND, queue=aa.WORK_ITEM_KIND
    )
    assert claim is not None

    with mock.patch("agent_utilities.orchestration.manager.Orchestrator", fake_cls):
        aa.process_one_activation(engine, claim, token=str(claim["lease_owner"]))

    item = wi.get_work_item(engine, wid)
    assert item is not None
    assert item["status"] != "succeeded", (
        "a canonical-executor failure must never be committed as WorkItem success"
    )
    traces = engine.by_label(aa._RUNTRACE_LABEL)
    assert len(traces) == 1
    assert traces[0]["executor_status"] == aa.ActivationExecutorStatus.FAILED.value
    assert engine.by_label(aa._RECEIPT_LABEL) == []


def test_worker_loop_end_to_end_with_canonical_executor(
    engine: ActivationEngine,
) -> None:
    """The SAME proof through ``run_activation_worker_loop`` — the function ``main()``
    starts N threads of — not only the lower-level ``process_one_activation``."""
    execute_agent_mock = mock.AsyncMock(return_value="ok")
    fake_cls = _fake_orchestrator_class(execute_agent_mock)

    aa.set_activation_executor(aa.canonical_activation_executor)
    instance_id = aa.register_agent_instance(
        engine, agent_name="researcher", tenant="t"
    )
    wid = aa.deliver_activation(
        engine, instance_id, message_ref="msg:loop", source="direct"
    )
    stop = threading.Event()
    with mock.patch("agent_utilities.orchestration.manager.Orchestrator", fake_cls):
        processed = aa.run_activation_worker_loop(
            engine, stop, tenants=["t"], max_activations=1
        )
    assert processed == 1
    assert execute_agent_mock.await_count == 1
    item = wi.get_work_item(engine, wid)
    assert item is not None
    assert item["status"] == "succeeded"
    assert engine.by_label(aa._RECEIPT_LABEL) == []
