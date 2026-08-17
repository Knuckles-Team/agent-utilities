"""GOC-17 / BUG-001 — truthful activation executor (P0, Phase-0 green foundation).

**The defect (source-confirmed, W01 orientation):** production's ONLY activation
entrypoint — the ``agent-activation-worker`` console script
(``agent_utilities.orchestration.agent_activation:main``) — never calls
:func:`agent_activation.set_activation_executor`. Grepping the whole repository for
every caller of ``set_activation_executor``/``process_one_activation``/
``run_activation_worker_loop``/``start_activation_worker_pool`` (the full call graph)
shows the setter is invoked ONLY from this test module and
``test_agent_activation.py`` — never from ``main()`` or any other production code
path. So a production worker's ``_EXECUTOR`` module global is permanently ``None``,
and :func:`agent_activation.process_one_activation` silently falls back to
``_default_executor`` — which drains the mailbox, writes one ``:ToolCall`` per
message, and commits the WorkItem ``succeeded`` — WITHOUT ever invoking a real
agent/tool turn. A consumer reading the WorkItem status or the ``:ToolCall`` nodes
cannot tell this apart from a genuine execution.

This module is the BUG-001 evidence + regression suite:

1. ``test_bug001_reproduction_*`` — the FAILING-FIRST proof. It encodes the REQUIRED
   (fixed) behaviour — an unbound executor with diagnostic mode off must fail closed,
   never report ``succeeded``, and never write a ``:ToolCall``. Run against the
   pre-fix code, it FAILS (proving the defect is real); after the fix it passes.
2. ``test_bug001_known_bad_input_*`` — the guard-rejects-bad-input demonstration
   (acceptance gate 5): a non-callable "executor" must be rejected at the SAME
   chokepoint (:func:`set_activation_executor`) that binds the canonical one.
3. ``test_bug001_no_toolcall_on_a_path_that_reports_success`` — the regression test
   for acceptance gate 6: sweep every way the code can reach a "succeeded" WorkItem
   and prove none of them can do so via the receipt-only path without being tagged
   ``executor_status=unavailable`` and WITHOUT writing a ``:ToolCall``.
"""

from __future__ import annotations

import threading

import pytest

from agent_utilities.orchestration import agent_activation as aa
from agent_utilities.orchestration import work_item as wi
from tests.unit.orchestration.test_agent_activation import ActivationEngine


@pytest.fixture
def engine() -> ActivationEngine:
    return ActivationEngine()


@pytest.fixture(autouse=True)
def _clean_activation_globals():
    """Every test starts from the TRUE production default: unbound, non-diagnostic.

    This is the exact state ``main()`` is in before it does anything — the state that
    let BUG-001 reproduce.
    """
    aa.set_activation_executor(None)
    aa.set_activation_diagnostic_mode(False)
    yield
    aa.set_activation_executor(None)
    aa.set_activation_diagnostic_mode(False)


# ── 1. failing-first reproduction ─────────────────────────────────────────────────────


def test_bug001_reproduction_unbound_executor_must_not_report_success(
    engine: ActivationEngine,
) -> None:
    """BUG-001 failing-first proof.

    Sanity-checks the exact fresh-process state (nothing bound, diagnostic mode off —
    this is what a freshly started ``agent-activation-worker`` looks like BEFORE this
    fix), then drives one activation through the real claim/execute/commit path and
    asserts the WorkItem can NOT be reported ``succeeded`` and NO ``:ToolCall`` can be
    written. Pre-fix, this assertion fails: the current code commits ``succeeded`` and
    writes a ``:ToolCall`` for ``activation.acknowledge`` despite never running an
    agent turn.
    """
    assert aa._EXECUTOR is None, "sanity: nothing bound (fresh module state)"
    assert not aa.activation_diagnostic_mode_enabled(), (
        "sanity: no dev/test override active"
    )

    instance_id = aa.register_agent_instance(
        engine, agent_name="prod-agent", tenant="tenant-prod"
    )
    wid = aa.deliver_activation(
        engine,
        instance_id,
        message_ref="msg:real-work",
        source=aa.ActivationSource.DIRECT,
    )
    claim = wi.claim_next(
        engine, resource_class=aa.WORK_ITEM_KIND, queue=aa.WORK_ITEM_KIND
    )
    assert claim is not None

    aa.process_one_activation(engine, claim, token=str(claim["lease_owner"]))

    item = wi.get_work_item(engine, wid)
    assert item is not None
    assert item["status"] != "succeeded", (
        "BUG-001: an unbound executor with diagnostic mode OFF reported the WorkItem "
        f"'succeeded' (got status={item['status']!r}) — a production worker can claim "
        "to have run an agent it never ran."
    )
    tool_calls = engine.by_label(aa._TOOLCALL_LABEL)
    assert tool_calls == [], (
        "BUG-001: a :ToolCall was written on a path with NO bound canonical executor "
        f"and diagnostic mode off (found {tool_calls!r}) — receipt-only acknowledgement "
        "must never be indistinguishable from a real tool call."
    )


def test_bug001_reproduction_via_the_real_worker_loop(
    engine: ActivationEngine,
) -> None:
    """Same reproduction, but through the ACTUAL production loop
    (:func:`run_activation_worker_loop`, the function ``main()`` starts N threads of)
    rather than a direct :func:`process_one_activation` call — proving the defect
    (and the fix) hold at the entrypoint a real deployment uses, not only at the
    lower-level function tests exercise directly.
    """
    instance_id = aa.register_agent_instance(
        engine, agent_name="prod-agent-2", tenant="t"
    )
    wid = aa.deliver_activation(
        engine, instance_id, message_ref="msg:x", source="direct"
    )
    stop = threading.Event()
    processed = aa.run_activation_worker_loop(
        engine, stop, tenants=["t"], max_activations=1
    )
    assert processed == 1
    item = wi.get_work_item(engine, wid)
    assert item is not None
    assert item["status"] != "succeeded", (
        "BUG-001 (via the production worker loop): unbound executor reported success"
    )
    assert engine.by_label(aa._TOOLCALL_LABEL) == []


# ── 2. known-bad-input guard demonstration (acceptance gate 5) ───────────────────────


@pytest.mark.parametrize(
    "bad_executor",
    [
        "not-callable-a-plain-string",
        123,
        object(),
        {"executor": "dict-is-not-callable"},
    ],
)
def test_bug001_known_bad_input_non_callable_executor_is_rejected(bad_executor) -> None:
    """The chokepoint guard actively REJECTS a malformed executor, not just lets a good
    one through. Binding must raise before ``_EXECUTOR`` is ever mutated — a caller
    that fat-fingers the binding call cannot silently leave production running with a
    broken (non-callable) "executor" that would blow up on the first activation.
    """
    with pytest.raises(aa.ActivationReadinessError):
        aa.set_activation_executor(bad_executor)  # type: ignore[arg-type]
    # The guard must be a REAL rejection: the global stays unbound, not partially set.
    assert aa._EXECUTOR is None


def test_bug001_known_bad_input_readiness_reports_unavailable(
    engine: ActivationEngine,
) -> None:
    """A production-shaped readiness check must fail CLOSED (not raise, not silently
    pass) when nothing valid is bound — this is the input the real ``main()`` startup
    check evaluates before it will let the worker pool claim any WorkItem.
    """
    ready, reason = aa.activation_worker_readiness()
    assert ready is False
    assert "no canonical executor" in reason.lower() or "unbound" in reason.lower()


def test_bug001_good_input_is_accepted_and_readiness_passes() -> None:
    """The counterpart of the known-bad-input proof: a REAL callable executor is
    accepted and readiness turns positive — the guard doesn't merely reject
    everything."""

    def real_executor(ctx: aa.ActivationContext) -> aa.ActivationResult:
        return aa.ActivationResult(outcome="succeeded")

    aa.set_activation_executor(real_executor)
    assert aa._EXECUTOR is real_executor
    ready, _reason = aa.activation_worker_readiness()
    assert ready is True


def test_bug001_worker_pool_refuses_to_start_when_unready(
    engine: ActivationEngine,
) -> None:
    """The startup-time twin of the chokepoint guard: ``start_activation_worker_pool``
    (what ``main()`` calls) must REFUSE to start a single thread — not start and then
    silently never process anything — when nothing is bound and diagnostic mode is
    off. Acceptance gate 1: "Production startup binds canonical executor and fails
    readiness when unavailable.\""""
    with pytest.raises(aa.ActivationReadinessError):
        aa.start_activation_worker_pool(engine, worker_count=2)


def test_bug001_worker_pool_starts_once_canonical_executor_is_bound(
    engine: ActivationEngine,
) -> None:
    """Counterpart: once a real executor is bound, the SAME startup call the guard
    just refused now succeeds and returns live worker threads."""

    def real_executor(ctx: aa.ActivationContext) -> aa.ActivationResult:
        return aa.ActivationResult(outcome="succeeded")

    aa.set_activation_executor(real_executor)
    threads, stop = aa.start_activation_worker_pool(engine, worker_count=2)
    try:
        assert len(threads) == 2
        assert all(t.is_alive() for t in threads)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=5.0)


# ── 3. no receipt-only success can carry a :ToolCall (acceptance gate 6) ─────────────


def test_bug001_no_toolcall_on_a_path_that_reports_success(
    engine: ActivationEngine,
) -> None:
    """Diagnostic mode (the explicit dev/test opt-in) may still let the receipt-only
    default executor run and the WorkItem terminate ``succeeded`` (mailbox drain is a
    real, bounded effect worth allowing in dev) — but it must NEVER write a
    ``:ToolCall``, and the run must be tagged ``executor_status=unavailable`` so no
    consumer can mistake it for a real agent turn.
    """
    aa.set_activation_diagnostic_mode(True)
    instance_id = aa.register_agent_instance(
        engine, agent_name="diag-agent", tenant="t"
    )
    aa.deliver_activation(engine, instance_id, message_ref="m1", source="timer")
    stop = threading.Event()
    aa.run_activation_worker_loop(engine, stop, tenants=["t"], max_activations=1)

    assert engine.by_label(aa._TOOLCALL_LABEL) == [], (
        "receipt-only diagnostic mode must never write a :ToolCall"
    )
    receipts = engine.by_label(aa._RECEIPT_LABEL)
    assert len(receipts) == 1
    assert (
        receipts[0]["executor_status"] == aa.ActivationExecutorStatus.UNAVAILABLE.value
    )


def test_bug001_bound_executor_that_fails_is_tagged_failed_not_unavailable(
    engine: ActivationEngine,
) -> None:
    """A REAL executor that raises/fails is a genuine execution attempt — it must be
    tagged FAILED (an agent turn happened and errored), never conflated with
    UNAVAILABLE (no attempt was made at all). This is the EXECUTED/FAILED/UNAVAILABLE
    three-way distinction the lane requires."""

    def failing_executor(ctx: aa.ActivationContext) -> aa.ActivationResult:
        return aa.ActivationResult(outcome="failed", retryable=True, error_ref="boom")

    aa.set_activation_executor(failing_executor)
    instance_id = aa.register_agent_instance(
        engine, agent_name="fail-agent", tenant="t"
    )
    wid = aa.deliver_activation(engine, instance_id, message_ref="m", source="direct")
    claim = wi.claim_next(
        engine, resource_class=aa.WORK_ITEM_KIND, queue=aa.WORK_ITEM_KIND
    )
    assert claim is not None
    aa.process_one_activation(engine, claim, token=str(claim["lease_owner"]))

    item = wi.get_work_item(engine, wid)
    assert item is not None
    assert item["status"] in ("retry", "ready", "failed", "dead_letter", "leased")
    traces = engine.by_label(aa._RUNTRACE_LABEL)
    assert len(traces) == 1
    assert traces[0]["executor_status"] == aa.ActivationExecutorStatus.FAILED.value
    # A failed real attempt still never writes a receipt (it's not receipt-only).
    assert engine.by_label(aa._RECEIPT_LABEL) == []
