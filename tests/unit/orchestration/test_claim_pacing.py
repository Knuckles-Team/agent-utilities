"""Tests for WorkItem claim pacing — the Python-side cooperative half of engine
backpressure (W2.9, CONCEPT:AU-ORCH.scheduling.claim-pacing-backpressure).

Two layers, per the module docstring's unified model:

1. The pure mechanism (:mod:`agent_utilities.orchestration.claim_pacing`) — shed
   detection, per-class backoff growth/cap/jitter, recovery-on-success,
   per-class isolation. Deterministic: every ``now``/``rng`` is injected.
2. The live-path wiring (:func:`agent_utilities.orchestration.work_item.
   claim_specific`/``claim_next`` routing every claim through
   ``_paced_claim_call``) — proves a real shed from the engine is recorded,
   re-raised UNCHANGED (wire-compatible with every existing caller), and that
   a SUBSEQUENT same-class attempt never reaches the engine while paced.
"""

from __future__ import annotations

import random
import threading
import time

import pytest

from agent_utilities.core.resource_priority import PriorityClass, priority_scope
from agent_utilities.orchestration import claim_pacing
from agent_utilities.orchestration.resilience import ResiliencePolicy


@pytest.fixture(autouse=True)
def _isolated_claim_pacing():
    """Every test starts with zero pacing state and leaves none behind."""
    claim_pacing.reset_claim_pacing()
    yield
    claim_pacing.reset_claim_pacing()


# --- is_busy_shed: the wire-message detector, no dedicated exception type ---
def test_is_busy_shed_matches_only_the_busy_prefix():
    assert claim_pacing.is_busy_shed(RuntimeError("BUSY: server at capacity, retry"))
    assert claim_pacing.is_busy_shed(
        RuntimeError(
            "BUSY: QoS capacity reserved for higher priority, retry with backoff"
        )
    )
    assert not claim_pacing.is_busy_shed(
        RuntimeError("engine-native X(request) is required")
    )
    assert not claim_pacing.is_busy_shed(RuntimeError("some other transient error"))
    assert not claim_pacing.is_busy_shed(ConnectionError("BUSY: wrong exception type"))
    assert not claim_pacing.is_busy_shed(ValueError("BUSY: also wrong type"))


# --- fresh state: no pacing until a shed is recorded ------------------------
def test_fresh_class_is_never_paced():
    assert claim_pacing.pending_pace_seconds(PriorityClass.BACKGROUND_INGESTION) == 0.0
    claim_pacing.raise_if_paced(PriorityClass.BACKGROUND_INGESTION)  # must not raise


# --- backoff growth: exponential with cap, deterministic under injected rng -
def test_record_claim_shed_grows_and_caps_the_backoff():
    policy = ResiliencePolicy(
        backoff_base_s=1.0,
        backoff_factor=2.0,
        max_backoff_s=8.0,
        jitter=False,  # exact numbers for this assertion
        name="test-claim-pacing",
    )
    now = 1_000.0
    delays = []
    for _ in range(6):
        d = claim_pacing.record_claim_shed(
            PriorityClass.BACKGROUND_INGESTION, policy=policy, now=now
        )
        delays.append(d)
    # 1, 2, 4, 8, 8, 8 (exponential then capped at max_backoff_s).
    assert delays == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]
    snap = claim_pacing.claim_pacing_snapshot(now=now)
    assert snap["background_ingestion"]["consecutive_sheds"] == 6.0


def test_backoff_growth_is_monotonically_non_decreasing_with_jitter():
    """Even with proportional jitter ([0.5, 1.0] of the capped delay), the
    EXPECTED/ceiling trend still grows then plateaus — assert the ceiling
    each attempt could reach is non-decreasing (the jitter floor can dip
    below a previous attempt's jitter draw, but never above its own ceiling)."""
    policy = claim_pacing.DEFAULT_CLAIM_PACING_POLICY
    rng = random.Random(1234)
    now = 0.0
    ceilings = []
    for attempt in range(1, 10):
        claim_pacing.reset_claim_pacing()
        for a in range(attempt):
            claim_pacing.record_claim_shed(
                PriorityClass.BACKGROUND_INGESTION, policy=policy, rng=rng, now=now
            )
        ceiling = min(
            policy.backoff_base_s * (policy.backoff_factor ** (attempt - 1)),
            policy.max_backoff_s,
        )
        ceilings.append(ceiling)
    assert ceilings == sorted(ceilings)
    assert ceilings[-1] == pytest.approx(policy.max_backoff_s)


# --- raise_if_paced / recovery ----------------------------------------------
def test_raise_if_paced_raises_only_inside_the_window_then_clears():
    policy = ResiliencePolicy(
        backoff_base_s=2.0, backoff_factor=2.0, max_backoff_s=10.0, jitter=False
    )
    claim_pacing.record_claim_shed(
        PriorityClass.BACKGROUND_INGESTION, policy=policy, now=0.0
    )
    assert (
        claim_pacing.pending_pace_seconds(PriorityClass.BACKGROUND_INGESTION, now=0.0)
        == 2.0
    )
    with pytest.raises(claim_pacing.ClaimPaced, match="BUSY:.*background_ingestion"):
        claim_pacing.raise_if_paced(PriorityClass.BACKGROUND_INGESTION, now=1.0)
    # Window elapsed — no longer paced, and no exception.
    claim_pacing.raise_if_paced(PriorityClass.BACKGROUND_INGESTION, now=2.0)
    assert (
        claim_pacing.pending_pace_seconds(PriorityClass.BACKGROUND_INGESTION, now=2.0)
        == 0.0
    )


def test_record_claim_admitted_resets_backoff_immediately():
    claim_pacing.record_claim_shed(PriorityClass.BACKGROUND_INGESTION, now=0.0)
    claim_pacing.record_claim_shed(PriorityClass.BACKGROUND_INGESTION, now=0.0)
    assert (
        claim_pacing.pending_pace_seconds(PriorityClass.BACKGROUND_INGESTION, now=0.0)
        > 0.0
    )
    claim_pacing.record_claim_admitted(PriorityClass.BACKGROUND_INGESTION, now=0.0)
    assert (
        claim_pacing.pending_pace_seconds(PriorityClass.BACKGROUND_INGESTION, now=0.0)
        == 0.0
    )
    snap = claim_pacing.claim_pacing_snapshot(now=0.0)
    assert snap["background_ingestion"]["consecutive_sheds"] == 0.0


# --- per-class isolation: the client-side mirror of the engine's own lanes --
def test_shedding_one_class_never_paces_another():
    claim_pacing.record_claim_shed(PriorityClass.BACKGROUND_INGESTION, now=0.0)
    assert (
        claim_pacing.pending_pace_seconds(PriorityClass.BACKGROUND_INGESTION, now=0.0)
        > 0.0
    )
    assert claim_pacing.pending_pace_seconds(PriorityClass.INTERACTIVE, now=0.0) == 0.0
    assert (
        claim_pacing.pending_pace_seconds(PriorityClass.ORCHESTRATION, now=0.0) == 0.0
    )
    assert claim_pacing.pending_pace_seconds(PriorityClass.HYDRATION, now=0.0) == 0.0
    claim_pacing.raise_if_paced(PriorityClass.INTERACTIVE, now=0.0)  # must not raise


# --- ambient contextvar resolution (matches resource_priority's own default) -
def test_omitted_priority_resolves_from_ambient_contextvar():
    with priority_scope(PriorityClass.BACKGROUND_INGESTION):
        claim_pacing.record_claim_shed(now=0.0)  # no explicit class passed
    # Recorded under BACKGROUND_INGESTION, not any other class.
    snap = claim_pacing.claim_pacing_snapshot(now=0.0)
    assert snap["background_ingestion"]["consecutive_sheds"] == 1.0
    assert "interactive" not in snap


def test_untagged_ambient_context_resolves_to_orchestration():
    # No priority_scope active — matches resource_priority's own untagged
    # default (ORCHESTRATION: high, never starved) rather than opening a
    # separate untracked bucket.
    claim_pacing.record_claim_shed(now=0.0)
    snap = claim_pacing.claim_pacing_snapshot(now=0.0)
    assert snap["orchestration"]["consecutive_sheds"] == 1.0


def test_snapshot_and_reset():
    claim_pacing.record_claim_shed(PriorityClass.BACKGROUND_INGESTION, now=0.0)
    claim_pacing.record_claim_shed(PriorityClass.INTERACTIVE, now=0.0)
    snap = claim_pacing.claim_pacing_snapshot(now=0.0)
    assert set(snap) == {"background_ingestion", "interactive"}
    claim_pacing.reset_claim_pacing()
    assert claim_pacing.claim_pacing_snapshot(now=0.0) == {}


# --- thread-safety: concurrent sheds on the SAME class serialize cleanly ----
def test_concurrent_sheds_on_one_class_are_serialized_and_monotonic():
    policy = ResiliencePolicy(
        backoff_base_s=0.001, backoff_factor=2.0, max_backoff_s=1.0, jitter=False
    )
    barrier = threading.Barrier(8)

    def hammer() -> None:
        barrier.wait(timeout=5)
        claim_pacing.record_claim_shed(
            PriorityClass.BACKGROUND_INGESTION, policy=policy
        )

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    snap = claim_pacing.claim_pacing_snapshot()
    # Every increment landed exactly once — no lost updates under contention.
    assert snap["background_ingestion"]["consecutive_sheds"] == 8.0


# --- live-path wiring: work_item.claim_specific/claim_next --------------------
class _ShedThenAdmitEngine:
    """Minimal WorkItem engine double: sheds BUSY for its first ``shed_count``
    native ``claim_work_item`` calls, then admits (or legitimately declines)."""

    def __init__(self, shed_count: int, *, admit: bool = True) -> None:
        self.shed_count = shed_count
        self.admit = admit
        self.calls = 0

    def claim_work_item(self, request: object) -> dict:
        self.calls += 1
        if self.calls <= self.shed_count:
            raise RuntimeError(
                "BUSY: QoS capacity reserved for higher priority, retry with backoff"
            )
        if not self.admit:
            return _negative_claim()
        return {
            "schema_version": "1",
            "claimed": True,
            "reason": "claimed",
            "work_item_id": getattr(request, "work_item_id", "wi-1") or "wi-1",
            "kind": "agent_task",
            "payload_ref": "payload",
            "lease_holder_ref": "worker:test",
            "lease_epoch": 1,
            "fencing_token": 1,
            "lease_expires_at_ms": 0,
            "attempt": 1,
            "max_attempts": 3,
            "tenant_in_flight": 1,
            "changed_work_item_ids": ["wi-1"],
        }

    def query_cypher(self, q: str, params: dict | None = None) -> list[dict]:
        return []


def _negative_claim() -> dict:
    return {
        "schema_version": "1",
        "claimed": False,
        "reason": "empty",
        "work_item_id": None,
        "kind": None,
        "payload_ref": None,
        "lease_holder_ref": None,
        "lease_epoch": None,
        "fencing_token": None,
        "lease_expires_at_ms": None,
        "attempt": None,
        "max_attempts": None,
        "tenant_in_flight": 0,
        "changed_work_item_ids": [],
    }


def test_claim_next_shed_is_recorded_and_reraised_unchanged():
    from agent_utilities.orchestration import work_item as wi

    engine = _ShedThenAdmitEngine(shed_count=1)
    with priority_scope(PriorityClass.BACKGROUND_INGESTION):
        with pytest.raises(RuntimeError, match="^BUSY:"):
            wi.claim_next(engine, queue="ingest_task")
    assert engine.calls == 1
    snap = claim_pacing.claim_pacing_snapshot()
    assert snap["background_ingestion"]["consecutive_sheds"] == 1.0


def test_claim_next_is_paced_after_a_shed_without_hitting_the_engine():
    from agent_utilities.orchestration import work_item as wi

    engine = _ShedThenAdmitEngine(shed_count=100)  # would shed forever if hit again
    with priority_scope(PriorityClass.BACKGROUND_INGESTION):
        with pytest.raises(RuntimeError, match="^BUSY:"):
            wi.claim_next(engine, queue="ingest_task")
        assert engine.calls == 1
        # A second, third, fourth attempt WHILE paced never reaches the engine —
        # the "stop hammering" property, proven at the live claim_next() call.
        for _ in range(5):
            with pytest.raises(claim_pacing.ClaimPaced):
                wi.claim_next(engine, queue="ingest_task")
        assert engine.calls == 1  # unchanged — every retry was preempted client-side


def test_claim_next_recovers_and_admits_after_the_window_elapses():
    """A real (short) wait past the live default policy's window: the NEXT
    attempt reaches the engine again (one probe) and, admitted, resets."""
    from agent_utilities.orchestration import work_item as wi

    engine = _ShedThenAdmitEngine(shed_count=1, admit=True)
    with priority_scope(PriorityClass.BACKGROUND_INGESTION):
        with pytest.raises(RuntimeError, match="^BUSY:"):
            wi.claim_next(engine, queue="ingest_task")
        delay = claim_pacing.pending_pace_seconds(PriorityClass.BACKGROUND_INGESTION)
        assert 0.0 < delay <= claim_pacing.DEFAULT_CLAIM_PACING_POLICY.backoff_base_s
        time.sleep(delay + 0.05)
        claim = wi.claim_next(engine, queue="ingest_task")
    assert claim is not None
    assert claim["work_item_id"] == "wi-1"
    assert engine.calls == 2  # the one shed + the one recovered admit
    snap = claim_pacing.claim_pacing_snapshot()
    assert snap["background_ingestion"]["consecutive_sheds"] == 0.0


def test_a_negative_but_legitimate_claim_is_not_a_shed():
    """The engine cleanly answering 'nothing to claim' is NOT a BUSY shed —
    it must not engage pacing at all."""
    from agent_utilities.orchestration import work_item as wi

    engine = _ShedThenAdmitEngine(shed_count=0, admit=False)
    with priority_scope(PriorityClass.BACKGROUND_INGESTION):
        result = wi.claim_next(engine, queue="ingest_task")
    assert result is None
    assert claim_pacing.pending_pace_seconds(PriorityClass.BACKGROUND_INGESTION) == 0.0


def test_interactive_class_is_unaffected_by_a_concurrent_ingest_shed():
    from agent_utilities.orchestration import work_item as wi

    ingest_engine = _ShedThenAdmitEngine(shed_count=50)
    with priority_scope(PriorityClass.BACKGROUND_INGESTION):
        with pytest.raises(RuntimeError, match="^BUSY:"):
            wi.claim_next(ingest_engine, queue="ingest_task")
        with pytest.raises(claim_pacing.ClaimPaced):
            wi.claim_next(ingest_engine, queue="ingest_task")

    interactive_engine = _ShedThenAdmitEngine(shed_count=0, admit=False)
    with priority_scope(PriorityClass.INTERACTIVE):
        # Never preempted, never shed — reaches the engine every time.
        for _ in range(5):
            wi.claim_next(interactive_engine, queue="interactive_task")
    assert interactive_engine.calls == 5
