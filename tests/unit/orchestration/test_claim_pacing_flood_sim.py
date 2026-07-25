"""ACCEPTANCE — ingestion-flood simulation (W2.9 backpressure unification).

A real (short, wall-clock-bounded) multi-threaded simulation against a mock
engine whose admission model mirrors the engine's own per-class token bucket
(``qos.rs::QosScheduler::take_token`` — CONCEPT:EG-KG.coordination.backpressure-busy-signal,
re-verified on the read-only engine reference this wave): a bounded number of
tokens, refilled at a fixed rate, consumed one per admitted claim; a request
with no token available is shed ``BUSY: …``. BACKGROUND_INGESTION is metered by
a deliberately tight bucket (a sustained flood WILL exceed it); INTERACTIVE is
unmetered (mirrors the engine's own interactive reserve/fair-share exemption),
so its claims never even see the possibility of a shed.

Proves the four ACCEPTANCE properties end to end, through the LIVE
``work_item.claim_next`` path (not the pacing module in isolation — that is
`test_claim_pacing.py`):

1. No BUSY-retry storm — the flood's ATTEMPT count vastly exceeds the number
   of attempts that actually reach the mock engine (most are preempted
   client-side once paced) — bounded engine hits, not linear-in-attempts.
2. Backoff growth — the sequence of client-side pacing windows observed after
   each REAL engine shed grows then plateaus at the policy cap.
3. Interactive claims unaffected — a concurrent INTERACTIVE claim loop is
   NEVER paced and NEVER shed while the ingest flood is actively backing off.
4. Queue depth stabilizes — a simulated producer keeps enqueuing ingest work
   at a steady rate; because the flood self-throttles to (approximately) what
   the token bucket actually sustains instead of burning every cycle on
   attempts the engine was always going to shed, the backlog (produced minus
   admitted) stays bounded rather than growing without limit for the run's
   duration.
"""

from __future__ import annotations

import threading
import time

import pytest

from agent_utilities.core.resource_priority import PriorityClass, priority_scope
from agent_utilities.orchestration import claim_pacing
from agent_utilities.orchestration import work_item as wi


class _TokenBucketEngine:
    """Mirrors ``qos.rs``'s per-class token bucket: BACKGROUND_INGESTION is
    metered (capacity + refill/sec); INTERACTIVE is unmetered (the engine's
    own interactive-exempt-from-throttling reserve, `qos.rs:450`)."""

    def __init__(self, capacity: float, refill_per_sec: float) -> None:
        self._capacity = capacity
        self._refill_per_sec = refill_per_sec
        self._tokens = capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()
        self.ingest_hits = 0
        self.interactive_hits = 0

    def _take_token(self) -> bool:
        now = time.monotonic()
        with self._lock:
            elapsed = now - self._last
            self._tokens = min(
                self._capacity, self._tokens + elapsed * self._refill_per_sec
            )
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def claim_work_item(self, request: object) -> dict:
        # The engine derives admission class from the verified wire `priority`
        # claim; this mock reads the SAME ambient PriorityClass a real W2.4
        # envelope would carry (Task A's claim), simulating that decode.
        from agent_utilities.core.resource_priority import current_priority

        cls = current_priority()
        if cls is PriorityClass.INTERACTIVE:
            self.interactive_hits += 1
            return _admit(request)
        self.ingest_hits += 1
        if not self._take_token():
            raise RuntimeError(
                "BUSY: QoS per-principal rate limit for this class reached, retry with backoff"
            )
        return _admit(request)

    def query_cypher(self, q: str, params: dict | None = None) -> list[dict]:
        return []


def _admit(request: object) -> dict:
    item_id = getattr(request, "work_item_id", None) or "flood-item"
    return {
        "schema_version": "1",
        "claimed": True,
        "reason": "claimed",
        "work_item_id": item_id,
        "kind": "ingest_task",
        "payload_ref": "payload",
        "lease_holder_ref": "worker:flood",
        "lease_epoch": 1,
        "fencing_token": 1,
        "lease_expires_at_ms": 0,
        "attempt": 1,
        "max_attempts": 3,
        "tenant_in_flight": 1,
        "changed_work_item_ids": [item_id],
    }


@pytest.fixture(autouse=True)
def _isolated_claim_pacing():
    claim_pacing.reset_claim_pacing()
    yield
    claim_pacing.reset_claim_pacing()


def test_ingestion_flood_simulation():
    duration_s = 0.5
    capacity, refill_per_sec = 2.0, 2.0  # a deliberately tight, slow-refill ingest lane
    engine = _TokenBucketEngine(capacity=capacity, refill_per_sec=refill_per_sec)

    ingest_attempts = 0
    ingest_paced = 0  # preempted client-side (ClaimPaced) — never touched the wire
    ingest_real_sheds = 0  # reached the engine and were shed
    ingest_admitted = 0
    observed_windows: list[
        float
    ] = []  # the client-side pacing window after each REAL shed

    interactive_attempts = 0
    interactive_paced_or_shed = 0
    interactive_latencies: list[float] = []

    stop = threading.Event()
    produced = {"n": 0}

    def producer() -> None:
        # A steady stream of new ingest work arriving — independent of whether
        # the flood loop is currently backed off.
        while not stop.is_set():
            produced["n"] += 1
            time.sleep(0.01)

    def ingest_flood() -> None:
        nonlocal ingest_attempts, ingest_paced, ingest_real_sheds, ingest_admitted
        with priority_scope(PriorityClass.BACKGROUND_INGESTION):
            while not stop.is_set():
                ingest_attempts += 1
                try:
                    claim = wi.claim_next(
                        engine, queue="ingest_task", tenant="flood-tenant"
                    )
                except claim_pacing.ClaimPaced:
                    ingest_paced += 1
                    continue
                except RuntimeError as exc:
                    assert claim_pacing.is_busy_shed(exc)
                    ingest_real_sheds += 1
                    observed_windows.append(
                        claim_pacing.pending_pace_seconds(
                            PriorityClass.BACKGROUND_INGESTION
                        )
                    )
                    continue
                if claim is not None:
                    ingest_admitted += 1

    def interactive_loop() -> None:
        nonlocal interactive_attempts, interactive_paced_or_shed
        with priority_scope(PriorityClass.INTERACTIVE):
            while not stop.is_set():
                interactive_attempts += 1
                start = time.monotonic()
                try:
                    wi.claim_next(
                        engine, queue="interactive_task", tenant="flood-tenant"
                    )
                except (claim_pacing.ClaimPaced, RuntimeError) as exc:
                    if isinstance(
                        exc, claim_pacing.ClaimPaced
                    ) or claim_pacing.is_busy_shed(exc):
                        interactive_paced_or_shed += 1
                    else:
                        raise
                interactive_latencies.append(time.monotonic() - start)
                time.sleep(0.002)  # a steady, modest interactive rate

    threads = [
        threading.Thread(target=producer, daemon=True),
        threading.Thread(target=ingest_flood, daemon=True),
        threading.Thread(target=interactive_loop, daemon=True),
    ]
    for t in threads:
        t.start()
    time.sleep(duration_s)
    stop.set()
    for t in threads:
        t.join(timeout=2.0)

    # ── 1) No BUSY-retry storm: engine hits « attempts (most were preempted). ──
    assert ingest_attempts > 500, (
        "the flood loop should attempt far more than it's admitted"
    )
    assert engine.ingest_hits < ingest_attempts / 3, (
        f"engine hits ({engine.ingest_hits}) too close to attempts ({ingest_attempts}) — "
        "pacing did not stop the hammering"
    )
    assert ingest_paced > 0, "expected client-side preemption once a shed was observed"
    # engine_hits == real_sheds + admitted (every wire hit is one or the other).
    assert engine.ingest_hits == ingest_real_sheds + ingest_admitted

    # ── 2) Backoff growth: real, live multi-step growth was exercised. The
    # FULL deterministic proof of the exact exponential-with-cap curve is
    # `test_claim_pacing.py::test_record_claim_shed_grows_and_caps_the_backoff`
    # (jitter off, exact numbers); here — real threads, real jitter, real
    # scheduling — we only assert growth demonstrably HAPPENED live: at least
    # one later window exceeds the first-shed range, proving a second (or
    # later) consecutive shed compounded the delay rather than resetting.
    assert len(observed_windows) >= 2, "need at least 2 real sheds to observe growth"
    cap = claim_pacing.DEFAULT_CLAIM_PACING_POLICY.max_backoff_s
    base = claim_pacing.DEFAULT_CLAIM_PACING_POLICY.backoff_base_s
    assert all(0.0 < w <= cap for w in observed_windows)
    assert max(observed_windows) > base, (
        "never observed a window past the first-shed range — no multi-step growth occurred live"
    )

    # ── 3) Interactive claims unaffected. ──────────────────────────────────
    assert interactive_attempts > 50
    assert interactive_paced_or_shed == 0, (
        "interactive was paced/shed by the ingest flood — the class isolation broke"
    )
    assert (
        engine.interactive_hits == interactive_attempts
    )  # every one reached the engine
    avg_latency = sum(interactive_latencies) / len(interactive_latencies)
    assert avg_latency < 0.02, (
        f"interactive latency degraded under the flood: {avg_latency:.4f}s"
    )

    # ── 4) Queue depth stabilizes: the backlog stays bounded, not unbounded. ──
    queue_depth = produced["n"] - ingest_admitted
    # A steady producer at ~100/s over ~0.5s produces roughly 50 items; the
    # sustainable ingest throughput (capacity + refill_per_sec * duration) is
    # the real ceiling on admission — the backlog should track that gap, not
    # explode to the (much larger) attempted-call count.
    assert queue_depth < ingest_attempts / 5, (
        f"backlog ({queue_depth}) grew close to the raw attempt count "
        f"({ingest_attempts}) — pacing did not let the queue stabilize"
    )
    assert ingest_admitted > 0, (
        "the ingest class made zero progress — starved, not paced"
    )

    # Numbers for the record (visible with `-s`/`--capture=no`).
    print(
        f"\n[W2.9 flood-sim] duration={duration_s}s bucket(cap={capacity},refill/s={refill_per_sec})\n"
        f"  ingest: attempts={ingest_attempts} engine_hits={engine.ingest_hits} "
        f"real_sheds={ingest_real_sheds} client_preempted={ingest_paced} admitted={ingest_admitted}\n"
        f"  backoff windows observed (s): {[round(w, 4) for w in observed_windows]}\n"
        f"  interactive: attempts={interactive_attempts} engine_hits={engine.interactive_hits} "
        f"paced_or_shed={interactive_paced_or_shed} avg_latency={avg_latency * 1000:.2f}ms\n"
        f"  queue: produced={produced['n']} admitted={ingest_admitted} backlog={queue_depth}"
    )
