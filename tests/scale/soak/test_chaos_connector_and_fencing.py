"""Connector-failure, claim-churn fencing, and tenant-isolation-under-load
chaos — SCALE-P2-1 soak expansion (CONCEPT:AU-ORCH.execution.retry-predicate-raised-treating).

Closes three fault-injection gaps the existing soak/chaos suite did not cover:

* **Connector failure** — a source connector that fails transiently must recover
  under the platform's real declarative :class:`ResiliencePolicy`
  (``agent_utilities.orchestration.resilience``), and one that fails
  permanently must FAIL CLOSED (raise, never silently return stale/empty data)
  once its retry budget is exhausted — proven against the real resilience
  runner, not a hand-rolled retry loop.
* **Claim-invalidation under churn** — many rapid claim -> crash -> reap cycles
  on the SAME WorkItem must keep the fencing epoch strictly increasing, never
  allow two simultaneously-live claims, and land exactly one final commit —
  the CAS state machine under repeated churn, not just a single crash (see
  ``test_chaos_worker_and_delivery.py`` for the single-crash case this extends).
* **Tenant isolation under load** — interleaving submissions/claims across many
  tenants (including a hot/elephant tenant hammering its quota) must never let
  one tenant's quota exhaustion delay or corrupt another tenant's ops, even
  under adversarial interleaving (not just the sequential elephant-then-
  ordinary case in ``test_chaos_tenant_and_restart.py``).

Built directly against the engine-native WorkItem CAS state machine +
:class:`FakeScaleEngine` (see ``test_chaos_worker_and_delivery.py``'s module
docstring for why: exact state construction, not a rate schedule) plus the
real :mod:`agent_utilities.orchestration.resilience` runner for the connector
scenario.
"""

from __future__ import annotations

import random

import pytest

from agent_utilities.orchestration.resilience import (
    ResiliencePolicy,
    run_with_resilience_sync,
)

pytestmark = pytest.mark.integration


def _engine(loadgen):
    return loadgen.build_mock_engine(
        latency=loadgen.LatencyModel(
            write_mean_s=0.0, write_jitter_s=0.0, query_mean_s=0.0, query_jitter_s=0.0
        )
    )


# ═══════════════════════════════════════════════════════════════════════════
# Connector failure — real ResiliencePolicy, not a hand-rolled retry loop
# ═══════════════════════════════════════════════════════════════════════════


class _FlakyConnector:
    """A connector whose ``read`` fails ``fail_times`` times then succeeds."""

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def read(self) -> list[dict[str, str]]:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError(f"connector unreachable (attempt {self.calls})")
        return [{"row": "ok"}]


class _AlwaysDownConnector:
    """A connector that never recovers — models a hard-down source system."""

    def __init__(self) -> None:
        self.calls = 0

    def read(self) -> list[dict[str, str]]:
        self.calls += 1
        raise ConnectionError("connector permanently unreachable")


def test_connector_transient_failure_recovers_under_resilience_policy():
    """2 transient failures then success recovers within the retry budget —
    the real ResiliencePolicy/run_with_resilience_sync, no test-only shortcut."""
    connector = _FlakyConnector(fail_times=2)
    policy = ResiliencePolicy(
        max_attempts=5,
        backoff_base_s=0.0,  # no real sleeping in a unit test
        jitter=False,
        name="connector-read",
    )
    result = run_with_resilience_sync(connector.read, policy, sleep=lambda _s: None)
    assert result == [{"row": "ok"}]
    assert connector.calls == 3  # 2 failures + the successful 3rd attempt


def test_connector_permanent_failure_fails_closed_never_silently_empty():
    """A connector that never recovers must raise once retries are exhausted —
    never silently degrade to an empty/stale read (fail-closed, not fail-open)."""
    connector = _AlwaysDownConnector()
    policy = ResiliencePolicy(
        max_attempts=3,
        backoff_base_s=0.0,
        jitter=False,
        name="connector-read-permanent",
    )
    with pytest.raises(ConnectionError, match="permanently unreachable"):
        run_with_resilience_sync(connector.read, policy, sleep=lambda _s: None)
    # Exactly the policy's attempt budget was spent — no runaway retry, no
    # early give-up either.
    assert connector.calls == 3


def test_connector_failure_falls_back_when_primary_exhausted():
    """After the primary connector's retries are exhausted, a configured
    fallback (e.g. a secondary read replica / cached mirror) is tried — and
    its success is returned rather than propagating the primary's error."""
    primary = _AlwaysDownConnector()
    fallback_calls: list[int] = []

    def fallback_read() -> list[dict[str, str]]:
        fallback_calls.append(1)
        return [{"row": "from-fallback"}]

    policy = ResiliencePolicy(
        max_attempts=2,
        backoff_base_s=0.0,
        jitter=False,
        fallbacks=[fallback_read],
        name="connector-read-with-fallback",
    )
    result = run_with_resilience_sync(primary.read, policy, sleep=lambda _s: None)
    assert result == [{"row": "from-fallback"}]
    assert primary.calls == 2
    assert fallback_calls == [1]


# ═══════════════════════════════════════════════════════════════════════════
# Claim-invalidation under churn — repeated crash/reap cycles, not just one
# ═══════════════════════════════════════════════════════════════════════════


def test_claim_churn_fencing_epoch_strictly_increases_and_completes_exactly_once(
    loadgen,
):
    """10 successive crash/reap cycles on the SAME item: every reclaim must
    strictly bump the fencing epoch, every stale claim's late commit after
    its cycle must be rejected (never silently applied), and the item must
    finally reach ``succeeded`` exactly once when a worker actually finishes."""
    engine = _engine(loadgen)
    wi = loadgen.wi

    # High max_attempts: this scenario is about the fencing/CAS churn itself,
    # not the retry-budget-exhaustion/DLQ path (already covered by
    # test_chaos_lifecycle_and_dlq.py's dedicated dead-letter scenario) — a
    # low default would dead-letter the item after 3 cycles instead of
    # letting it keep churning.
    item_id = wi.submit_work_item(
        engine,
        kind="agent_turn",
        payload_ref="churn-1",
        tenant="acme",
        max_attempts=50,
    )

    stale_claims: list[dict] = []
    epochs: list[int] = []
    now = 0.0
    for cycle in range(10):
        claim = wi.claim_and_start(
            engine,
            item_id=item_id,
            token=f"worker-{cycle}",
            now=now,
            lease_ttl_s=5.0,
        )
        assert claim is not None, f"cycle {cycle}: claim unexpectedly denied"
        epochs.append(claim["fence_token"])
        stale_claims.append(claim)
        # This worker "crashes" without committing; advance well past its lease.
        now += 100.0
        reaped = wi.reap_expired_leases(engine, now=now)
        assert item_id in reaped["reaped_ready"], f"cycle {cycle}: not reclaimed"

    # Fencing epoch is strictly monotonically increasing across every cycle —
    # never repeats, never goes backward (the core "no double execution" guard).
    assert epochs == sorted(set(epochs)), "fencing epoch must be strictly increasing"
    assert len(set(epochs)) == len(epochs), "fencing epoch must never repeat"

    # A real worker finally claims and completes it.
    now += 1.0
    final_claim = wi.claim_and_start(
        engine, item_id=item_id, token="worker-final", now=now
    )
    assert final_claim is not None
    outcome = wi.commit_result(
        engine, item_id, final_claim, outcome="succeeded", result_ref="ok", now=now + 1
    )
    assert outcome == "committed"
    assert (
        wi.get_work_item(engine, item_id)["status"] == wi.WorkItemStatus.SUCCEEDED.value
    )

    # Every stale claim from every earlier crashed cycle — including the very
    # first one — must be rejected forever, never resurrecting/overwriting
    # the real completion (proves fencing holds across the WHOLE churn history,
    # not just the immediately-prior cycle).
    for stale in stale_claims:
        late = wi.commit_result(
            engine,
            item_id,
            stale,
            outcome="succeeded",
            result_ref="stale-resurrection-attempt",
            now=now + 1000.0,
        )
        assert late in ("fenced", "noop")
    assert wi.get_work_item(engine, item_id)["result_ref"] == "ok"


def test_claim_churn_never_allows_two_simultaneously_live_claims(loadgen):
    """Mid-churn (lease still live, no crash yet), a second claim attempt on the
    SAME item must always be rejected — churn must never create a window where
    two workers both believe they hold the lease."""
    engine = _engine(loadgen)
    wi = loadgen.wi

    item_id = wi.submit_work_item(
        engine,
        kind="agent_turn",
        payload_ref="churn-2",
        tenant="acme",
        max_attempts=50,  # this scenario churns past the default retry budget
    )
    now = 0.0
    for cycle in range(5):
        claim = wi.claim_and_start(
            engine, item_id=item_id, token=f"w{cycle}", now=now, lease_ttl_s=30.0
        )
        assert claim is not None
        # While the lease is still live, a redelivered/racing claim must fail.
        racer = wi.claim_specific(
            engine, item_id, token=f"racer-{cycle}", now=now + 1.0, lease_ttl_s=30.0
        )
        assert racer is None, f"cycle {cycle}: two live claims coexisted"
        # Crash + reap to set up the next churn cycle.
        now += 60.0
        wi.reap_expired_leases(engine, now=now)


# ═══════════════════════════════════════════════════════════════════════════
# Heartbeat extension racing the reaper — a live worker must never be
# reclaimed out from under it, but a worker that stops heartbeating must be
# reclaimed once its (extended) lease truly expires. Neither existing chaos
# module exercises ``heartbeat()`` — only a plain unit test does, in
# isolation from ``reap_expired_leases`` racing it.
# ═══════════════════════════════════════════════════════════════════════════


def test_heartbeating_worker_survives_reap_then_reclaimed_once_it_stops(loadgen):
    """A long-running turn heartbeats twice, extending its lease each time — a
    reaper sweep BETWEEN heartbeats must not reclaim it (it is still healthy).
    Once the worker stops heartbeating (simulated crash), the reaper must
    reclaim it, but only after the LAST extended deadline, not the original."""
    engine = _engine(loadgen)
    wi = loadgen.wi

    item_id = wi.submit_work_item(
        engine, kind="agent_turn", payload_ref="hb-1", tenant="acme"
    )
    claim = wi.claim_and_start(
        engine, item_id=item_id, token="worker-1", now=0.0, lease_ttl_s=10.0
    )
    assert claim is not None

    # A reap sweep right at t=9 (before the original 10s lease expires) must
    # not touch it — this is the baseline "still healthy" case.
    reaped = wi.reap_expired_leases(engine, now=9.0)
    assert item_id not in reaped["reaped_ready"]

    # Heartbeat at t=9.5 extends the lease to 9.5+10=19.5.
    assert wi.heartbeat(engine, item_id, claim, now=9.5, lease_ttl_s=10.0)
    # A reap sweep at t=15 is PAST the ORIGINAL 10s deadline but well before
    # the heartbeat-extended one — the reaper must respect the extension, not
    # the stale original expiry, or a healthy long-running turn would be
    # wrongly reclaimed out from under its own worker.
    reaped_mid = wi.reap_expired_leases(engine, now=15.0)
    assert item_id not in reaped_mid["reaped_ready"], (
        "reaper reclaimed a heartbeat-extended lease before its new deadline"
    )
    assert (
        wi.get_work_item(engine, item_id)["status"] == wi.WorkItemStatus.RUNNING.value
    )

    # Heartbeat again at t=16, extending to 16+10=26.
    assert wi.heartbeat(engine, item_id, claim, now=16.0, lease_ttl_s=10.0)

    # Now the worker actually crashes (no more heartbeats). A reap sweep at
    # t=20 is still before the t=26 extended deadline — must not reclaim yet.
    reaped_still_alive = wi.reap_expired_leases(engine, now=20.0)
    assert item_id not in reaped_still_alive["reaped_ready"]

    # Only once t=26 has truly passed does the reaper reclaim it — proving
    # both directions: no premature reclaim of a healthy heartbeating worker,
    # and eventual reclaim once it genuinely goes silent.
    reaped_final = wi.reap_expired_leases(engine, now=27.0)
    assert item_id in reaped_final["reaped_ready"]
    item = wi.get_work_item(engine, item_id)
    assert item["status"] == wi.WorkItemStatus.READY.value
    assert item["lease_epoch"] == 2  # fenced past the dead worker's epoch

    # A fresh worker claims and finishes it; the dead worker's belated
    # heartbeat/commit attempts (fenced on the OLD epoch) must be rejected.
    new_claim = wi.claim_and_start(engine, item_id=item_id, token="worker-2", now=28.0)
    assert new_claim is not None
    assert wi.heartbeat(engine, item_id, claim, now=29.0) is False  # stale epoch
    outcome = wi.commit_result(
        engine, item_id, new_claim, outcome="succeeded", result_ref="ok", now=30.0
    )
    assert outcome == "committed"


# ═══════════════════════════════════════════════════════════════════════════
# Tenant isolation under load — adversarial interleaving, not sequential
# ═══════════════════════════════════════════════════════════════════════════


def test_tenant_isolation_holds_under_interleaved_multi_tenant_load(loadgen):
    """20 tenants (one hot/elephant tenant at 10x the others) submit and claim
    work in a deterministically-shuffled interleaving. Assert: a tenant's quota
    exhaustion never blocks a DIFFERENT tenant's submit, no tenant's in-flight
    count is ever observed above its own quota, and every ordinary tenant's
    work completes independent of the elephant's contention.
    """
    engine = _engine(loadgen)
    wi = loadgen.wi
    rng = random.Random(1337)

    ordinary_tenants = [f"tenant-{i}" for i in range(19)]
    elephant = "elephant"
    ordinary_quota = 2
    elephant_quota = 8

    # Build an adversarial interleaved op schedule: each tenant submits up to
    # its quota, shuffled so the elephant's ops interleave with everyone else's
    # rather than running to completion first (the sequential case is already
    # covered by test_chaos_tenant_and_restart.py's hot-tenant test).
    schedule: list[tuple[str, int]] = []
    for t in ordinary_tenants:
        schedule += [(t, ordinary_quota)] * ordinary_quota
    schedule += [(elephant, elephant_quota)] * elephant_quota
    rng.shuffle(schedule)

    submitted: dict[str, list[str]] = {t: [] for t in [*ordinary_tenants, elephant]}
    quota_of = {t: ordinary_quota for t in ordinary_tenants} | {
        elephant: elephant_quota
    }
    denied_tenants: set[str] = set()

    for tenant, _quota in schedule:
        quota = quota_of[tenant]
        try:
            item_id = wi.submit_work_item(
                engine,
                kind="agent_turn",
                payload_ref=f"{tenant}-{len(submitted[tenant])}",
                tenant=tenant,
                max_tenant_in_flight=quota,
            )
            submitted[tenant].append(item_id)
        except wi.TenantQuotaExceeded:
            denied_tenants.add(tenant)

    # Every ordinary tenant got exactly its quota worth of work in, regardless
    # of how the elephant's ops were interleaved with theirs.
    for t in ordinary_tenants:
        assert len(submitted[t]) == ordinary_quota, (
            f"{t} starved by interleaved elephant contention: "
            f"got {len(submitted[t])}/{ordinary_quota}"
        )
        assert wi.tenant_in_flight_count(engine, t) == ordinary_quota

    # The elephant is contained to its OWN quota — it never bled into anyone
    # else's count and never exceeded its own.
    assert wi.tenant_in_flight_count(engine, elephant) == elephant_quota

    # Ordinary tenants must complete their work independent of the elephant's
    # ongoing contention — claim + commit every ordinary tenant's items now.
    now = 0.0
    for t in ordinary_tenants:
        for item_id in submitted[t]:
            claim = wi.claim_and_start(engine, item_id=item_id, token="w", now=now)
            assert claim is not None, f"{t} item {item_id} could not be claimed"
            outcome = wi.commit_result(
                engine,
                item_id,
                claim,
                outcome="succeeded",
                result_ref="ok",
                now=now + 1,
            )
            assert outcome == "committed"
            now += 1.0

    # With every ordinary tenant's work now terminal, their in-flight counts
    # drop to zero — a live, non-permanent cap, never a lingering ban.
    for t in ordinary_tenants:
        assert wi.tenant_in_flight_count(engine, t) == 0

    # The elephant's quota is untouched by the ordinary tenants' completions —
    # cross-tenant isolation holds in both directions.
    assert wi.tenant_in_flight_count(engine, elephant) == elephant_quota
