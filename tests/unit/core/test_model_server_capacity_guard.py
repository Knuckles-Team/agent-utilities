"""Model-server-capacity guard: the aggregate ceiling + circuit breaker.

CONCEPT:AU-ORCH.dispatch.embedding-fanout (per-endpoint server-capacity ceiling shared by every demand
source) / ORCH-1.103 (capacity-aware backpressure + circuit breaking) / KG-2.298
(the ``max_concurrent_requests`` config). Proves the four guarantees:

(a) the AGGREGATE in-flight to one endpoint never exceeds the configured ceiling
    even when three demand sources (embeds + enrichment + orchestration) all
    saturate the SAME model at once;
(b) a simulated 503/429 trips the breaker → the client BACKS OFF (a measurable
    cooldown) and recovers via a single half-open probe — no retry storm;
(c) the priority edict still holds WITHIN the ceiling (interactive ahead of a
    saturating background fan-out), proving the outer gate is priority-aware;
(d) the default ceiling is conservative AND configurable per endpoint.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from agent_utilities.core import model_circuit_breaker as cb
from agent_utilities.core import model_concurrency as mc
from agent_utilities.core.config import ChatModelConfig
from agent_utilities.core.model_circuit_breaker import (
    CircuitState,
    ModelCircuitBreaker,
)


@pytest.fixture(autouse=True)
def _isolate():
    mc.reset_controllers()
    yield
    mc.reset_controllers()


def _install_model(
    monkeypatch, model_id: str, *, ceiling: int | None, declared: int = 1
):
    from agent_utilities.core import config as config_mod

    monkeypatch.setattr(
        config_mod.config,
        "chat_models",
        [
            ChatModelConfig(
                id=model_id,
                provider="openai",
                parallel_instances=1,
                max_parallel_calls=declared,
                max_concurrent_requests=ceiling,
            )
        ],
    )


# --- (d) the ceiling: conservative default + configurable --------------------
def test_default_ceiling_is_conservative(monkeypatch):
    monkeypatch.delenv("MODEL_MAX_CONCURRENT_REQUESTS", raising=False)
    # An unknown / under-declared model can NEVER ramp to MODEL_MAX_CONCURRENCY(512):
    # it is capped at the conservative default (32), not the local CPU count.
    assert mc.server_ceiling("totally-unknown-model") == 32


def test_ceiling_global_default_env_overrides(monkeypatch):
    monkeypatch.setenv("MODEL_MAX_CONCURRENT_REQUESTS", "8")
    assert mc.server_ceiling("totally-unknown-model") == 8


def test_explicit_per_model_ceiling_wins(monkeypatch):
    _install_model(monkeypatch, "gb10", ceiling=5, declared=64)
    # Explicit server capacity wins absolutely — even BELOW the optimistic
    # instances×calls product (the box genuinely can't sustain 64).
    assert mc.server_ceiling("gb10") == 5


def test_ceiling_never_below_declared_capacity(monkeypatch):
    monkeypatch.setenv("MODEL_MAX_CONCURRENT_REQUESTS", "8")
    _install_model(monkeypatch, "big", ceiling=None, declared=20)
    # No explicit cap → max(declared=20, default=8) = 20 (never throttle below what
    # the model declares it can serve).
    assert mc.server_ceiling("big") == 20


def test_resolve_capacity_is_clamped_to_ceiling(monkeypatch):
    monkeypatch.setenv("KG_ADAPTIVE_CONCURRENCY", "false")
    _install_model(monkeypatch, "capped", ceiling=4, declared=64)
    # Static capacity is 64 but the server ceiling is 4 → resolve clamps to 4.
    assert mc.resolve_capacity("capped") == 4
    assert mc.server_ceiling("capped") == 4


# --- (a) aggregate in-flight across ALL demand sources <= the ceiling ---------
def test_aggregate_in_flight_never_exceeds_endpoint_ceiling(monkeypatch):
    monkeypatch.setenv("KG_ADAPTIVE_CONCURRENCY", "false")
    ceiling = 4
    _install_model(monkeypatch, "shared-endpoint", ceiling=ceiling, declared=8)

    lock = threading.Lock()
    state = {"cur": 0, "peak": 0}

    async def work(_x):
        with lock:
            state["cur"] += 1
            state["peak"] = max(state["peak"], state["cur"])
        await asyncio.sleep(0.01)
        with lock:
            state["cur"] -= 1
        return _x

    async def driver():
        # THREE independent demand sources all hammering the SAME endpoint at once,
        # each asking for far more concurrency (8) than the server can take.
        items = list(range(24))
        await asyncio.gather(
            mc.map_concurrent(items, work, model="shared-endpoint", capacity=8),
            mc.map_concurrent(items, work, model="shared-endpoint", capacity=8),
            mc.map_concurrent(items, work, model="shared-endpoint", capacity=8),
        )

    asyncio.run(driver())
    # The SUM across all three sources is bounded by the ONE shared endpoint ceiling.
    assert state["peak"] <= ceiling, f"peak={state['peak']} exceeded ceiling={ceiling}"
    # And it actually used the capacity (not over-throttled to ~1).
    assert state["peak"] >= 2


# --- (c) the priority edict holds WITHIN the ceiling (outer gate is priority) -
def test_priority_edict_holds_within_ceiling_via_fanout(monkeypatch):
    monkeypatch.setenv("KG_ADAPTIVE_CONCURRENCY", "false")
    ceiling = 2
    _install_model(monkeypatch, "prio-endpoint", ceiling=ceiling, declared=8)
    from agent_utilities.core.resource_priority import PriorityClass, priority_scope

    release = asyncio.Event()
    interactive_admitted = asyncio.Event()

    async def bg_hold(_x):
        await release.wait()  # saturate the endpoint
        return _x

    async def interactive(_x):
        interactive_admitted.set()
        return _x

    async def driver():
        with priority_scope(PriorityClass.BACKGROUND_INGESTION):
            bg = asyncio.create_task(
                mc.map_concurrent(
                    list(range(16)), bg_hold, model="prio-endpoint", capacity=8
                )
            )
        await asyncio.sleep(0.05)  # let background saturate the reserved-minus headroom
        with priority_scope(PriorityClass.INTERACTIVE):
            itask = asyncio.create_task(
                mc.map_concurrent([1], interactive, model="prio-endpoint", capacity=1)
            )
        # Interactive lands in the reserved headroom immediately despite saturation.
        await asyncio.wait_for(interactive_admitted.wait(), timeout=1.0)
        await itask
        release.set()
        await bg

    asyncio.run(driver())
    assert interactive_admitted.is_set()


# --- (b) the circuit breaker: 503/429 → backoff, not a retry storm ------------
def test_breaker_trips_on_overload_and_backs_off():
    b = ModelCircuitBreaker(
        "m", fail_threshold=1, base_cooldown_s=0.05, max_cooldown_s=0.2
    )
    assert b.state is CircuitState.CLOSED
    # A single 503 trips it (react to the first sign of saturation).
    b.record(ok=False, status=503)
    assert b.state is CircuitState.OPEN
    # before_call now BACKS OFF for ~the cooldown instead of hammering.
    start = time.monotonic()
    b.before_call_sync()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.04, f"breaker did not back off (elapsed={elapsed})"


def test_breaker_half_open_probe_recovers():
    b = ModelCircuitBreaker(
        "m", fail_threshold=1, base_cooldown_s=0.02, max_cooldown_s=0.1
    )
    b.record(ok=False, status=429)
    assert b.state is CircuitState.OPEN
    time.sleep(0.03)  # let the cooldown elapse
    b.before_call_sync()  # admitted as the single half-open probe
    assert b.state is CircuitState.HALF_OPEN
    b.record(ok=True)  # probe succeeds → close + reset backoff
    assert b.state is CircuitState.CLOSED


def test_breaker_backoff_is_exponential():
    b = ModelCircuitBreaker(
        "m",
        fail_threshold=1,
        base_cooldown_s=0.01,
        max_cooldown_s=10.0,
        backoff_factor=2.0,
    )
    b.record(ok=False, status=503)
    c1 = b.snapshot()["cooldown_s"]
    # A failed half-open probe re-opens with a LONGER cooldown.
    time.sleep(0.02)
    b.before_call_sync()  # become the probe
    b.record(ok=False, status=503)  # probe fails
    c2 = b.snapshot()["cooldown_s"]
    assert c2 > c1


def test_benign_non_overload_error_does_not_trip():
    b = ModelCircuitBreaker("m", fail_threshold=1)
    b.record(ok=False, status=400)  # a client error, not capacity saturation
    assert b.state is CircuitState.CLOSED


def test_breaker_disabled_is_noop(monkeypatch):
    monkeypatch.setenv("MODEL_CIRCUIT_BREAKER", "false")
    b = cb.get_circuit_breaker("disabled-model")
    b.record(ok=False, status=503)
    # Disabled → never trips, before_call returns immediately.
    assert b.state is CircuitState.CLOSED
    start = time.monotonic()
    b.before_call_sync()
    assert time.monotonic() - start < 0.02


def test_fanout_overload_trips_the_endpoint_breaker(monkeypatch):
    monkeypatch.setenv("KG_ADAPTIVE_CONCURRENCY", "false")
    _install_model(monkeypatch, "saturating", ceiling=4)

    class _Overloaded(Exception):
        status_code = 503

    def boom(_x):
        raise _Overloaded("server out of memory")

    # The fan-out feeds the overload into the per-endpoint breaker (wiring proof).
    with pytest.raises(_Overloaded):
        mc.map_concurrent_sync([1], boom, model="saturating", capacity=1)
    assert cb.get_circuit_breaker("saturating").state is CircuitState.OPEN
