"""Per-model-endpoint circuit breaker + capacity-aware backpressure (CONCEPT:ORCH-1.103).

A saturating model server must slow the CLIENT, never crash the server. The
adaptive controller (CONCEPT:KG-2.145) already *shrinks* a model's concurrency
target when latency inflates or an overload status arrives, but shrinking the gate
width is a slow, statistical signal — it does not *stop the bleeding* the instant
the endpoint starts shedding load. This module adds the fast, decisive half: a
classic three-state circuit breaker per model endpoint, modelled on the graph-os
engine breaker.

* **CLOSED** — normal. Calls pass straight through.
* **OPEN** — the endpoint just shed load (a ``429``/``503``/``504``/timeout, or an
  opaque-under-load failure). New calls BACK OFF: they wait out an
  **exponential-backoff cooldown** instead of hammering a server that is already
  over its memory budget. This is the anti-retry-storm guarantee.
* **HALF_OPEN** — once the cooldown elapses, exactly ONE probe call is admitted. If
  it succeeds the breaker closes (and the backoff resets); if it fails the breaker
  re-opens with a longer cooldown.

The breaker composes with — and sits OUTSIDE — the server-capacity ceiling gate
(CONCEPT:ORCH-1.102): the ceiling decides the steady-state MAX in flight; the
breaker reacts to a server that is *already* saturating by throttling the client
to near-zero until it recovers. It is pure in-process bookkeeping (a lock + a few
timestamps), feeds off the SAME ``(ok, status)`` samples the fan-out already
collects, and is a complete no-op while the endpoint is healthy.

Enabled by default; disable with ``MODEL_CIRCUIT_BREAKER=false``. Tunables:
``MODEL_BREAKER_FAIL_THRESHOLD`` (consecutive overloads to trip, default ``1`` —
react to the first sign of saturation), ``MODEL_BREAKER_BASE_COOLDOWN_S``
(default ``0.5``), ``MODEL_BREAKER_MAX_COOLDOWN_S`` (default ``30``),
``MODEL_BREAKER_BACKOFF_FACTOR`` (default ``2.0``).
"""

from __future__ import annotations

import asyncio
import threading
import time
from enum import Enum

__all__ = [
    "CircuitState",
    "ModelCircuitBreaker",
    "get_circuit_breaker",
    "reset_circuit_breakers",
    "OVERLOAD_STATUSES",
]

# Statuses that mean "the endpoint is overloaded / shedding load" — the same set
# the adaptive controller treats as a congestion event (CONCEPT:KG-2.145).
OVERLOAD_STATUSES = frozenset({429, 502, 503, 504, 529})

# A single backpressure sleep is sliced so OPEN→HALF_OPEN transitions are picked up
# promptly even when the configured cooldown is long.
_MAX_SLEEP_SLICE_S = 0.25
# Defaults (overridable via env; see module docstring).
_DEFAULT_FAIL_THRESHOLD = 1
_DEFAULT_BASE_COOLDOWN_S = 0.5
_DEFAULT_MAX_COOLDOWN_S = 30.0
_DEFAULT_BACKOFF_FACTOR = 2.0


class CircuitState(Enum):
    """The three breaker states (CONCEPT:ORCH-1.103)."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


def _is_overload(ok: bool, status: int | None) -> bool:
    """A call counts as a saturation signal when its status is an overload status,
    or it failed opaquely (no status) — an opaque failure under load is treated as
    congestion-ish, mirroring the adaptive controller."""
    if status in OVERLOAD_STATUSES:
        return True
    return bool(not ok and status is None)


class ModelCircuitBreaker:
    """A three-state circuit breaker for ONE model endpoint (CONCEPT:ORCH-1.103).

    Symmetric async (:meth:`before_call`) and sync (:meth:`before_call_sync`)
    admission faces share one logic core and one lock, so an async orchestration
    call and a sync enrichment call to the same endpoint trip and recover the same
    breaker. ``record`` feeds the ``(ok, status)`` outcome of each call back in.
    """

    def __init__(
        self,
        model_key: str,
        *,
        fail_threshold: int = _DEFAULT_FAIL_THRESHOLD,
        base_cooldown_s: float = _DEFAULT_BASE_COOLDOWN_S,
        max_cooldown_s: float = _DEFAULT_MAX_COOLDOWN_S,
        backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
    ) -> None:
        self.model_key = model_key
        self.fail_threshold = max(1, int(fail_threshold))
        self.base_cooldown_s = max(0.0, float(base_cooldown_s))
        self.max_cooldown_s = max(self.base_cooldown_s, float(max_cooldown_s))
        self.backoff_factor = max(1.0, float(backoff_factor))
        self._state = CircuitState.CLOSED
        self._consecutive = 0
        self._trips = 0
        self._open_until = 0.0
        self._probe_in_flight = False
        self._probe_deadline = 0.0
        self._lock = threading.Lock()

    # -- the testable decision core (caller holds nothing) -------------------
    def _cooldown(self) -> float:
        """Exponential backoff cooldown for the current trip count, capped."""
        exp = max(0, self._trips - 1)
        return min(
            self.max_cooldown_s, self.base_cooldown_s * (self.backoff_factor**exp)
        )

    def _wait_for(self, now: float) -> float:
        """Seconds to back off before proceeding; ``0`` ⇒ proceed now.

        Reserves the single HALF_OPEN probe for the first caller that finds the
        cooldown elapsed. Caller-agnostic (used by both faces). Caller holds nothing;
        this takes the lock itself.
        """
        with self._lock:
            if self._state is CircuitState.CLOSED:
                return 0.0
            if self._state is CircuitState.OPEN:
                if now >= self._open_until:
                    # Cooldown elapsed → become the half-open probe.
                    self._state = CircuitState.HALF_OPEN
                    self._probe_in_flight = True
                    self._probe_deadline = now + self._cooldown() + self.max_cooldown_s
                    return 0.0
                return min(_MAX_SLEEP_SLICE_S, self._open_until - now)
            # HALF_OPEN
            if not self._probe_in_flight or now >= self._probe_deadline:
                # No probe running (or the prior probe never reported back) → take it.
                self._probe_in_flight = True
                self._probe_deadline = now + self._cooldown() + self.max_cooldown_s
                return 0.0
            # Another caller is probing → wait a short slice and re-check.
            return _MAX_SLEEP_SLICE_S

    def record(self, *, ok: bool, status: int | None = None) -> None:
        """Feed one call's outcome back into the breaker (CONCEPT:ORCH-1.103).

        An overload sample trips the breaker (immediately when half-open, else after
        ``fail_threshold`` consecutive overloads); any non-overload outcome — a
        success OR a benign error — is a recovery signal that closes it and resets
        the backoff. Never raises.
        """
        if not _enabled():
            return
        overload = _is_overload(ok, status)
        with self._lock:
            was_half = self._state is CircuitState.HALF_OPEN
            if overload:
                self._consecutive += 1
                if was_half or self._consecutive >= self.fail_threshold:
                    self._trips += 1
                    self._state = CircuitState.OPEN
                    self._open_until = time.monotonic() + self._cooldown()
                self._probe_in_flight = False
            else:
                # Success or benign (non-capacity) error → recover.
                self._consecutive = 0
                if self._state is not CircuitState.CLOSED:
                    self._state = CircuitState.CLOSED
                    self._trips = 0
                self._probe_in_flight = False

    # -- async face ----------------------------------------------------------
    async def before_call(self) -> None:
        """Async backpressure gate: await out the cooldown when the breaker is open."""
        if not _enabled():
            return
        while True:
            wait = self._wait_for(time.monotonic())
            if wait <= 0:
                return
            await asyncio.sleep(min(wait, _MAX_SLEEP_SLICE_S))

    # -- sync face -----------------------------------------------------------
    def before_call_sync(self) -> None:
        """Sync backpressure gate: sleep out the cooldown when the breaker is open."""
        if not _enabled():
            return
        while True:
            wait = self._wait_for(time.monotonic())
            if wait <= 0:
                return
            time.sleep(min(wait, _MAX_SLEEP_SLICE_S))

    # -- introspection (tests / observability) -------------------------------
    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "model": self.model_key,
                "state": self._state.value,
                "consecutive_overloads": self._consecutive,
                "trips": self._trips,
                "cooldown_s": self._cooldown(),
                "open_for_s": max(0.0, self._open_until - time.monotonic()),
            }


# --- module config + cached per-model breakers ------------------------------
def _enabled() -> bool:
    from agent_utilities.core._env import setting

    return bool(setting("MODEL_CIRCUIT_BREAKER", True))


def _tunables() -> dict[str, float]:
    from agent_utilities.core._env import setting

    return {
        "fail_threshold": int(
            setting("MODEL_BREAKER_FAIL_THRESHOLD", _DEFAULT_FAIL_THRESHOLD)
        ),
        "base_cooldown_s": float(
            setting("MODEL_BREAKER_BASE_COOLDOWN_S", _DEFAULT_BASE_COOLDOWN_S)
        ),
        "max_cooldown_s": float(
            setting("MODEL_BREAKER_MAX_COOLDOWN_S", _DEFAULT_MAX_COOLDOWN_S)
        ),
        "backoff_factor": float(
            setting("MODEL_BREAKER_BACKOFF_FACTOR", _DEFAULT_BACKOFF_FACTOR)
        ),
    }


_lock = threading.Lock()
_breakers: dict[str, ModelCircuitBreaker] = {}


def _key(model: str | None) -> str:
    return (model or "__default__").strip().lower() or "__default__"


def get_circuit_breaker(model: str | None = None) -> ModelCircuitBreaker:
    """Return (creating if needed) the cached breaker for ``model`` (CONCEPT:ORCH-1.103).

    One breaker per model endpoint, so embeds + enrichment + orchestration on the
    SAME endpoint trip and recover together — a server shedding load slows ALL of
    them, not just the one stream that saw the 503.
    """
    k = _key(model)
    with _lock:
        b = _breakers.get(k)
        if b is None:
            tun = _tunables()
            b = ModelCircuitBreaker(
                k,
                fail_threshold=int(tun["fail_threshold"]),
                base_cooldown_s=tun["base_cooldown_s"],
                max_cooldown_s=tun["max_cooldown_s"],
                backoff_factor=tun["backoff_factor"],
            )
            _breakers[k] = b
        return b


def reset_circuit_breakers() -> None:
    """Drop all cached breakers (test isolation / config reload). CONCEPT:ORCH-1.103."""
    with _lock:
        _breakers.clear()
