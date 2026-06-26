"""Circuit breaker for the epistemic-graph engine client path.

CONCEPT:OS-5.23 — Gateway Middle-Tier Hardening.

Every Python-side engine call goes through
:class:`~agent_utilities.knowledge_graph.core.graph_compute.GraphComputeEngine`,
whose ``SyncEpistemicGraphClient`` talks UDS/TCP to the Rust engine. When the
engine is down, every caller used to pay a full connect/timeout each time —
hammering a dead socket and stalling request handlers. This module adds a
small, shared circuit breaker per ENDPOINT:

* **closed** — calls pass through; ``ENGINE_BREAKER_THRESHOLD`` consecutive
  connect/timeout failures (``OSError``/``EOFError`` — application-level
  errors do NOT trip it) open the circuit.
* **open** — calls fail FAST with the typed :class:`EngineCircuitOpenError`
  (a ``ConnectionError`` subclass, so existing ``except ConnectionError``
  handlers keep working) until ``ENGINE_BREAKER_COOLDOWN`` elapses.
* **half-open** — exactly one probe call is allowed through; success closes
  the circuit, failure re-opens it for another cooldown.

One breaker instance is shared per endpoint (process-wide registry), so all
``GraphComputeEngine`` instances pointing at the same engine agree. State is
exported on ``agent_utilities_gateway_engine_breaker_state{endpoint}`` and
call outcomes on ``agent_utilities_gateway_engine_requests_total{op,outcome}``.

``ENGINE_BREAKER_THRESHOLD=0`` disables tripping (calls still pass through
and outcome metrics still flow).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from agent_utilities.observability.gateway_metrics import (
    ENGINE_BREAKER_STATE,
    ENGINE_REQUESTS,
    ENGINE_SHARD_REQUESTS,
)

logger = logging.getLogger(__name__)

# Transport-level failures that indicate a dead/unreachable engine.
# ConnectionError and TimeoutError are OSError subclasses (py>=3.10).
_TRIP_EXCEPTIONS: tuple[type[BaseException], ...] = (OSError, EOFError)

_STATE_VALUES = {"closed": 0.0, "half_open": 1.0, "open": 2.0}


class EngineCircuitOpenError(ConnectionError):
    """Engine circuit breaker is open — fail fast instead of reconnecting."""


class CircuitBreaker:
    """Thread-safe closed → open → half-open circuit breaker.

    Reusable beyond the engine client: subclasses may override ``error_cls``
    (the typed fail-fast exception), ``subject`` (log/message wording), and
    ``_export_state`` (which gauge carries the state) — the MCP multiplexer's
    per-child breaker (CONCEPT:ECO-4.34) does exactly that."""

    #: Raised on short-circuited calls; a ``ConnectionError`` subclass so
    #: existing ``except ConnectionError`` handlers keep working.
    error_cls: type[ConnectionError] = EngineCircuitOpenError
    #: Human wording for log lines and error messages.
    subject: str = "epistemic-graph engine"

    def __init__(
        self, endpoint: str, threshold: int = 5, cooldown: float = 15.0
    ) -> None:
        self.endpoint = endpoint
        self.threshold = int(threshold)
        self.cooldown = float(cooldown)
        self._lock = threading.Lock()
        self._failures = 0
        self._state = "closed"
        self._opened_at = 0.0
        self._probe_in_flight = False
        self._export_state()

    # ------------------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    @property
    def enabled(self) -> bool:
        return self.threshold > 0

    def _state_value(self) -> float:
        """Numeric gauge encoding of the current state (0/1/2)."""
        return _STATE_VALUES[self._state]

    def _export_state(self) -> None:
        ENGINE_BREAKER_STATE.labels(endpoint=self.endpoint).set(self._state_value())

    def _set_state(self, state: str) -> None:
        if state == self._state:
            return
        log = logger.warning if state == "open" else logger.info
        log(
            "%s circuit breaker %s -> %s (endpoint=%s, failures=%d). (CONCEPT:OS-5.23)",
            self.subject,
            self._state,
            state,
            self.endpoint,
            self._failures,
        )
        self._state = state
        self._export_state()

    # ------------------------------------------------------------------
    def before_call(self) -> None:
        """Gate a call: raise :class:`EngineCircuitOpenError` when open."""
        if not self.enabled:
            return
        with self._lock:
            if self._state == "closed":
                return
            now = time.monotonic()
            if self._state == "open":
                remaining = self._opened_at + self.cooldown - now
                if remaining > 0:
                    raise self.error_cls(
                        f"{self.subject} breaker OPEN for "
                        f"{self.endpoint!r} after {self._failures} consecutive "
                        f"connection failures; retrying in {remaining:.1f}s."
                    )
                self._set_state("half_open")
                self._probe_in_flight = True
                return
            # half_open: exactly one probe at a time
            if self._probe_in_flight:
                raise self.error_cls(
                    f"{self.subject} breaker HALF-OPEN for "
                    f"{self.endpoint!r}: probe already in flight."
                )
            self._probe_in_flight = True

    def record_success(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._failures = 0
            self._probe_in_flight = False
            self._set_state("closed")

    def record_failure(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._failures += 1
            was_probe = self._probe_in_flight
            self._probe_in_flight = False
            if was_probe or self._failures >= self.threshold:
                self._opened_at = time.monotonic()
                self._set_state("open")


# ---------------------------------------------------------------------------
# Per-endpoint registry — ONE shared breaker per engine endpoint.
# ---------------------------------------------------------------------------

_breakers: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_breaker(endpoint: str) -> CircuitBreaker:
    """Return the process-wide shared breaker for ``endpoint``."""
    with _registry_lock:
        breaker = _breakers.get(endpoint)
        if breaker is None:
            from agent_utilities.core.config import config

            breaker = _breakers[endpoint] = CircuitBreaker(
                endpoint,
                threshold=config.engine_breaker_threshold,
                cooldown=config.engine_breaker_cooldown,
            )
        return breaker


def reset_breakers() -> None:
    """Drop all registered breakers (test isolation)."""
    with _registry_lock:
        _breakers.clear()


# ---------------------------------------------------------------------------
# Transparent client proxy — guards every callable attribute with the breaker.
# ---------------------------------------------------------------------------

_PASSTHROUGH_TYPES = (str, bytes, int, float, bool, dict, list, tuple, set)


def _record_outcome(breaker: CircuitBreaker, op: str, outcome: str) -> None:
    """Count one engine call by op AND by shard endpoint (CONCEPT:OS-5.28)."""
    ENGINE_REQUESTS.labels(op=op, outcome=outcome).inc()
    ENGINE_SHARD_REQUESTS.labels(endpoint=breaker.endpoint, outcome=outcome).inc()


# Adaptive transient-retry (CONCEPT:KG-2.262). A single dropped connection
# (``ConnectionReset``/``BrokenPipe`` mid-op) is TRANSIENT: the client transparently
# re-establishes the socket on its next call (``client._reconnect``). Without a retry
# here, that first failed call propagated AND counted toward the breaker — so a brief
# blip cascaded N callers into a tripped breaker (the failure mode that wedged whole
# ingest/finalize runs). We RETRY the op a bounded number of times with backoff before
# counting a failure; the retry rides the client's reconnect, so the blip self-heals.
_MAX_TRANSIENT_RETRIES = 2
_RETRY_BACKOFF_BASE_S = 0.25


def _guard(fn: Any, breaker: CircuitBreaker, op: str) -> Any:
    def call(*args: Any, **kwargs: Any) -> Any:
        try:
            breaker.before_call()
        except EngineCircuitOpenError:
            _record_outcome(breaker, op, "short_circuited")
            raise
        for attempt in range(_MAX_TRANSIENT_RETRIES + 1):
            try:
                result = fn(*args, **kwargs)
            except _TRIP_EXCEPTIONS:
                if attempt < _MAX_TRANSIENT_RETRIES:
                    # transient drop — let the client reconnect on the retry, and do
                    # NOT count it against the breaker yet (adaptive, KG-2.262).
                    _record_outcome(breaker, op, "retry")
                    time.sleep(_RETRY_BACKOFF_BASE_S * (2**attempt))
                    continue
                breaker.record_failure()
                _record_outcome(breaker, op, "connection_error")
                raise
            except Exception:
                # Application-level error (bad query, missing node...): the engine
                # answered, so the circuit stays closed.
                _record_outcome(breaker, op, "error")
                raise
            breaker.record_success()
            _record_outcome(breaker, op, "ok")
            return result

    call.__name__ = getattr(fn, "__name__", op)
    call.__qualname__ = f"breaker_guard({op})"
    return call


class BreakerClientProxy:
    """Attribute-transparent proxy over an engine client (or sub-namespace).

    Callable attributes (``nodes.add``, ``graph.shortest_path``...) are
    wrapped with the breaker guard; namespace attributes (``nodes``,
    ``edges``...) are wrapped recursively so the dotted path becomes the
    bounded ``op`` metric label. The raw client stays reachable via
    ``proxy.__wrapped__`` for the rare call that must hand the real client
    across the wire (e.g. VF2 pattern matching).
    """

    def __init__(self, target: Any, breaker: CircuitBreaker, prefix: str = "") -> None:
        self.__wrapped__ = target
        self._breaker = breaker
        self._prefix = prefix

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.__wrapped__, name)
        op = f"{self._prefix}.{name}" if self._prefix else name
        if callable(attr):
            return _guard(attr, self._breaker, op)
        if name.startswith("_") or isinstance(attr, _PASSTHROUGH_TYPES) or attr is None:
            return attr
        return BreakerClientProxy(attr, self._breaker, op)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"BreakerClientProxy({self.__wrapped__!r}, state={self._breaker.state})"


def wrap_client_with_breaker(client: Any, breaker: CircuitBreaker) -> Any:
    """Wrap an engine client so every call is breaker-guarded + metered."""
    return BreakerClientProxy(client, breaker)


def unwrap_client(client: Any) -> Any:
    """Return the raw client beneath a :class:`BreakerClientProxy` (or as-is)."""
    return getattr(client, "__wrapped__", client)


__all__ = [
    "BreakerClientProxy",
    "CircuitBreaker",
    "EngineCircuitOpenError",
    "get_breaker",
    "reset_breakers",
    "unwrap_client",
    "wrap_client_with_breaker",
]
