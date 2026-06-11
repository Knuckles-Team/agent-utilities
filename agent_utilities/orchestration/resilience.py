"""Declarative Resilience Policy — retry / backoff / fallback / timeout.

CONCEPT:ORCH-1.36 — Declarative Resilience Policy

Closes the Reliability & Failure-Management gap (L7) versus the agentic
reference architecture. The platform already ships the canonical circuit
breaker (:class:`agent_utilities.knowledge_graph.core.engine_breaker.CircuitBreaker`),
a per-server breaker on the live specialist-execution path
(``ctx.deps.server_health`` in ``agent_utilities/graph/executor.py``),
and KG-persisted durable checkpoints
(:class:`agent_utilities.orchestration.durable_execution.DurableExecutionManager`),
but there was no *declarative* policy describing how an individual unit of work
should retry, back off, fall back, or time out.

This module provides that missing primitive as a single, composable object:

* :class:`ResiliencePolicy` — a declarative description of the retry/backoff/
  fallback/timeout behavior for one callable.
* :func:`compute_backoff` — deterministic exponential backoff with optional,
  *seedable* jitter (an injectable ``rng`` keeps tests reproducible; the helper
  never calls ``random`` or ``time`` in a way that breaks determinism).
* :func:`run_with_resilience` / :func:`run_with_resilience_sync` — execute a
  primary callable under a policy, honoring ``retry_on`` + backoff, then trying
  each fallback once, returning the first success or re-raising the last error.

It composes with — and never replaces — the circuit breaker and the per-attempt
timeout that already guard the live specialist-execution path in
``agent_utilities/graph/executor.py``.
"""

from __future__ import annotations

import asyncio
import logging
import random as _random_module
import time as _time_module
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:  # OpenTelemetry is optional — degrade to logging when absent.
    from opentelemetry import trace as _otel_trace

    _tracer = _otel_trace.get_tracer("agent-utilities.resilience")
except Exception:  # noqa: BLE001 - any import/runtime issue → no tracing
    _otel_trace = None
    _tracer = None


# Exceptions that are virtually always *deterministic* — retrying them just
# burns budget. ``ValueError``/``TypeError``/``KeyError`` signal a programming
# or input error; permission errors signal an authorization wall.
NON_RETRYABLE: tuple[type[Exception], ...] = (
    ValueError,
    TypeError,
    KeyError,
    PermissionError,
    NotImplementedError,
)

# Transient errors worth retrying by default.
DEFAULT_RETRYABLE: tuple[type[Exception], ...] = (
    TimeoutError,
    ConnectionError,
)

# Type alias for a retry predicate.
RetryPredicate = Callable[[BaseException], bool]


class RetryableError(Exception):
    """An error the raising site has already judged retryable.

    Carries an optional ``backoff_s`` delay override: when set, the resilience
    runners sleep exactly that long before the next attempt instead of the
    policy-computed backoff. This lets a call site encode a delay the policy
    cannot know (e.g. "schema was just healed — retry immediately", or a
    server-provided ``Retry-After``) without forking the retry loop.

    The policy still decides retryability: include the (sub)class in
    ``retry_on`` as usual.
    """

    def __init__(self, *args: Any, backoff_s: float | None = None) -> None:
        super().__init__(*args)
        self.backoff_s = backoff_s


@dataclass(frozen=True)
class ResiliencePolicy:
    """Declarative retry / backoff / fallback / timeout policy for one callable.

    Attributes:
        max_attempts: Number of times the *primary* callable is attempted
            (``>= 1``). ``1`` means "no retry".
        backoff_base_s: Base delay (seconds) for the first retry.
        backoff_factor: Exponential growth factor between retries (e.g. ``2.0``).
        max_backoff_s: Hard cap on any single backoff delay (seconds).
        backoff_strategy: ``"exponential"`` (default) grows the delay as
            ``base * factor ** (attempt - 1)``; ``"linear"`` grows it as
            ``base * attempt`` (``backoff_factor`` is ignored).
        jitter: When ``True``, apply ``jitter_strategy`` to the capped delay
            using an injectable RNG so it stays deterministic under test.
        jitter_strategy: ``"proportional"`` (default) multiplies the capped
            delay by a random factor in ``[0.5, 1.0]``; ``"additive"`` adds a
            random ``[0, backoff_base_s)`` offset on top of the capped delay.
        retry_on: Either a tuple of exception types to retry, or a predicate
            ``Callable[[BaseException], bool]``. Defaults to retrying transient
            errors (:data:`DEFAULT_RETRYABLE`) while *never* retrying the
            deterministic / authorization errors in :data:`NON_RETRYABLE`.
        timeout_s: Optional per-attempt timeout (seconds). ``None`` disables the
            policy-level timeout (the caller may still impose its own).
        fallbacks: Ordered list of callables tried once each, in order, after
            the primary attempts are exhausted.
        name: Human-readable label used in spans / logs.
    """

    max_attempts: int = 3
    backoff_base_s: float = 0.5
    backoff_factor: float = 2.0
    max_backoff_s: float = 30.0
    backoff_strategy: str = "exponential"
    jitter: bool = True
    jitter_strategy: str = "proportional"
    retry_on: tuple[type[Exception], ...] | RetryPredicate = DEFAULT_RETRYABLE
    timeout_s: float | None = None
    fallbacks: list[Callable[..., Any]] = field(default_factory=list)
    name: str = "resilience"

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(
                f"ResiliencePolicy.max_attempts must be >= 1, got {self.max_attempts}"
            )
        if self.backoff_base_s < 0 or self.max_backoff_s < 0:
            raise ValueError("backoff delays must be non-negative")
        if self.backoff_factor < 1.0:
            raise ValueError("backoff_factor must be >= 1.0")
        if self.backoff_strategy not in ("exponential", "linear"):
            raise ValueError(
                f"backoff_strategy must be 'exponential' or 'linear', "
                f"got {self.backoff_strategy!r}"
            )
        if self.jitter_strategy not in ("proportional", "additive"):
            raise ValueError(
                f"jitter_strategy must be 'proportional' or 'additive', "
                f"got {self.jitter_strategy!r}"
            )

    def should_retry(self, exc: BaseException) -> bool:
        """Decide whether ``exc`` is retryable under this policy.

        A predicate ``retry_on`` is consulted directly. A tuple ``retry_on`` is
        treated as an allow-list, but the deterministic / authorization errors
        in :data:`NON_RETRYABLE` are *always* rejected first (so a caller cannot
        accidentally retry a ``ValueError`` by passing ``retry_on=(Exception,)``).
        """
        retry_on = self.retry_on
        if callable(retry_on) and not isinstance(retry_on, tuple):
            try:
                return bool(retry_on(exc))
            except Exception:  # noqa: BLE001 - a broken predicate must not crash
                logger.warning(
                    "[CONCEPT:ORCH-1.36] retry_on predicate raised; treating as non-retryable"
                )
                return False
        # Tuple allow-list: hard-deny the known-deterministic family first.
        if isinstance(exc, NON_RETRYABLE):
            return False
        return isinstance(exc, tuple(retry_on))


#: Sensible default applied on the live specialist-execution path.
DEFAULT_POLICY = ResiliencePolicy(
    max_attempts=3,
    backoff_base_s=0.5,
    backoff_factor=2.0,
    max_backoff_s=10.0,
    jitter=True,
    retry_on=DEFAULT_RETRYABLE,
    timeout_s=None,
    fallbacks=[],
    name="default",
)


def compute_backoff(
    attempt: int,
    policy: ResiliencePolicy,
    rng: _random_module.Random | None = None,
) -> float:
    """Return the backoff delay (seconds) to wait *before* ``attempt``.

    ``attempt`` is 1-indexed: the delay before the 1st retry (i.e. between
    attempt 1 and attempt 2) is ``compute_backoff(1, ...)`` and equals
    ``backoff_base_s`` (factor**0). The delay grows as
    ``backoff_base_s * backoff_factor ** (attempt - 1)`` (exponential, the
    default) or ``backoff_base_s * attempt`` (``backoff_strategy="linear"``)
    and is capped at ``max_backoff_s``.

    When ``policy.jitter`` is ``True`` the (capped) delay is jittered per
    ``policy.jitter_strategy`` — multiplied by a factor in ``[0.5, 1.0]``
    (proportional, the default) or increased by ``[0, backoff_base_s)``
    (additive) — drawn from ``rng``, an injectable :class:`random.Random` so
    tests are deterministic. With ``jitter`` off the result is exact and
    independent of any RNG.
    """
    if attempt < 1:
        raise ValueError(f"attempt must be >= 1, got {attempt}")
    if policy.backoff_strategy == "linear":
        raw = policy.backoff_base_s * attempt
    else:
        raw = policy.backoff_base_s * (policy.backoff_factor ** (attempt - 1))
    capped = min(raw, policy.max_backoff_s)
    if not policy.jitter:
        return capped
    _rng = rng if rng is not None else _random_module.Random()  # noqa: S311 # nosec B311 - jitter only, not security
    if policy.jitter_strategy == "additive":
        return capped + _rng.random() * policy.backoff_base_s
    return capped * (0.5 + 0.5 * _rng.random())


def _emit_event(name: str, attributes: dict[str, Any]) -> None:
    """Emit an OTel span event when tracing is active; always log."""
    if _tracer is not None:
        span = _otel_trace.get_current_span()
        if span is not None:
            try:
                span.add_event(name, attributes=attributes)
            except Exception:  # noqa: BLE001 - tracing must never break execution
                pass


async def run_with_resilience(
    primary: Callable[..., Awaitable[Any] | Any],
    policy: ResiliencePolicy,
    *args: Any,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    rng: _random_module.Random | None = None,
    **kwargs: Any,
) -> Any:
    """Run ``primary`` under ``policy`` (async).

    Attempts ``primary`` up to ``policy.max_attempts`` times, retrying only when
    :meth:`ResiliencePolicy.should_retry` accepts the raised exception and
    sleeping ``compute_backoff(...)`` between attempts (via the injectable
    ``sleep`` coroutine so tests never actually wait). Each attempt is bounded by
    ``policy.timeout_s`` when set. After the primary attempts are exhausted, each
    callable in ``policy.fallbacks`` is tried once in order; the first success is
    returned. If everything fails, the last exception is raised.

    ``primary`` and fallbacks may be sync or async; coroutine results are
    awaited transparently.
    """
    last_exc: BaseException | None = None

    async def _invoke(fn: Callable[..., Any]) -> Any:
        if policy.timeout_s is not None:
            return await asyncio.wait_for(
                _maybe_await(fn(*args, **kwargs)), timeout=policy.timeout_s
            )
        return await _maybe_await(fn(*args, **kwargs))

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await _invoke(primary)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - policy decides retryability
            last_exc = exc
            retryable = policy.should_retry(exc)
            has_more = attempt < policy.max_attempts
            _emit_event(
                "resilience.attempt_failed",
                {
                    "policy": policy.name,
                    "attempt": attempt,
                    "max_attempts": policy.max_attempts,
                    "error": repr(exc),
                    "retryable": retryable,
                },
            )
            if not retryable or not has_more:
                logger.debug(
                    "[CONCEPT:ORCH-1.36] '%s' attempt %d/%d failed (retryable=%s): %s",
                    policy.name,
                    attempt,
                    policy.max_attempts,
                    retryable,
                    exc,
                )
                break
            delay = _retry_delay(exc, attempt, policy, rng)
            logger.info(
                "[CONCEPT:ORCH-1.36] '%s' retrying after attempt %d/%d "
                "(backoff=%.3fs): %s",
                policy.name,
                attempt,
                policy.max_attempts,
                delay,
                exc,
            )
            if delay > 0:
                await sleep(delay)

    # Primary exhausted — try fallbacks in order, once each.
    for idx, fb in enumerate(policy.fallbacks):
        _emit_event(
            "resilience.fallback",
            {"policy": policy.name, "fallback_index": idx},
        )
        logger.info(
            "[CONCEPT:ORCH-1.36] '%s' primary exhausted; trying fallback %d/%d",
            policy.name,
            idx + 1,
            len(policy.fallbacks),
        )
        try:
            return await _invoke(fb)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - try the next fallback
            last_exc = exc
            logger.warning(
                "[CONCEPT:ORCH-1.36] '%s' fallback %d/%d failed: %s",
                policy.name,
                idx + 1,
                len(policy.fallbacks),
                exc,
            )

    assert last_exc is not None  # max_attempts >= 1 guarantees at least one try
    raise last_exc


def run_with_resilience_sync(
    primary: Callable[..., Any],
    policy: ResiliencePolicy,
    *args: Any,
    sleep: Callable[[float], Any] = _time_module.sleep,
    rng: _random_module.Random | None = None,
    **kwargs: Any,
) -> Any:
    """Synchronous counterpart of :func:`run_with_resilience`.

    Honors ``retry_on`` + backoff (sleeping via the injectable ``sleep``), then
    tries each fallback once, returning the first success or re-raising the last
    exception. ``policy.timeout_s`` is *not* enforceable for arbitrary blocking
    sync callables without threads, so it is treated as advisory here (the async
    variant enforces it); the retry/backoff/fallback semantics are identical.
    """
    last_exc: BaseException | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return primary(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - policy decides retryability
            last_exc = exc
            retryable = policy.should_retry(exc)
            has_more = attempt < policy.max_attempts
            _emit_event(
                "resilience.attempt_failed",
                {
                    "policy": policy.name,
                    "attempt": attempt,
                    "max_attempts": policy.max_attempts,
                    "error": repr(exc),
                    "retryable": retryable,
                },
            )
            if not retryable or not has_more:
                break
            delay = _retry_delay(exc, attempt, policy, rng)
            logger.info(
                "[CONCEPT:ORCH-1.36] '%s' (sync) retrying after attempt %d/%d "
                "(backoff=%.3fs): %s",
                policy.name,
                attempt,
                policy.max_attempts,
                delay,
                exc,
            )
            if delay > 0:
                sleep(delay)

    for idx, fb in enumerate(policy.fallbacks):
        _emit_event(
            "resilience.fallback",
            {"policy": policy.name, "fallback_index": idx},
        )
        try:
            return fb(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - try the next fallback
            last_exc = exc
            logger.warning(
                "[CONCEPT:ORCH-1.36] '%s' (sync) fallback %d/%d failed: %s",
                policy.name,
                idx + 1,
                len(policy.fallbacks),
                exc,
            )

    assert last_exc is not None
    raise last_exc


async def _maybe_await(value: Awaitable[Any] | Any) -> Any:
    """Await ``value`` if it is awaitable, else return it as-is."""
    if asyncio.iscoroutine(value) or isinstance(value, Awaitable):
        return await value
    return value


def _retry_delay(
    exc: BaseException,
    attempt: int,
    policy: ResiliencePolicy,
    rng: _random_module.Random | None,
) -> float:
    """Delay before the next attempt: the exception's hint, else the policy's."""
    if isinstance(exc, RetryableError) and exc.backoff_s is not None:
        return max(0.0, exc.backoff_s)
    return compute_backoff(attempt, policy, rng=rng)
