"""Per-agent-type circuit breaker in the parallel engine (CONCEPT:AU-ORCH.execution.parallel-engine-visualizer).

The engine's forked breaker was collapsed onto the canonical OS-5.23
``engine_breaker.CircuitBreaker`` state machine via the same
subclass-parameterization the multiplexer's per-child breaker uses
(CONCEPT:AU-ECO.mcp.profile-differences-from-client). These tests pin the preserved semantics: open after
``threshold`` consecutive failures, stay open across waves (infinite
cooldown — no half-open probe), close again on a recorded success.
"""

from __future__ import annotations

from agent_utilities.graph.parallel_engine import (
    AgentBreakerOpenError,
    AgentTypeCircuitBreaker,
    ParallelEngine,
)
from agent_utilities.knowledge_graph.core.engine_breaker import CircuitBreaker


def test_is_canonical_breaker_subclass():
    assert issubclass(AgentTypeCircuitBreaker, CircuitBreaker)
    assert issubclass(AgentBreakerOpenError, ConnectionError)


def test_opens_after_threshold_consecutive_failures():
    breaker = AgentTypeCircuitBreaker("researcher", threshold=3)
    for _ in range(2):
        breaker.record_failure()
    breaker.before_call()  # still closed below threshold
    breaker.record_failure()
    assert breaker.state == "open"
    try:
        breaker.before_call()
        raise AssertionError("expected AgentBreakerOpenError")
    except AgentBreakerOpenError:
        pass


def test_stays_open_without_probe_until_success():
    breaker = AgentTypeCircuitBreaker("researcher", threshold=1)
    breaker.record_failure()
    assert breaker.state == "open"
    # Infinite cooldown: never transitions to half-open by itself.
    for _ in range(3):
        try:
            breaker.before_call()
            raise AssertionError("expected AgentBreakerOpenError")
        except AgentBreakerOpenError:
            pass
    breaker.record_success()
    assert breaker.state == "closed"
    breaker.before_call()  # closed again — calls pass


def test_success_resets_consecutive_failure_count():
    breaker = AgentTypeCircuitBreaker("writer", threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "closed"  # never hit 3 consecutive


def test_threshold_zero_disables_breaker():
    # Canonical convention: 0 = breaker off (the deleted fork treated 0 as
    # "always open" — a footgun deliberately not preserved).
    breaker = AgentTypeCircuitBreaker("auditor", threshold=0)
    for _ in range(5):
        breaker.record_failure()
    breaker.before_call()  # never opens


def test_parallel_engine_keeps_one_breaker_per_agent_type():
    engine = ParallelEngine(engine=None)
    a = engine._agent_breaker("agent-a")
    b = engine._agent_breaker("agent-b")
    assert engine._agent_breaker("agent-a") is a
    assert a is not b
    assert a.endpoint == "agent-a"
    assert a.threshold == engine._breaker_threshold
