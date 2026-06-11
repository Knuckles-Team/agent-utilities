"""Tests for the epistemic-graph engine circuit breaker (CONCEPT:OS-5.23).

Covers:
- closed → open after N consecutive connect/timeout failures
- open → fast, typed EngineCircuitOpenError (a ConnectionError subclass)
- half-open probe after cooldown (success closes, failure re-opens,
  concurrent calls during the probe are short-circuited)
- threshold=0 disables tripping
- BreakerClientProxy: transparent wrapping of a fake client, trip vs
  application-error classification, op-label naming, unwrap
- per-endpoint shared registry
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core import engine_breaker
from agent_utilities.knowledge_graph.core.engine_breaker import (
    CircuitBreaker,
    EngineCircuitOpenError,
    get_breaker,
    reset_breakers,
    unwrap_client,
    wrap_client_with_breaker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordingMetric:
    """Counter/gauge stand-in capturing every labels(...).inc()/set() call."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict, str, float]] = []
        self._labels: dict = {}

    def labels(self, **kwargs):
        clone = RecordingMetric()
        clone.calls = self.calls
        clone._labels = kwargs
        return clone

    def inc(self, amount: float = 1.0) -> None:
        self.calls.append((self._labels, "inc", amount))

    def set(self, value: float) -> None:
        self.calls.append((self._labels, "set", value))


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_breakers()
    yield
    reset_breakers()


@pytest.fixture
def clock(monkeypatch):
    clk = FakeClock()
    monkeypatch.setattr(engine_breaker.time, "monotonic", clk)
    return clk


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class TestCircuitBreakerStates:
    def test_starts_closed_and_allows_calls(self):
        br = CircuitBreaker("unix:///tmp/x.sock", threshold=3, cooldown=10)
        assert br.state == "closed"
        br.before_call()  # no raise

    def test_opens_after_threshold_consecutive_failures(self, clock):
        br = CircuitBreaker("ep", threshold=3, cooldown=10)
        br.record_failure()
        br.record_failure()
        assert br.state == "closed"
        br.record_failure()
        assert br.state == "open"

    def test_open_raises_typed_connection_error(self, clock):
        br = CircuitBreaker("ep", threshold=1, cooldown=10)
        br.record_failure()
        with pytest.raises(EngineCircuitOpenError) as exc:
            br.before_call()
        assert isinstance(exc.value, ConnectionError)
        assert "ep" in str(exc.value)

    def test_success_resets_failure_streak(self, clock):
        br = CircuitBreaker("ep", threshold=3, cooldown=10)
        br.record_failure()
        br.record_failure()
        br.record_success()
        br.record_failure()
        br.record_failure()
        assert br.state == "closed"  # streak restarted — not cumulative

    def test_half_open_probe_after_cooldown_success_closes(self, clock):
        br = CircuitBreaker("ep", threshold=1, cooldown=10)
        br.record_failure()
        assert br.state == "open"
        clock.advance(10.1)
        br.before_call()  # the probe is admitted
        assert br.state == "half_open"
        br.record_success()
        assert br.state == "closed"
        br.before_call()  # closed again — no raise

    def test_half_open_probe_failure_reopens(self, clock):
        br = CircuitBreaker("ep", threshold=2, cooldown=10)
        br.record_failure()
        br.record_failure()
        clock.advance(10.1)
        br.before_call()
        assert br.state == "half_open"
        br.record_failure()  # single probe failure re-opens immediately
        assert br.state == "open"
        with pytest.raises(EngineCircuitOpenError):
            br.before_call()

    def test_only_one_probe_admitted_in_half_open(self, clock):
        br = CircuitBreaker("ep", threshold=1, cooldown=10)
        br.record_failure()
        clock.advance(10.1)
        br.before_call()  # probe 1
        with pytest.raises(EngineCircuitOpenError):
            br.before_call()  # probe 2 rejected while probe 1 in flight

    def test_open_still_raises_before_cooldown(self, clock):
        br = CircuitBreaker("ep", threshold=1, cooldown=10)
        br.record_failure()
        clock.advance(5)
        with pytest.raises(EngineCircuitOpenError):
            br.before_call()

    def test_threshold_zero_disables(self, clock):
        br = CircuitBreaker("ep", threshold=0, cooldown=10)
        for _ in range(50):
            br.record_failure()
        assert br.state == "closed"
        br.before_call()  # never raises


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_shared_instance_per_endpoint(self):
        assert get_breaker("a") is get_breaker("a")
        assert get_breaker("a") is not get_breaker("b")

    def test_defaults_come_from_config(self):
        br = get_breaker("cfg-endpoint")
        assert br.threshold == 5
        assert br.cooldown == 15.0

    def test_reset_breakers_clears(self):
        first = get_breaker("a")
        reset_breakers()
        assert get_breaker("a") is not first


# ---------------------------------------------------------------------------
# Client proxy
# ---------------------------------------------------------------------------


class FakeNamespace:
    def __init__(self, fail_with: Exception | None = None):
        self.fail_with = fail_with
        self.calls: list[tuple] = []

    def add(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.fail_with is not None:
            raise self.fail_with
        return "added"


class FakeClient:
    version = "1.0"  # plain attribute must pass through untouched

    def __init__(self, fail_with: Exception | None = None):
        self.nodes = FakeNamespace(fail_with)

    def supports(self, name):
        return name == "ParseFiles"


class TestBreakerClientProxy:
    def test_transparent_success(self):
        br = CircuitBreaker("ep", threshold=2, cooldown=10)
        proxy = wrap_client_with_breaker(FakeClient(), br)
        assert proxy.nodes.add("n1", {"k": 1}) == "added"
        assert proxy.supports("ParseFiles") is True
        assert proxy.version == "1.0"
        assert br.state == "closed"

    def test_connection_errors_trip_breaker(self, clock):
        br = CircuitBreaker("ep", threshold=2, cooldown=10)
        proxy = wrap_client_with_breaker(
            FakeClient(fail_with=ConnectionRefusedError("down")), br
        )
        with pytest.raises(ConnectionRefusedError):
            proxy.nodes.add("n1")
        with pytest.raises(ConnectionRefusedError):
            proxy.nodes.add("n1")
        assert br.state == "open"
        # open: the underlying client is NOT called again (fast fail)
        client = unwrap_client(proxy)
        before = len(client.nodes.calls)
        with pytest.raises(EngineCircuitOpenError):
            proxy.nodes.add("n1")
        assert len(client.nodes.calls) == before

    def test_application_errors_do_not_trip(self):
        br = CircuitBreaker("ep", threshold=1, cooldown=10)
        proxy = wrap_client_with_breaker(
            FakeClient(fail_with=ValueError("bad query")), br
        )
        for _ in range(5):
            with pytest.raises(ValueError, match="bad query"):
                proxy.nodes.add("n1")
        assert br.state == "closed"

    def test_outcome_metrics_and_op_labels(self, monkeypatch):
        fake = RecordingMetric()
        monkeypatch.setattr(engine_breaker, "ENGINE_REQUESTS", fake)
        br = CircuitBreaker("ep", threshold=1, cooldown=10)
        proxy = wrap_client_with_breaker(FakeClient(), br)
        proxy.nodes.add("n1")
        assert ({"op": "nodes.add", "outcome": "ok"}, "inc", 1.0) in fake.calls

        failing = wrap_client_with_breaker(
            FakeClient(fail_with=BrokenPipeError("x")),
            CircuitBreaker("ep2", threshold=1, cooldown=10),
        )
        with pytest.raises(BrokenPipeError):
            failing.nodes.add("n1")
        assert (
            {"op": "nodes.add", "outcome": "connection_error"},
            "inc",
            1.0,
        ) in fake.calls
        with pytest.raises(EngineCircuitOpenError):
            failing.nodes.add("n1")
        assert (
            {"op": "nodes.add", "outcome": "short_circuited"},
            "inc",
            1.0,
        ) in fake.calls

    def test_unwrap_client(self):
        br = CircuitBreaker("ep", threshold=1, cooldown=10)
        raw = FakeClient()
        proxy = wrap_client_with_breaker(raw, br)
        assert unwrap_client(proxy) is raw
        assert unwrap_client(raw) is raw

    def test_breaker_state_gauge_exported(self, monkeypatch):
        fake = RecordingMetric()
        monkeypatch.setattr(engine_breaker, "ENGINE_BREAKER_STATE", fake)
        br = CircuitBreaker("gauge-ep", threshold=1, cooldown=10)
        assert ({"endpoint": "gauge-ep"}, "set", 0.0) in fake.calls
        br.record_failure()
        assert ({"endpoint": "gauge-ep"}, "set", 2.0) in fake.calls
