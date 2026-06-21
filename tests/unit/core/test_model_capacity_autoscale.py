"""Tests for the adaptive per-model concurrency controller (CONCEPT:KG-2.145).

These never hit the network: a fake metrics fetcher is injected so the AIMD logic,
bounds, and fail-safe paths are exercised deterministically.
"""

from __future__ import annotations

import pytest

from agent_utilities.core.model_capacity_autoscale import (
    AdaptiveCapacityController,
    metrics_url_from_base,
    parse_vllm_gauge,
    reset_adaptive_controllers,
)


@pytest.fixture(autouse=True)
def _isolate():
    reset_adaptive_controllers()
    yield
    reset_adaptive_controllers()


def _metrics(
    running: float, waiting_capacity: float, model_name: str = "bge-m3"
) -> str:
    return (
        f'vllm:num_requests_running{{model_name="{model_name}"}} {running}\n'
        f'vllm:num_requests_waiting_by_reason{{model_name="{model_name}",'
        f'reason="capacity"}} {waiting_capacity}\n'
        f'vllm:num_requests_waiting{{model_name="{model_name}"}} {waiting_capacity}\n'
    )


def _controller(
    fetcher, *, floor: int = 4, ceiling: int = 512, model_name: str = "bge-m3"
) -> AdaptiveCapacityController:
    return AdaptiveCapacityController(
        model_key="embedding",
        model_name=model_name,
        metrics_url="http://vllm-embed.arpa/metrics",
        floor=floor,
        ceiling=ceiling,
        fetcher=fetcher,
        min_poll_interval_s=0.0,  # poll every resolve() for deterministic ramping
    )


# --- metrics URL derivation -------------------------------------------------


def test_metrics_url_drops_v1_and_appends_metrics():
    assert metrics_url_from_base("http://vllm-embed.arpa/v1") == (
        "http://vllm-embed.arpa/metrics"
    )
    assert metrics_url_from_base("http://vllm.arpa") == "http://vllm.arpa/metrics"
    assert metrics_url_from_base("http://vllm.arpa/v1/") == "http://vllm.arpa/metrics"


# --- gauge parsing ----------------------------------------------------------


def test_parse_gauge_filters_by_model_and_reason():
    text = _metrics(running=3, waiting_capacity=2)
    assert parse_vllm_gauge(
        text, "vllm:num_requests_running", model_name="bge-m3"
    ) == pytest.approx(3.0)
    assert parse_vllm_gauge(
        text,
        "vllm:num_requests_waiting_by_reason",
        model_name="bge-m3",
        reason="capacity",
    ) == pytest.approx(2.0)
    # wrong model → absent
    assert (
        parse_vllm_gauge(text, "vllm:num_requests_running", model_name="other") is None
    )


def test_parse_gauge_absent_is_none_not_zero():
    assert parse_vllm_gauge("# nothing here\n", "vllm:num_requests_running") is None


# --- AIMD ramp UP -----------------------------------------------------------


def test_ramps_up_when_near_full_and_no_capacity_waiting():
    # running stays high relative to target, no capacity-waiting → additive increase.
    ctrl = _controller(lambda _url: _metrics(running=1000, waiting_capacity=0), floor=4)
    targets = [ctrl.resolve() for _ in range(6)]
    # strictly grows over polls toward the ceiling, never below floor.
    assert targets[0] >= 4
    assert targets[-1] > targets[0]
    assert all(b >= a for a, b in zip(targets, targets[1:], strict=False))
    assert targets[-1] <= ctrl.ceiling


# --- AIMD back OFF ----------------------------------------------------------


def test_backs_off_on_capacity_waiting_not_below_floor():
    # First ramp up a bit, then saturate → multiplicative decrease toward floor.
    state = {"running": 1000.0, "waiting": 0.0}

    def fetch(_url):
        return _metrics(running=state["running"], waiting_capacity=state["waiting"])

    ctrl = _controller(fetch, floor=4)
    for _ in range(5):
        ctrl.resolve()
    high = ctrl.current_target
    assert high > 4
    # Now saturated: capacity-waiting present, running drops to a sustainable level.
    state["running"] = 6.0
    state["waiting"] = 3.0
    for _ in range(10):
        ctrl.resolve()
    assert ctrl.current_target < high
    assert ctrl.current_target >= ctrl.floor  # never below floor


# --- fail-safe --------------------------------------------------------------


def test_failsafe_unreachable_metrics_returns_floor():
    def boom(_url):
        raise OSError("connection refused")

    ctrl = _controller(boom, floor=7)
    # ramp attempts can't read anything → pinned at floor, no exception.
    assert ctrl.resolve() == 7
    assert ctrl.resolve() == 7


def test_failsafe_garbage_metrics_returns_floor():
    ctrl = _controller(lambda _url: "not prometheus at all\n<<garbage>>", floor=5)
    assert ctrl.resolve() == 5


# --- bounds -----------------------------------------------------------------


def test_never_exceeds_ceiling():
    ctrl = _controller(
        lambda _url: _metrics(running=100000, waiting_capacity=0), floor=4, ceiling=16
    )
    for _ in range(50):
        ctrl.resolve()
    assert ctrl.current_target == 16


def test_idle_holds_target():
    # running far below 0.8*target, no waiting → hold (no change from floor).
    ctrl = _controller(lambda _url: _metrics(running=0, waiting_capacity=0), floor=4)
    targets = [ctrl.resolve() for _ in range(5)]
    assert targets == [4, 4, 4, 4, 4]


# --- utilization snapshot ---------------------------------------------------


def test_utilization_returns_parsed_gauges():
    ctrl = _controller(lambda _url: _metrics(running=5, waiting_capacity=2), floor=4)
    ctrl.resolve()
    snap = ctrl.utilization()
    assert snap["running"] == pytest.approx(5.0)
    assert snap["waiting_capacity"] == pytest.approx(2.0)
    assert snap["metrics_ok"] is True
    assert snap["saturated"] is True
    assert snap["floor"] == 4
    assert snap["current_target"] >= 4
    assert snap["metrics_url"] == "http://vllm-embed.arpa/metrics"


# --- integration via adaptive_capacity / config -----------------------------


def test_adaptive_capacity_flag_off_returns_floor(monkeypatch):
    from agent_utilities.core import model_capacity_autoscale as mod

    monkeypatch.setenv("KG_ADAPTIVE_CONCURRENCY", "0")
    # Even with a fetcher that would ramp, flag-off → static floor.
    cap = mod.adaptive_capacity(
        "embedding",
        4,
        fetcher=lambda _url: _metrics(running=1000, waiting_capacity=0),
    )
    assert cap == 4


def test_adaptive_capacity_no_endpoint_returns_floor(monkeypatch):
    from agent_utilities.core import model_capacity_autoscale as mod

    monkeypatch.setenv("KG_ADAPTIVE_CONCURRENCY", "1")

    class _NoEndpointConfig:
        def model_endpoint(self, _model):
            return ("m", None)  # no base_url → nothing to scrape

    monkeypatch.setattr(
        "agent_utilities.core.config.config", _NoEndpointConfig(), raising=False
    )
    cap = mod.adaptive_capacity(
        "embedding",
        9,
        fetcher=lambda _url: _metrics(running=1000, waiting_capacity=0),
    )
    assert cap == 9


def test_adaptive_capacity_ramps_via_config_endpoint(monkeypatch):
    from agent_utilities.core import model_capacity_autoscale as mod

    monkeypatch.setenv("KG_ADAPTIVE_CONCURRENCY", "1")
    monkeypatch.setenv("MODEL_MAX_CONCURRENCY", "512")

    class _Config:
        def model_endpoint(self, _model):
            return ("bge-m3", "http://vllm-embed.arpa/v1")

    monkeypatch.setattr("agent_utilities.core.config.config", _Config(), raising=False)

    fetcher = lambda _url: _metrics(running=1000, waiting_capacity=0)  # noqa: E731
    first = mod.adaptive_capacity("embedding", 4, fetcher=fetcher)
    # subsequent calls reuse the cached controller; min_poll uses the real default
    # interval, so we drive the controller directly to prove ramping is wired.
    ctrl = mod._get_controller("embedding", 4)
    assert ctrl is not None
    ctrl.min_poll_interval_s = 0.0
    for _ in range(5):
        ctrl.resolve()
    assert ctrl.current_target > first
    assert first >= 4


def test_get_utilization_shape(monkeypatch):
    from agent_utilities.core import model_capacity_autoscale as mod

    monkeypatch.setenv("KG_ADAPTIVE_CONCURRENCY", "1")

    class _Config:
        def model_endpoint(self, _model):
            return ("bge-m3", "http://vllm-embed.arpa/v1")

        def model_capacity(self, _model):
            return 4

    monkeypatch.setattr("agent_utilities.core.config.config", _Config(), raising=False)
    # seed a controller with a fake fetcher
    ctrl = mod._get_controller(
        "embedding", 4, fetcher=lambda _url: _metrics(running=3, waiting_capacity=0)
    )
    assert ctrl is not None
    ctrl.min_poll_interval_s = 0.0
    snap = mod.get_utilization("embedding")
    assert {
        "running",
        "waiting_capacity",
        "current_target",
        "last_poll",
        "floor",
        "ceiling",
    } <= set(snap)
    assert snap["adaptive"] is True
