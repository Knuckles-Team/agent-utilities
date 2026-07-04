"""Tests for the adaptive per-model concurrency controller (CONCEPT:AU-KG.compute.surfaces-universal-latency-signal).

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


# --- universal latency-gradient signal (NO /metrics) ------------------------


def _latency_controller(
    *, floor: int = 4, ceiling: int = 512
) -> AdaptiveCapacityController:
    """A controller with NO metrics endpoint — pure latency-gradient tuning."""
    return AdaptiveCapacityController(
        model_key="chat",
        model_name="some-openai-model",
        metrics_url=None,  # LM Studio / llama.cpp / OpenAI: no /metrics
        floor=floor,
        ceiling=ceiling,
        update_samples=1,  # re-tune on every sample for deterministic ramping
        update_interval_s=0.0,
        gradient_target=0.9,
    )


def test_latency_flat_low_ramps_up_no_metrics():
    # Flat, low latency (no inflation) → gradient ≈ 1 → additive increase, ramping
    # UP toward the ceiling, with NOTHING from the server.
    ctrl = _latency_controller(floor=4)
    for _ in range(40):
        ctrl.record_sample(latency_s=0.10, ok=True)
    assert ctrl.current_target > 4
    assert ctrl.resolve() <= ctrl.ceiling
    assert ctrl.utilization()["signal"] == "latency"


def test_latency_inflation_backs_off_not_below_floor():
    ctrl = _latency_controller(floor=4)
    # Establish a fast baseline + ramp up.
    for _ in range(20):
        ctrl.record_sample(latency_s=0.10, ok=True)
    high = ctrl.current_target
    assert high > 4
    # Now latency inflates badly (queueing) → gradient well below target → back off.
    for _ in range(60):
        ctrl.record_sample(latency_s=2.0, ok=True)
    assert ctrl.current_target < high
    assert ctrl.current_target >= ctrl.floor  # never below floor


def test_overload_status_immediate_backoff():
    ctrl = _latency_controller(floor=4)
    for _ in range(20):
        ctrl.record_sample(latency_s=0.10, ok=True)
    high = ctrl.current_target
    assert high > 4
    # A single 429 is a congestion event → immediate multiplicative decrease.
    ctrl.record_sample(latency_s=0.10, ok=False, status=429)
    assert ctrl.current_target < high
    # 503 likewise.
    mid = ctrl.current_target
    ctrl.record_sample(latency_s=0.10, ok=False, status=503)
    assert ctrl.current_target <= mid
    assert ctrl.current_target >= ctrl.floor


def test_baseline_tracks_minimum():
    ctrl = _latency_controller(floor=4)
    ctrl.record_sample(latency_s=0.5, ok=True)
    assert ctrl.baseline_latency == pytest.approx(0.5)
    # A faster sample snaps the baseline down toward it.
    ctrl.record_sample(latency_s=0.1, ok=True)
    assert ctrl.baseline_latency < 0.5
    # A slow sample barely moves it (doesn't chase inflation).
    before = ctrl.baseline_latency
    ctrl.record_sample(latency_s=5.0, ok=True)
    assert ctrl.baseline_latency < before * 1.5


def test_gradient_computation():
    ctrl = _latency_controller(floor=4)
    # baseline 0.1, avg ~0.1 → gradient ≈ 1
    for _ in range(10):
        ctrl.record_sample(latency_s=0.1, ok=True)
    assert ctrl.utilization()["gradient"] == pytest.approx(1.0, abs=0.05)
    # Push avg up to ~0.4 (baseline still ~0.1) → gradient ≈ 0.25-ish, well < 1.
    ctrl2 = _latency_controller(floor=4)
    ctrl2.record_sample(latency_s=0.1, ok=True)  # baseline
    for _ in range(20):
        ctrl2.record_sample(latency_s=0.4, ok=True)
    assert ctrl2.utilization()["gradient"] < 0.9


def test_few_samples_hold():
    ctrl = _latency_controller(floor=4)
    # Below _MIN_SAMPLES → hold at floor.
    ctrl.record_sample(latency_s=0.1, ok=True)
    ctrl.record_sample(latency_s=0.1, ok=True)
    assert ctrl.current_target == 4


def test_latency_respects_ceiling():
    ctrl = _latency_controller(floor=4, ceiling=10)
    for _ in range(200):
        ctrl.record_sample(latency_s=0.05, ok=True)
    assert ctrl.current_target == 10


# --- hybrid: /metrics capacity-waiting forces back-off ----------------------


def test_metrics_capacity_waiting_forces_backoff_even_when_latency_ok():
    # Latency looks great (would ramp up), but vLLM /metrics reports
    # capacity-waiting > 0 → hard saturation → forced back-off.
    ctrl = _controller(lambda _url: _metrics(running=2, waiting_capacity=5), floor=4)
    ctrl.min_poll_interval_s = 0.0
    # ramp up via latency first
    for _ in range(30):
        ctrl.record_sample(latency_s=0.05, ok=True)
    high = ctrl.current_target
    assert high > 4
    # resolve() polls /metrics → capacity-waiting forces a decrease.
    after = ctrl.resolve()
    assert after < high
    snap = ctrl.utilization()
    assert snap["signal"] == "hybrid"  # both signals present
    assert snap["saturated"] is True


def test_metrics_absent_falls_back_to_latency_only():
    # Endpoint with NO vLLM gauges (LM Studio etc.) → metrics_available False,
    # signal stays "latency".
    ctrl = AdaptiveCapacityController(
        model_key="chat",
        model_name="lmstudio-model",
        metrics_url="http://lmstudio.local/metrics",
        floor=4,
        ceiling=512,
        fetcher=lambda _url: "# HELP foo\nfoo 1\n",  # not vLLM
        update_samples=1,
        update_interval_s=0.0,
    )
    ctrl.min_poll_interval_s = 0.0
    for _ in range(30):
        ctrl.record_sample(latency_s=0.05, ok=True)
    ctrl.resolve()
    snap = ctrl.utilization()
    assert snap["metrics_available"] is False
    assert snap["signal"] == "latency"
    assert snap["current_target"] > 4  # latency still ramped it


# --- record_sample integration / flag-off -----------------------------------


def test_record_sample_module_fn_noop_when_flag_off(monkeypatch):
    from agent_utilities.core import model_capacity_autoscale as mod

    monkeypatch.setenv("KG_ADAPTIVE_CONCURRENCY", "0")
    # Must not raise and must not create/tune anything.
    mod.record_sample("embedding", latency_s=0.1, ok=True)
    mod.record_sample("embedding", latency_s=0.1, ok=False, status=429)


def test_utilization_surfaces_latency_fields():
    ctrl = _latency_controller(floor=4)
    for _ in range(10):
        ctrl.record_sample(latency_s=0.1, ok=True)
    ctrl.record_sample(latency_s=0.1, ok=False, status=503)
    snap = ctrl.utilization()
    assert {
        "baseline_latency",
        "recent_avg_latency",
        "gradient",
        "error_rate",
        "signal",
    } <= set(snap)
    assert snap["error_rate"] > 0
    assert snap["signal"] == "latency"


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
