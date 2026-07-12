"""Regression tests for the shared health-intelligence kernels
(``agent_utilities.observability.health``) — the metric-agnostic generalization of
fan-manager's proven thermal kernels (``fan_manager.kg_control``). Mirrors
``agents/fan-manager/tests/test_kg_control.py``'s test patterns, generalized off °C.
"""

from __future__ import annotations

from agent_utilities.observability import health as h


def _trend(avg, control, at, *, span=5):
    return {
        "avg": avg,
        "min": avg - span,
        "max": avg + span,
        "avg_control": control,
        "samples": 150,
        "window_s": 3600,
        "observed_at": at,
    }


# --- numeric helpers --------------------------------------------------------- #
def test_percentile_interpolates():
    assert h._percentile([50, 52, 54, 56, 58, 60, 62, 64], 50) == 57.0
    assert h._percentile([10], 95) == 10.0
    assert h._percentile([], 50) is None


def test_slope_degenerate_cases():
    assert h._slope([1, 2], [1, 2]) is None  # fewer than 3 points
    assert h._slope([1, 1, 1], [5, 6, 7]) is None  # zero variance in x


# --- compute_baseline --------------------------------------------------------- #
def test_compute_baseline_distills_distribution_and_inertia():
    trends = [
        _trend(t, f, f"2026-07-{d:02d}T00:00:00Z")
        for d, (t, f) in enumerate(
            [(50, 10), (52, 10), (54, 15), (56, 15), (58, 20), (60, 20), (62, 25), (64, 25)],
            1,
        )
    ]
    b = h.compute_baseline(
        trends, value_key="avg", peak_key="max", control_key="avg_control"
    )
    assert b is not None
    assert b["p50"] == 57.0
    assert b["p95"] == 68.3  # p95 of the max tail (avg+5)
    assert b["min_env"] == 50.0 and b["max_env"] == 64.0
    assert b["avg_control"] == 17.5 and b["windows"] == 8
    assert b["inertia"] is not None and b["inertia"] > 0  # control varied


def test_compute_baseline_insufficient_history_is_none():
    trends = [_trend(55, 10, f"2026-07-0{d}T00:00:00Z") for d in range(1, 4)]  # 3 < default 6
    assert h.compute_baseline(trends, value_key="avg") is None
    assert h.compute_baseline(trends, value_key="avg", min_windows=3) is not None


def test_compute_baseline_no_control_key_has_no_inertia():
    trends = [
        _trend(t, None, f"2026-07-{d:02d}T00:00:00Z")
        for d, t in enumerate([50, 52, 54, 56, 58, 60], 1)
    ]
    b = h.compute_baseline(trends, value_key="avg")
    assert b is not None
    assert b["inertia"] is None and b["avg_control"] is None


def test_compute_baseline_falls_back_to_value_key_without_peak_key():
    trends = [
        _trend(t, None, f"2026-07-{d:02d}T00:00:00Z")
        for d, t in enumerate([50, 52, 54, 56, 58, 60], 1)
    ]
    b = h.compute_baseline(trends, value_key="avg")  # no peak_key
    assert b is not None
    assert b["p95"] == h._percentile([50, 52, 54, 56, 58, 60], 95)


# --- detect_anomaly ------------------------------------------------------------ #
_BASE = {"p50": 55.0, "p95": 62.0, "min_env": 45.0, "max_env": 60.0, "inertia": 0.4, "windows": 300}


def test_detect_anomaly_above_baseline():
    recent = [_trend(t, 20, f"2026-07-14T0{i}:00:00Z") for i, t in enumerate([79, 80, 81])]
    a = h.detect_anomaly(recent, _BASE, value_key="avg", control_key="avg_control")
    assert a is not None
    assert a["kind"] == "above-baseline" and a["observed"] == 80.0 and a["zscore"] >= 3.0


def test_detect_anomaly_none_when_normal():
    recent = [_trend(t, 20, f"2026-07-14T0{i}:00:00Z") for i, t in enumerate([56, 57, 58])]
    assert h.detect_anomaly(recent, _BASE, value_key="avg", control_key="avg_control") is None


def test_detect_anomaly_saturated():
    recent = [_trend(61, c, f"2026-07-14T0{i}:00:00Z") for i, c in enumerate([96, 97, 98])]
    a = h.detect_anomaly(recent, _BASE, value_key="avg", control_key="avg_control")
    assert a is not None
    assert a["kind"] == "saturated"  # control pinned yet still above the load envelope


def test_detect_anomaly_none_without_baseline_or_recent():
    assert h.detect_anomaly([], _BASE, value_key="avg") is None
    assert h.detect_anomaly([_trend(80, 20, "2026-07-14T00:00:00Z")], None, value_key="avg") is None


def test_detect_anomaly_without_control_key_never_saturates():
    recent = [_trend(61, 99, f"2026-07-14T0{i}:00:00Z") for i in range(3)]
    # no control_key given -> saturated branch can never trigger, only above-baseline
    assert h.detect_anomaly(recent, _BASE, value_key="avg") is None


# --- correlate ------------------------------------------------------------------ #
def test_correlate_collapses_simultaneous_spikes():
    anoms = {
        "a": {"kind": "above-baseline"},
        "b": {"kind": "above-baseline"},
        "c": None,
        "d": {"kind": "saturated"},
    }
    h.correlate(anoms, total=4)
    a, b, d = anoms["a"], anoms["b"], anoms["d"]
    assert a is not None and b is not None and d is not None
    assert a["kind"] == "systemic" and b["kind"] == "systemic"
    assert d["kind"] == "saturated"  # different fault, not retagged


def test_correlate_leaves_a_lone_spike_alone():
    anoms = {"a": {"kind": "above-baseline"}, "b": None, "c": None, "d": None}
    h.correlate(anoms, total=4)
    a = anoms["a"]
    assert a is not None and a["kind"] == "above-baseline"  # 1 of 4 -> not systemic


def test_correlate_custom_kind_and_systemic_label():
    anoms = {"a": {"kind": "saturated"}, "b": {"kind": "saturated"}, "c": None}
    h.correlate(anoms, total=3, kind="saturated", systemic_kind="rack-power-event")
    assert anoms["a"]["kind"] == "rack-power-event"
    assert anoms["b"]["kind"] == "rack-power-event"


# --- HealthTrendBuffer ----------------------------------------------------------- #
def test_health_trend_buffer_distills_not_per_sample():
    buf = h.HealthTrendBuffer(window_s=3600, max_samples=5000)
    t0 = 1_000_000.0
    # samples within the window never flush
    for i in range(10):
        out = buf.add(50.0 + i, control=20.0, at=t0 + i)
        assert out is None
    # a sample after the window elapses triggers ONE distilled flush
    out = buf.add(70.0, control=25.0, at=t0 + 3601)
    assert out is not None
    assert out["samples"] == 11
    assert out["min"] == 50.0 and out["max"] == 70.0
    assert out["avg_control"] is not None
    assert out["window_s"] == 3600
    # buffer resets after flush
    assert buf._buf == []


def test_health_trend_buffer_flushes_on_sample_cap():
    buf = h.HealthTrendBuffer(window_s=3600, max_samples=3)
    t0 = 2_000_000.0
    assert buf.add(1.0, at=t0) is None
    assert buf.add(2.0, at=t0 + 1) is None
    out = buf.add(3.0, at=t0 + 2)  # hits the cap before the window elapses
    assert out is not None and out["samples"] == 3


def test_health_trend_buffer_no_values_returns_none_on_flush():
    buf = h.HealthTrendBuffer(window_s=1, max_samples=5000)
    out = buf.add(None, at=1000.0)
    assert out is None
    out = buf.add(None, at=1002.0)  # window elapsed, but no non-None values ever seen
    assert out is None


def test_health_trend_buffer_ignores_control_when_absent():
    buf = h.HealthTrendBuffer(window_s=10, max_samples=5000)
    buf.add(1.0, at=0.0)
    out = buf.add(2.0, at=11.0)
    assert out is not None
    assert out["avg_control"] is None
