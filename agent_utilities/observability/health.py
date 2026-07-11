#!/usr/bin/python
from __future__ import annotations

"""Metric-agnostic health-intelligence kernels — CONCEPT:AU-OS.observability.unified-health-kernels.

Generalizes fan-manager's proven thermal-intelligence pattern
(``fan_manager.kg_control`` / ``fan_manager.fan_manager._kg_record_thermal_sample``,
see ``reports/epistemic-graph-fan-control-plan.md``) from °C to **any named numeric
signal** a telemetry-producing agent emits — CPU%, disk%, load average, latency,
restart-count, and so on (``reports/unified-infra-intelligence-plan.md``).

The reasoning is pure, cheap statistics — percentiles + a least-squares slope over a
bounded window of already-distilled trend rows — no ML, no numpy, O(window). Four
kernels compose the full pattern:

1. :class:`HealthTrendBuffer` — **distill, don't dump.** A rolling in-memory window
   collapses raw samples into ONE lightweight trend dict per aggregate window
   (generalizes the ``_thermal_buf``/flush logic in ``fan_manager.fan_manager``).
2. :func:`compute_baseline` — **learn a baseline.** A per-entity fingerprint (p50/p95,
   idle/load envelope, inertia) from accumulated trend rows.
3. :func:`detect_anomaly` — **detect.** z-score off the entity's *own* baseline (not a
   global threshold) — the earliest signal of drift.
4. :func:`correlate` — **correlate.** Collapse simultaneous same-kind anomalies across
   entities into one systemic cause (generalizes ``classify_ambient``'s "ambient").

KG I/O (typed ``:HealthTrend``/``:HealthBaseline``/``:HealthAnomaly`` writers/readers)
lives in :mod:`agent_utilities.observability.health_ingest`; this module is reason-only.
"""

import math
import time
from typing import Any

DEFAULT_MIN_WINDOWS = 6
DEFAULT_Z_THRESH = 3.0
DEFAULT_SATURATED_CONTROL = 95.0


# --------------------------------------------------------------------------- #
# small numeric helpers (pure, stdlib only) — lifted from fan_manager.kg_control
# --------------------------------------------------------------------------- #
def _percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolated percentile of ``values`` (``pct`` in 0..100)."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(math.floor(k))
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def _slope(xs: list[float], ys: list[float]) -> float | None:
    """Least-squares slope of ``ys`` on ``xs`` (``None`` if degenerate)."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False)) / denom


# --------------------------------------------------------------------------- #
# 1. distill-to-trend rolling window
# --------------------------------------------------------------------------- #
class HealthTrendBuffer:
    """Rolling in-memory window that distills raw samples to ONE trend dict.

    Generalizes the ``_thermal_buf`` / flush logic in
    ``fan_manager.fan_manager._kg_record_thermal_sample`` from CPU °C to any named
    numeric signal. Samples accumulate cheaply in memory and are collapsed to a
    single ``{min,max,avg,avg_control,samples,window_s}`` dict once the aggregate
    window elapses *or* a hard sample cap is hit (whichever first — the cap guards
    memory on a very hot sampling rate). The high-resolution stream stays wherever
    the caller already ships it (e.g. Prometheus); this buffer keeps the KG write
    rate bounded to one row per window per entity/signal.
    """

    def __init__(self, *, window_s: int = 3600, max_samples: int = 5000) -> None:
        self.window_s = window_s
        self.max_samples = max_samples
        self._buf: list[tuple[float, float | None, float | None]] = []
        self._last_flush: float | None = None

    def add(
        self,
        value: float | None,
        *,
        control: float | None = None,
        at: float | None = None,
    ) -> dict[str, Any] | None:
        """Append one ``(value, control)`` sample; return a distilled trend dict when
        the window elapses or the cap is hit, else ``None``."""
        now = at if at is not None else time.time()
        if self._last_flush is None:
            # seed the window baseline from the first sample's own clock (real or
            # caller-supplied), never from construction time — a buffer built well
            # before its first sample must not count that gap toward the window.
            self._last_flush = now
        try:
            v = float(value) if value is not None else None
        except (TypeError, ValueError):
            v = None
        try:
            c = float(control) if control is not None else None
        except (TypeError, ValueError):
            c = None
        self._buf.append((now, v, c))
        if now - self._last_flush < self.window_s and len(self._buf) < self.max_samples:
            return None
        return self._flush(now)

    def _flush(self, now: float) -> dict[str, Any] | None:
        values = [v for _, v, _ in self._buf if v is not None]
        controls = [c for _, _, c in self._buf if c is not None]
        n = len(self._buf)
        window_s = self.window_s
        self._buf.clear()
        self._last_flush = now
        if not values:
            return None
        return {
            "min": min(values),
            "max": max(values),
            "avg": round(sum(values) / len(values), 3),
            "avg_control": round(sum(controls) / len(controls), 3)
            if controls
            else None,
            "samples": n,
            "window_s": window_s,
        }


# --------------------------------------------------------------------------- #
# 2. learn a baseline
# --------------------------------------------------------------------------- #
def compute_baseline(
    trends: list[dict[str, Any]],
    *,
    value_key: str,
    peak_key: str | None = None,
    control_key: str | None = None,
    min_windows: int = DEFAULT_MIN_WINDOWS,
) -> dict[str, Any] | None:
    """Distill an entity's trend-window history into a baseline fingerprint.

    Generalizes ``fan_manager.kg_control.compute_baseline`` to any named signal:
    ``value_key``/``peak_key``/``control_key`` select which fields of each trend row
    carry the average value, the peak value (for p95; falls back to ``value_key`` when
    absent), and an optional control signal (e.g. fan %, replica count). Returns
    ``None`` when there is too little history (fewer than ``min_windows`` rows with a
    value) to trust. Otherwise a dict with the value distribution (``p50``/``p95``),
    the idle/load envelope (``min_env``/``max_env``), the mean control level
    (``avg_control``), and ``inertia`` — the |signal per control-unit| slope, i.e. how
    much a control step actually moves the value (``None`` when the control barely
    varied across the window, or no ``control_key`` was given).
    """
    avg_v = [float(v) for r in trends if (v := r.get(value_key)) is not None]
    peak_v = (
        [float(v) for r in trends if (v := r.get(peak_key)) is not None]
        if peak_key
        else []
    )
    ctrl_v = (
        [float(v) for r in trends if (v := r.get(control_key)) is not None]
        if control_key
        else []
    )
    if not avg_v or len(avg_v) < min_windows:
        return None
    p50 = _percentile(avg_v, 50)
    p95 = _percentile(peak_v or avg_v, 95)
    if p50 is None or p95 is None:
        return None

    inertia = None
    if control_key:
        pairs = [
            (float(c), float(v))
            for r in trends
            if (c := r.get(control_key)) is not None
            and (v := r.get(value_key)) is not None
        ]
        if len({c for c, _ in pairs}) >= 3:
            s = _slope([c for c, _ in pairs], [v for _, v in pairs])
            inertia = round(abs(s), 3) if s is not None else None

    return {
        "p50": round(p50, 3),
        "p95": round(p95, 3),
        "min_env": round(min(avg_v), 3),
        "max_env": round(max(avg_v), 3),
        "avg_control": round(sum(ctrl_v) / len(ctrl_v), 3) if ctrl_v else None,
        "inertia": inertia,
        "windows": len(avg_v),
    }


# --------------------------------------------------------------------------- #
# 3. detect anomalies off the entity's OWN baseline
# --------------------------------------------------------------------------- #
def detect_anomaly(
    recent: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
    *,
    value_key: str,
    control_key: str | None = None,
    z_thresh: float = DEFAULT_Z_THRESH,
    saturated_control: float = DEFAULT_SATURATED_CONTROL,
) -> dict[str, Any] | None:
    """Flag an entity drifting off its own baseline.

    Generalizes ``fan_manager.kg_control.detect_anomaly``. ``above-baseline``: the
    recent window's average is beyond ``p95`` *and* ``z_thresh`` z-scores above
    ``p50`` — the early signal of a genuine regression. ``saturated``: the control
    signal is pinned at/above ``saturated_control`` yet the value is still beyond the
    baseline's load envelope (``max_env``) — the generic form of fan-manager's
    "cooling-saturated" (fans pinned, still hot): whatever knob the caller controls is
    maxed out and it isn't enough. ``None`` when the entity is behaving normally, or
    when there's no baseline/recent data.

    Unlike the fan-manager original (which floors the z-score spread at a fixed 3°C,
    a domain-specific magic number), this generalizes the floor to a signal-agnostic
    epsilon — correct for arbitrary units without a per-signal tunable.
    """
    if not baseline or not recent:
        return None
    r_vals = [float(v) for r in recent if (v := r.get(value_key)) is not None]
    r_ctrl = (
        [float(v) for r in recent if (v := r.get(control_key)) is not None]
        if control_key
        else []
    )
    if not r_vals:
        return None
    r_avg = sum(r_vals) / len(r_vals)
    p50, p95 = float(baseline["p50"]), float(baseline["p95"])
    spread = (p95 - p50) or 1e-6
    z = (r_avg - p50) / spread

    kind = None
    if r_avg > p95 and z >= z_thresh:
        kind = "above-baseline"
    elif (
        r_ctrl
        and (sum(r_ctrl) / len(r_ctrl)) >= saturated_control
        and r_avg > float(baseline["max_env"])
    ):
        kind = "saturated"
    if not kind:
        return None
    return {
        "kind": kind,
        "zscore": round(z, 2),
        "observed": round(r_avg, 3),
        "expected": round(p50, 3),
    }


# --------------------------------------------------------------------------- #
# 4. correlate simultaneous anomalies into one systemic cause
# --------------------------------------------------------------------------- #
def correlate(
    anomalies_by_entity: dict[str, dict[str, Any] | None],
    total: int,
    *,
    kind: str = "above-baseline",
    systemic_kind: str = "systemic",
) -> dict[str, dict[str, Any] | None]:
    """Retag simultaneous same-``kind`` anomalies across entities as ``systemic_kind``.

    Generalizes ``fan_manager.kg_control.classify_ambient`` (where "ambient" — the
    room/rack AC, not N independent faults — is one instance of ``systemic_kind``).
    If a majority of ``total`` entities show ``kind`` at once, it's a shared cause,
    not independent faults — collapse them to one signal instead of an alert storm.
    Mutates and returns ``anomalies_by_entity``.
    """
    matching = [e for e, a in anomalies_by_entity.items() if a and a["kind"] == kind]
    if len(matching) >= max(2, math.ceil(total / 2)):
        for e in matching:
            a = anomalies_by_entity[e]
            if a is not None:
                a["kind"] = systemic_kind
    return anomalies_by_entity
