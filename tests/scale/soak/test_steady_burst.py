"""Steady + burst (CI-scaled) — SCALE-P2-1 soak scenario 1.

The full contract calls for a 24-72h steady-state + burst soak at the real 1M
population (see ``test_hardware_pending.py`` for that — documented, not run here).
This is the CI-runnable, scaled-down analogue: a short steady phase at the
contract's nominal rate, then a burst phase at several times that rate, both
driven by :func:`scripts.scale.loadgen.run_workload` against the in-memory mock
engine, asserting:

* the declared SLO percentiles hold at BOTH phases (queue/query/write/end-to-end);
* resource use is bounded — the queue backlog (submitted-but-not-yet-succeeded
  count) does not grow unbounded during the burst, i.e. the worker pool drains
  faster than the burst arrival rate at this scale;
* no lost/duplicate/cross-tenant/falsely-completed side effects.
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.integration


async def _run(loadgen, contract, *, scale: float, duration_s: float, workers: int):
    return await loadgen.run_workload(
        contract,
        scale=scale,
        duration_s=duration_s,
        turn_duration_s=0.01,  # CI speed override — see loadgen.py docstring
        num_workers=workers,
        seed=42,
    )


def test_steady_phase_slos_and_invariants_hold(loadgen, contract):
    report = asyncio.run(_run(loadgen, contract, scale=0.05, duration_s=6.0, workers=6))

    assert report.counts["turns_submitted"] > 15, (
        "need enough samples for a meaningful percentile"
    )
    for axis, target_pass in report.slo_pass.items():
        assert all(target_pass.values()), (
            f"{axis} SLO missed: {report.latency_ms[axis]} vs {report.slo_target[axis]}"
        )

    assert report.invariants["duplicate_side_effects"] == {}
    assert report.invariants["falsely_completed"] == []
    assert report.invariants["stuck_leases"] == []
    assert report.invariants["cross_tenant_violations"] == []
    assert report.ok


def test_burst_phase_bounded_backlog_and_slos_hold(loadgen, contract):
    # Burst: same duration, much higher scale (proportionally higher rates) —
    # the CI-sized analogue of the contract's burst-on-top-of-steady scenario.
    steady = asyncio.run(_run(loadgen, contract, scale=0.02, duration_s=3.0, workers=8))
    burst = asyncio.run(_run(loadgen, contract, scale=0.15, duration_s=3.0, workers=8))

    assert burst.counts["turns_submitted"] > steady.counts["turns_submitted"]

    # Bounded resource use: the worker pool must clear the backlog within the
    # drain grace window — i.e. essentially every submitted turn reaches a
    # terminal state, not an ever-growing queue of stuck `ready`/`leased` items.
    outstanding = burst.counts["turns_submitted"] - burst.counts["turns_succeeded"]
    outstanding_fraction = outstanding / max(1, burst.counts["turns_submitted"])
    assert outstanding_fraction < 0.05, (
        f"burst backlog did not drain: {outstanding}/{burst.counts['turns_submitted']} "
        "turns never reached succeeded — unbounded resource growth"
    )

    assert burst.invariants["duplicate_side_effects"] == {}
    assert burst.invariants["falsely_completed"] == []
    assert burst.invariants["cross_tenant_violations"] == []
    for axis, target_pass in burst.slo_pass.items():
        assert all(target_pass.values()), (
            f"{axis} SLO missed under burst: {burst.latency_ms[axis]}"
        )
