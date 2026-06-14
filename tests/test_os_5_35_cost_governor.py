"""Throughput-per-dollar scaling governor (CONCEPT:OS-5.35).

The OS-5.29 target-tracking math is unchanged; this only trims a scale-up that
would breach a configured hourly budget and surfaces a cost/throughput lens. With
no budget configured the autoscaler behaves exactly as before.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "unit"))

from fleet_autonomy_fakes import FakeEngine  # noqa: E402

from agent_utilities.orchestration.action_policy import ActionPolicy  # noqa: E402
from agent_utilities.orchestration.cost_governor import (  # noqa: E402
    cost_aware_cap,
    replica_cost_per_hour,
    throughput_per_dollar,
)
from agent_utilities.orchestration.fleet_actuation import DryRunActuator  # noqa: E402
from agent_utilities.orchestration.fleet_autoscaler import FleetAutoscaler  # noqa: E402
from agent_utilities.orchestration.fleet_reconciler import ScalingSpec  # noqa: E402

pytestmark = pytest.mark.concept("OS-5.35")


# ── pure cost model ──────────────────────────────────────────────────


class TestCostModel:
    def test_power_draw_beats_tier_fallback(self):
        # 200W @ $0.15/kWh = 0.03 $/h; ignores the tier rate.
        assert replica_cost_per_hour(tier="gpu", power_watts=200) == pytest.approx(0.03)

    def test_tier_fallback_when_no_power(self):
        assert replica_cost_per_hour(tier="gpu") == 0.90
        assert replica_cost_per_hour(tier="cpu") == 0.05
        assert replica_cost_per_hour(tier="unknown") == 0.05  # default

    def test_throughput_per_dollar(self):
        assert throughput_per_dollar(100.0, 2.0) == 50.0
        assert throughput_per_dollar(100.0, 0.0) == 0.0  # free/idle ⇒ 0


class TestCostAwareCap:
    def test_no_budget_is_noop(self):
        v = cost_aware_cap(10, 1, cost_per_replica_hour=1.0, budget_per_hour=None)
        assert v.replicas == 10 and not v.capped

    def test_scale_up_capped_to_budget(self):
        v = cost_aware_cap(10, 1, cost_per_replica_hour=1.0, budget_per_hour=3.0)
        assert v.replicas == 3 and v.capped
        assert v.cost_per_hour == 3.0 and "cost cap" in v.reason

    def test_within_budget_uncapped(self):
        v = cost_aware_cap(3, 1, cost_per_replica_hour=1.0, budget_per_hour=10.0)
        assert v.replicas == 3 and not v.capped

    def test_scale_down_never_capped(self):
        v = cost_aware_cap(1, 5, cost_per_replica_hour=1.0, budget_per_hour=2.0)
        assert v.replicas == 1 and not v.capped  # a scale-down always saves money

    def test_cap_floors_at_current(self):
        # current already over budget: never forced below current here.
        v = cost_aware_cap(8, 5, cost_per_replica_hour=1.0, budget_per_hour=2.0)
        assert v.replicas == 5 and v.capped


# ── autoscaler integration ───────────────────────────────────────────


class _Obs:
    def __init__(self, replicas: int, status: str = "up") -> None:
        self.replicas = replicas
        self.status = status


class _Signal:
    name = "test"

    def __init__(self, value: float) -> None:
        self.value = value

    def signal_value(self, service: str, signal: str) -> float:
        return self.value


def _spec() -> ScalingSpec:
    return ScalingSpec(
        min_replicas=1,
        max_replicas=20,
        signal="queue_depth",
        target=100.0,
        scale_up_step=20,
        scale_down_step=1,
        cooldown_s=0.0,
    )


def _scaler(monkeypatch, *, budget, cost=1.0, value=1000.0):
    if budget is not None:
        monkeypatch.setenv("FLEET_SCALE_BUDGET_USD_PER_HOUR", str(budget))
    monkeypatch.setenv("FLEET_REPLICA_COST_USD_PER_HOUR", str(cost))
    eng = FakeEngine()
    return FleetAutoscaler(
        eng,
        observer=object(),
        actuator=DryRunActuator(),
        policy=ActionPolicy(engine=eng),
        signal_provider=_Signal(value),
    )


class TestAutoscalerCostCap:
    def test_scale_up_trimmed_by_budget(self, monkeypatch):
        # target tracking wants 10 (1000/100); budget $3/h @ $1/replica ⇒ cap to 3.
        scaler = _scaler(monkeypatch, budget=3.0)
        ev = scaler._evaluate_service("svc", _spec(), _Obs(replicas=1))
        assert ev.desired == 3
        assert ev.current == 1

    def test_no_budget_unchanged(self, monkeypatch):
        scaler = _scaler(monkeypatch, budget=None)
        ev = scaler._evaluate_service("svc", _spec(), _Obs(replicas=1))
        assert ev.desired == 10  # full target-tracking result, uncapped
