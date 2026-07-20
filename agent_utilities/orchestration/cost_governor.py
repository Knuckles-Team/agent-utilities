#!/usr/bin/python
from __future__ import annotations

"""Throughput-per-dollar scaling governor.

CONCEPT:AU-OS.scaling.cost-aware-autoscaling — cost-and-throughput aware autoscaling that caps scale-up to a configured hourly budget and surfaces a throughput-per-dollar efficiency metric instead of reacting to a single load signal alone

The OS-5.29 autoscaler does per-service target-tracking on one load signal vs.
static bounds — "add replicas when one metric is hot", with no notion of what the
extra replicas cost or whether they are the most efficient place to spend compute.
This governor adds the cost/throughput half the paper's §5.1 digital-worker-collective
lever needs, as a small, pure, opt-in layer over the unchanged target-tracking math:

* a per-replica hourly cost estimate (from node-class power draw + an energy price,
  or an explicit per-tier rate), and
* a budget cap that limits a scale-up so the running cost stays within a configured
  hourly ceiling, plus a throughput-per-dollar metric recorded for observability.

When no budget is configured the cap is a no-op — the autoscaler behaves exactly as
before. Cheap, in-repo; the node-class rates only need tuning on real hardware.
"""

import math
from dataclasses import dataclass

#: Default energy price (USD per kWh) used when deriving cost from node power draw.
DEFAULT_USD_PER_KWH = 0.15

#: Coarse per-tier hourly USD/replica fallback when no node power data is available.
#: Tunable; reflects "a GPU replica costs far more per hour than a CPU one".
TIER_HOURLY_USD: dict[str, float] = {
    "cpu": 0.05,
    "ai": 0.90,  # GPU-backed
    "gpu": 0.90,
    "gateway": 0.05,
    "worker": 0.05,
}
_DEFAULT_TIER_HOURLY_USD = 0.05


def replica_cost_per_hour(
    *,
    tier: str = "",
    power_watts: float | None = None,
    usd_per_kwh: float = DEFAULT_USD_PER_KWH,
) -> float:
    """Estimate the hourly USD cost of one replica.

    Prefers a measured ``power_watts`` (× energy price) when known; otherwise falls
    back to the coarse per-``tier`` rate. Always returns a positive number so a
    budget cap is well-defined.
    """
    if power_watts is not None and power_watts > 0:
        return (float(power_watts) / 1000.0) * float(usd_per_kwh)
    return TIER_HOURLY_USD.get((tier or "").lower(), _DEFAULT_TIER_HOURLY_USD)


def throughput_per_dollar(load_value: float, cost_per_hour: float) -> float:
    """A simple efficiency metric: served-load units per USD/hour (0 if free/idle)."""
    if cost_per_hour <= 0:
        return 0.0
    return float(load_value) / float(cost_per_hour)


@dataclass
class CostVerdict:
    """The cost-aware adjustment to a target-tracking decision."""

    replicas: int
    cost_per_hour: float
    throughput_per_dollar: float
    capped: bool = False
    reason: str = ""


def cost_aware_cap(
    desired: int,
    current: int,
    *,
    cost_per_replica_hour: float,
    budget_per_hour: float | None,
    load_value: float = 0.0,
) -> CostVerdict:
    """Cap a scale-up so the fleet cost stays within ``budget_per_hour``.

    Only scale-*ups* are capped (a scale-down always saves money); a ``None`` budget
    is a no-op. The cap floors at ``current`` (never forces a scale-down) — if even
    the current replica count is over budget, that is the reconciler's problem, not
    a reason to shed load here.
    """
    cost_now = max(0, int(desired)) * cost_per_replica_hour
    tpd = throughput_per_dollar(load_value, cost_now)
    if budget_per_hour is None or desired <= current or cost_per_replica_hour <= 0:
        return CostVerdict(desired, cost_now, tpd)

    affordable = int(math.floor(budget_per_hour / cost_per_replica_hour))
    if affordable >= desired:
        return CostVerdict(desired, cost_now, tpd)
    capped_replicas = max(current, affordable)
    capped_cost = capped_replicas * cost_per_replica_hour
    return CostVerdict(
        replicas=capped_replicas,
        cost_per_hour=capped_cost,
        throughput_per_dollar=throughput_per_dollar(load_value, capped_cost),
        capped=True,
        reason=(
            f"cost cap: {desired}→{capped_replicas} replicas "
            f"(${cost_per_replica_hour:.3g}/replica/h, budget ${budget_per_hour:.3g}/h)"
        ),
    )
