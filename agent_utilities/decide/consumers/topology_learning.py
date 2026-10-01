"""The statistical rung over swarm topology plans (AU-CONTROL-R020).

Cold start is deterministic-only: plans come from EG's rungs 1-3, and extra
width is never bought for quality until a head is calibrated. This module is
AU's side of getting there, all through EG's governed surfaces:

* :func:`plan_features` -- one plan's feature row under the published
  ``decide.schema.au.swarm.topology`` body (:mod:`agent_utilities.decide.schemas`);
* :func:`credit_topology_outcome` -- an INDEPENDENT ``OutcomeEvaluation`` of a
  committed plan (``DecisionLog.evaluate``): EG credits it to the whole slate
  (template, widths and fills together) and refuses an evaluator that ran the
  swarm or holds its lease; censored states carry no label;
* :func:`plan_expected_cost` / :func:`claim_cost_drift` -- a plan's declared
  cost in priced microunits, and whether a committed plan's cost drifted past
  the policy tolerance from its priced offer.

The full-label synthetic gold set (:func:`~agent_utilities.decide.topology.
gold_set.gold_items`/:func:`~agent_utilities.decide.topology.gold_set.
gold_dataset`) a ``DecisionFit``/``DecisionEval`` pins by digest lives next to
the reference templates it is built from.

Exploration over width and rounds is ZERO until a head is calibrated:
nothing here samples an alternative plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

QUESTION = "au.swarm.topology"
#: Fixed-point scale of every feature (EG's Q32).
Q32_ONE = 1 << 32
#: Censored fidelities: neither a success nor a failure (§8).
CENSORED = frozenset({"cancelled", "trace_incomplete", "outcome_uncertain"})


def plan_features(
    plan: Mapping[str, Any],
    *,
    subtasks: int,
    headroom: int,
    pooled: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """One plan's feature values, keyed like the published schema.

    ``headroom`` is the observed headroom the plan leased from (its ratio
    is the leased share); ``pooled`` are the Beta-Binomial pooled outcome
    rates (class, template, width bucket), absent until ``n_min`` is reached.
    """
    slots = list(plan.get("slots") or ())

    def leased_amount() -> int:
        per_cell = (plan.get("lease") or {}).get("per_cell") or ()
        return sum(int(cell.get("amount") or 0) for cell in per_cell)

    features = {
        "width": float(sum(int(s["width"]) for s in slots)),
        "rounds": float(max((int(s["rounds"]) for s in slots), default=1)),
        "depth": float(len(slots)),
        "subtasks": float(subtasks),
        "headroom_ratio": leased_amount() / headroom if headroom > 0 else 1.0,
        "declared_p95_ms": float(plan.get("makespan_ms") or 0),
    }
    features.update({f"pooled.{k}": float(v) for k, v in (pooled or {}).items()})
    return features


@dataclass(frozen=True, slots=True)
class TopologyOutcome:
    """One independent evaluation of an executed plan."""

    record_id: str
    evaluation_id: str
    selected_agent: str
    lease_holder: str
    fidelity: str
    success: bool | None


def evaluation_op(tenant: str, outcome: TopologyOutcome) -> dict[str, Any]:
    """The ``DecisionLog.evaluate`` op; a censored run carries no label."""
    censored = outcome.fidelity in CENSORED
    return {
        "op": "evaluate",
        "tenant_id": tenant,
        "evaluation": {
            "record_id": outcome.record_id,
            "evaluation_id": outcome.evaluation_id,
            "class": "observation",
            "selected_agent": outcome.selected_agent,
            "lease_holder": outcome.lease_holder,
            "fidelity": outcome.fidelity,
            "success": None if censored else outcome.success,
        },
    }


async def credit_topology_outcome(
    transport: Any, tenant: str, outcome: TopologyOutcome
) -> Any:
    """Join an independent evaluation to a committed plan (slate crediting)."""
    return await transport.log(evaluation_op(tenant, outcome))


def plan_expected_cost(
    plan: Mapping[str, Any], *, micros_per_token: int, micros_per_lease_unit: int = 0
) -> int | None:
    """A plan's declared cost in microunits: tokens plus leased amount, priced.

    ``None`` when a slot declares no tokens: an unknown cost is never zero.
    """
    tokens = [s.get("tokens") for s in plan.get("slots") or ()]
    if not tokens or any(not isinstance(t, int) for t in tokens):
        return None
    leased = sum(
        int(c.get("amount") or 0)
        for c in (plan.get("lease") or {}).get("per_cell") or ()
    )
    return (
        sum(int(t) for t in tokens) * micros_per_token + leased * micros_per_lease_unit
    )


def claim_cost_drift(offered: int, committed: int, tolerance_ppm: int) -> bool:
    """Whether the committed plan's cost drifted past the policy tolerance from
    the priced offer: the claim is then refused and the offer re-priced."""
    return abs(committed - offered) * 1_000_000 > tolerance_ppm * max(offered, 1)


__all__ = [
    "CENSORED",
    "QUESTION",
    "Q32_ONE",
    "TopologyOutcome",
    "claim_cost_drift",
    "credit_topology_outcome",
    "evaluation_op",
    "plan_expected_cost",
    "plan_features",
]
