"""ST-12: the statistical rung over swarm topology plans (SWARM-TOPOLOGY-DECIDE-DESIGN §8).

Cold start is deterministic-only: plans come from EG's rungs 1-3, and extra
width is never bought for quality until a head is calibrated. This module is
AU's side of getting there, all through EG's governed surfaces:

* :func:`plan_features` -- one plan's feature row under the published
  ``decide.schema.au.swarm.topology`` body (:mod:`agent_utilities.decide.schemas`);
* :func:`credit_topology_outcome` -- an INDEPENDENT ``OutcomeEvaluation`` of a
  committed plan (``DecisionLog.evaluate``): EG credits it to the whole slate
  (template + widths + fills, EH-012) and refuses an evaluator that ran the
  swarm or holds its lease; censored states carry no label;
* :func:`gold_items` / :func:`gold_dataset` -- the full-label gold set: every
  synthetic task lists the reference topologies acceptable for it BY
  CONSTRUCTION (the 2026-09-17 synthetic-data directive), as a
  ``LabelledDataset`` a ``DecisionFit``/``DecisionEval`` pins by digest, so a
  head is promoted only through EH-040's receipt gate (Learn-then-Test).

Exploration over width and rounds is ZERO (operator ruling 2026-09-24, Q2):
nothing here samples an alternative plan.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent_utilities.decide.schemas import feature_schema_body
from agent_utilities.decide.topology.templates import REFERENCE_TEMPLATES, TemplateSpec
from agent_utilities.layers.clients import LayerUnavailable

QUESTION = "au.swarm.topology"
#: Fixed-point scale of every feature (EG's Q32).
Q32_ONE = 1 << 32
#: Censored fidelities: neither a success nor a failure (§8).
CENSORED = frozenset({"cancelled", "trace_incomplete", "outcome_uncertain"})


def _q32(value: float) -> int:
    return int(round(value * Q32_ONE))


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
    leased = sum(
        int(c.get("amount") or 0)
        for c in (plan.get("lease") or {}).get("per_cell") or ()
    )
    features = {
        "width": float(sum(int(s["width"]) for s in slots)),
        "rounds": float(max((int(s["rounds"]) for s in slots), default=1)),
        "depth": float(len(slots)),
        "subtasks": float(subtasks),
        "headroom_ratio": leased / headroom if headroom > 0 else 1.0,
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


#: Synthetic task shapes: (name, task-shape local name, subtasks, per-agent load).
_TASKS: tuple[tuple[str, str, int, int], ...] = (
    ("independent-2", "IndependentSubtasks", 2, 1),
    ("independent-6", "IndependentSubtasks", 6, 1),
    ("sequential", "SequentialDependency", 1, 1),
    ("negotiation", "NeedsNegotiation", 3, 1),
    ("independent-check", "NeedsIndependentCheck", 1, 1),
    ("atomic", "TaskShape", 1, 1),
)
#: The reference topologies acceptable for each task shape, BY CONSTRUCTION.
_ACCEPTABLE: dict[str, frozenset[str]] = {
    "IndependentSubtasks": frozenset({"FanOutJoin", "SupervisorWorkers"}),
    "SequentialDependency": frozenset({"Pipeline", "Single"}),
    "NeedsNegotiation": frozenset({"Council", "Debate"}),
    "NeedsIndependentCheck": frozenset({"CritiqueLoop"}),
    "TaskShape": frozenset({"Single"}),
}


def _needed_width(spec: TemplateSpec, subtasks: int, per_agent: int) -> int:
    fan = [s for s in spec.slots if s.role == "child"]
    if not fan:
        return 1
    need = -(-subtasks // per_agent)
    low, high = fan[0].widths
    return max(low, min(need, high))


def _candidate(spec: TemplateSpec, subtasks: int, per_agent: int) -> dict[str, Any]:
    width = _needed_width(spec, subtasks, per_agent)
    slots = [
        {
            "node_id": s.node_id,
            "width": width if s.role == "child" else s.widths[0],
            "rounds": s.rounds,
        }
        for s in spec.slots
    ]
    return {"class_iri": spec.class_iri, "slots": slots, "lease": {"per_cell": []}}


@dataclass(frozen=True, slots=True)
class GoldItem:
    """One synthetic task, its candidate plans and the acceptable ones."""

    item_id: str
    class_key: str
    candidates: tuple[tuple[str, dict[str, float]], ...]
    acceptable: tuple[str, ...]


def gold_items(
    specs: Sequence[TemplateSpec] = REFERENCE_TEMPLATES, headroom: int = 16
) -> list[GoldItem]:
    """The gold set: every task against every reference topology."""
    items = []
    for name, shape, subtasks, per_agent in _TASKS:
        candidates = tuple(
            (
                spec.graph_id,
                plan_features(
                    _candidate(spec, subtasks, per_agent),
                    subtasks=subtasks,
                    headroom=headroom,
                ),
            )
            for spec in specs
        )
        acceptable = tuple(
            spec.graph_id for spec in specs if spec.class_local in _ACCEPTABLE[shape]
        )
        items.append(GoldItem(f"gold:{name}", shape, candidates, acceptable))
    return items


def _row(values: Mapping[str, float], names: Sequence[str]) -> list[int]:
    return [_q32(values.get(name, 0.0)) for name in names]


def _item(item: GoldItem, names: Sequence[str]) -> dict[str, Any]:
    return {
        "item_id": item.item_id,
        "recorded_at_ms": 0,
        "class_key": item.class_key,
        "candidate_ids": [cid for cid, _ in item.candidates],
        "features": [v for _, values in item.candidates for v in _row(values, names)],
        "label": {
            "label": "gold",
            "acceptable": list(item.acceptable),
            "source": "synthetic_construction",
        },
    }


def _eg_digest(value: Mapping[str, Any]) -> str:
    try:
        stat = importlib.import_module("epistemic_graph.decision_stat")
    except ImportError as exc:
        raise LayerUnavailable("EG decision_stat helpers are absent") from exc
    return str(stat.dataset_digest(value))


def gold_dataset(
    items: Sequence[GoldItem],
    *,
    digest_of: Callable[[Mapping[str, Any]], str] = _eg_digest,
) -> tuple[dict[str, Any], str]:
    """``(LabelledDataset, gold_set_digest)`` over the published schema."""
    schema = feature_schema_body(QUESTION)
    names = [feature["name"] for feature in schema["features"]]
    dataset = {
        "schema_version": 1,
        "feature_schema_digest": digest_of(schema),
        "feature_names": names,
        "scale": "q32",
        "items": [_item(item, names) for item in items],
        "synthetic": True,
    }
    return dataset, digest_of(dataset)


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
    "GoldItem",
    "TopologyOutcome",
    "claim_cost_drift",
    "credit_topology_outcome",
    "evaluation_op",
    "gold_dataset",
    "gold_items",
    "plan_expected_cost",
    "plan_features",
]
