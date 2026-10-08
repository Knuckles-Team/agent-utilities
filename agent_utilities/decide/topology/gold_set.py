"""The synthetic full-label gold set over the reference swarm topologies
(AU-CONTROL-R020).

Every synthetic task lists the reference topologies acceptable for it BY
CONSTRUCTION, as a ``LabelledDataset`` a ``DecisionFit``/``DecisionEval``
pins by digest, so a learned topology-routing term is promoted only through
the decision engine's fit/eval/receipt gate (promote only with a passing
benchmark).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent_utilities.decide.consumers.topology_learning import (
    QUESTION,
    Q32_ONE,
    plan_features,
)
from agent_utilities.decide.schemas import feature_schema_body
from agent_utilities.decide.topology.templates import REFERENCE_TEMPLATES, TemplateSpec
from agent_utilities.layers.clients import LayerUnavailable

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


def _candidate(spec: TemplateSpec, subtasks: int, per_agent: int) -> dict[str, Any]:
    fan = [s for s in spec.slots if s.role == "child"]
    if not fan:
        width = 1
    else:
        need = -(-subtasks // per_agent)
        low, high = fan[0].widths
        width = max(low, min(need, high))
    slots = [
        {
            "node_id": s.node_id,
            "width": width if s.role == "child" else s.widths[0],
            "rounds": s.rounds,
        }
        for s in spec.slots
    ]
    plan = {"class_iri": spec.class_iri, "slots": slots, "lease": {"per_cell": []}}
    return plan_features(plan, subtasks=subtasks, headroom=16)


@dataclass(frozen=True, slots=True)
class GoldItem:
    """One synthetic task, its candidate plans and the acceptable ones."""

    item_id: str
    class_key: str
    candidates: tuple[tuple[str, dict[str, float]], ...]
    acceptable: tuple[str, ...]


def gold_items(specs: Sequence[TemplateSpec] = REFERENCE_TEMPLATES) -> list[GoldItem]:
    """The gold set: every task against every reference topology."""
    items = []
    for name, shape, subtasks, per_agent in _TASKS:
        candidates = tuple(
            (spec.graph_id, _candidate(spec, subtasks, per_agent)) for spec in specs
        )
        acceptable = tuple(
            spec.graph_id for spec in specs if spec.class_local in _ACCEPTABLE[shape]
        )
        items.append(GoldItem(f"gold:{name}", shape, candidates, acceptable))
    return items


def _item(item: GoldItem, names: Sequence[str]) -> dict[str, Any]:
    features = [
        int(round(values.get(name, 0.0) * Q32_ONE))
        for _, values in item.candidates
        for name in names
    ]
    return {
        "item_id": item.item_id,
        "recorded_at_ms": 0,
        "class_key": item.class_key,
        "candidate_ids": [cid for cid, _ in item.candidates],
        "features": features,
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


__all__ = ["GoldItem", "gold_dataset", "gold_items"]
