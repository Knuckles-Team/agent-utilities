"""wger wellness source extractor (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Reads body/nutrition/workout data from a wger instance into canonical wellness
OWL nodes: weight + measurements → :BodyMeasurement, workout sessions →
:WorkoutSession, nutrition plans → :MealPlan. Stamped externalToolId +
domain="wger". Client (``wger_agent.auth.get_client()``) injected; tolerant of the
DRF ``{"results": [...]}`` pagination shape (and bare lists).
"""

from __future__ import annotations

from typing import Any

from ...core import owl_bridge
from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "wger"
_DOMAIN = "wger"


def _get(config: Any, key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else getattr(config, key, None)


def _rows(res: Any) -> list[dict]:
    """DRF ``{"results": [...]}`` or a bare list → list[dict]."""
    if isinstance(res, dict):
        res = res.get("results", res.get("data", []))
    return [r for r in res if isinstance(r, dict)] if isinstance(res, list) else []


def _call(client: Any, name: str) -> Any:
    m = getattr(client, name, None)
    try:
        return m() if callable(m) else None
    except Exception:  # noqa: BLE001 - tolerant of unconfigured endpoints
        return None


def extract(config: Any) -> ExtractionBatch:
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    for w in _rows(_call(client, "get_weight_entries")):
        wid = w.get("id")
        if wid is None:
            continue
        nodes.append(
            GraphNode(
                id=f"wger:weight:{wid}",
                type="BodyMeasurement",
                props={
                    "kind": "weight",
                    "value": w.get("weight"),
                    "date": w.get("date"),
                    "externalToolId": str(wid),
                    "domain": _DOMAIN,
                },
            )
        )
    for m in _rows(_call(client, "get_measurements")):
        mid = m.get("id")
        if mid is None:
            continue
        nodes.append(
            GraphNode(
                id=f"wger:meas:{mid}",
                type="BodyMeasurement",
                props={
                    "kind": "measurement",
                    "category": m.get("category"),
                    "value": m.get("value"),
                    "date": m.get("date"),
                    "externalToolId": str(mid),
                    "domain": _DOMAIN,
                },
            )
        )
    for s in _rows(_call(client, "get_workout_sessions")):
        sid = s.get("id")
        if sid is None:
            continue
        node_id = f"wger:session:{sid}"
        nodes.append(
            GraphNode(
                id=node_id,
                type="WorkoutSession",
                props={
                    "date": s.get("date"),
                    "impression": s.get("impression"),
                    "notes": s.get("notes"),
                    "externalToolId": str(sid),
                    "domain": _DOMAIN,
                },
            )
        )
        routine = s.get("routine")
        if routine is not None:
            edges.append(
                EnrichmentEdge(
                    source=node_id,
                    target=f"wger:routine:{routine}",
                    rel_type="PART_OF",
                )
            )
    for p in _rows(_call(client, "get_nutrition_plans")):
        pid = p.get("id")
        if pid is None:
            continue
        nodes.append(
            GraphNode(
                id=f"wger:nutplan:{pid}",
                type="MealPlan",
                props={
                    "description": p.get("description"),
                    "only_logging": p.get("only_logging"),
                    "externalToolId": str(pid),
                    "domain": _DOMAIN,
                },
            )
        )

    owl_bridge.register_promotable_node_types({n.type for n in nodes})
    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_extractor(
    CATEGORY, extract, description="wger wellness (weight/measurements/sessions) → KG"
)
