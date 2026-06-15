"""Kafka source extractor (CONCEPT:KG-2.9).

Topics → :Topic, consumer groups → :Service. Stamped externalToolId +
domain="kafka". Client injected; tolerant of native/REST list shapes.
"""

from __future__ import annotations

from typing import Any

from ..models import ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "kafka"
_DOMAIN = "kafka"


def _get(config: Any, key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else getattr(config, key, None)


def _names(res: Any) -> list[str]:
    if isinstance(res, dict):
        for k in ("topics", "data", "value", "groups"):
            v = res.get(k)
            if isinstance(v, list):
                res = v
                break
    if isinstance(res, list):
        out = []
        for r in res:
            if isinstance(r, str):
                out.append(r)
            elif isinstance(r, dict):
                n = (
                    r.get("name")
                    or r.get("topic")
                    or r.get("group_id")
                    or r.get("groupId")
                )
                if n:
                    out.append(str(n))
        return out
    return []


def _call(client: Any, name: str) -> Any:
    m = getattr(client, name, None)
    try:
        return m() if callable(m) else None
    except Exception:  # noqa: BLE001
        return None


def extract(config: Any) -> ExtractionBatch:
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=[])

    for t in _names(_call(client, "list_topics")):
        nodes.append(
            GraphNode(
                id=f"kafka_topic:{t}",
                type="Topic",
                props={"name": t, "externalToolId": t, "domain": _DOMAIN},
            )
        )
    for g in _names(_call(client, "list_consumer_groups")):
        nodes.append(
            GraphNode(
                id=f"kafka_group:{g}",
                type="Service",
                props={
                    "name": g,
                    "ci_class": "consumer_group",
                    "externalToolId": g,
                    "domain": _DOMAIN,
                },
            )
        )
    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=[])


register_extractor(CATEGORY, extract, description="Kafka topics/consumer-groups → KG")
