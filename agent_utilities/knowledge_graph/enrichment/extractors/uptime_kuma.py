"""Uptime Kuma source extractor (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Monitors → :Service (with current up/down status). Stamped externalToolId +
domain="uptime_kuma". Client injected; tolerant of dict/list monitor shapes.
"""

from __future__ import annotations

from typing import Any

from ..models import ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "uptime_kuma"
_DOMAIN = "uptime_kuma"


def _get(config: Any, key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else getattr(config, key, None)


def _monitors(client: Any) -> list[dict]:
    m = getattr(client, "get_monitors", None)
    try:
        res = m() if callable(m) else None
    except Exception:  # noqa: BLE001
        return []
    if isinstance(res, dict):  # uptime_kuma_api returns {id: monitor}
        return [v for v in res.values() if isinstance(v, dict)]
    return [r for r in res if isinstance(r, dict)] if isinstance(res, list) else []


def extract(config: Any) -> ExtractionBatch:
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=[])
    for mon in _monitors(client):
        mid = mon.get("id") or mon.get("name")
        if mid is None:
            continue
        nodes.append(
            GraphNode(
                id=f"uptime_monitor:{mid}",
                type="Service",
                props={
                    "name": mon.get("name") or str(mid),
                    "ci_class": "monitor",
                    "url": mon.get("url"),
                    "active": mon.get("active"),
                    "externalToolId": str(mid),
                    "domain": _DOMAIN,
                },
            )
        )
    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=[])


register_extractor(CATEGORY, extract, description="Uptime Kuma monitors → KG")
