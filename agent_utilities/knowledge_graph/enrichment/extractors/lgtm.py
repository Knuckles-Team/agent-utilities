"""LGTM (Grafana/Loki/Tempo/Mimir) source extractor (CONCEPT:KG-2.9).

Dashboards → :Dashboard, alerts → :Alert, datasources → :DataSource. Stamped
externalToolId + domain="lgtm". Client injected; tolerant. Replaces the stub
hydrator with a live read.
"""

from __future__ import annotations

from typing import Any

from ..models import ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "lgtm"
_DOMAIN = "lgtm"


def _get(config: Any, key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else getattr(config, key, None)


def _call(client: Any, name: str) -> list[dict]:
    m = getattr(client, name, None)
    try:
        res = m() if callable(m) else None
    except Exception:  # noqa: BLE001
        return []
    if isinstance(res, dict):
        res = res.get("data") or res.get("value") or list(res.values())
    return [r for r in res if isinstance(r, dict)] if isinstance(res, list) else []


def extract(config: Any) -> ExtractionBatch:
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=[])

    for d in _call(client, "get_dashboards"):
        uid = d.get("uid") or d.get("id") or d.get("title")
        if not uid:
            continue
        nodes.append(
            GraphNode(
                id=f"lgtm_dash:{uid}",
                type="Dashboard",
                props={
                    "name": d.get("title") or str(uid),
                    "externalToolId": str(uid),
                    "domain": _DOMAIN,
                },
            )
        )
    for a in _call(client, "get_alerts"):
        aid = (
            a.get("fingerprint")
            or a.get("id")
            or (a.get("labels") or {}).get("alertname")
        )
        if not aid:
            continue
        nodes.append(
            GraphNode(
                id=f"lgtm_alert:{aid}",
                type="Alert",
                props={
                    "name": (a.get("labels") or {}).get("alertname") or str(aid),
                    "state": a.get("status") or a.get("state"),
                    "externalToolId": str(aid),
                    "domain": _DOMAIN,
                },
            )
        )
    for ds in _call(client, "list_datasources"):
        did = ds.get("uid") or ds.get("id") or ds.get("name")
        if not did:
            continue
        nodes.append(
            GraphNode(
                id=f"lgtm_ds:{did}",
                type="DataSource",
                props={
                    "name": ds.get("name") or str(did),
                    "ds_type": ds.get("type"),
                    "externalToolId": str(did),
                    "domain": _DOMAIN,
                },
            )
        )
    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=[])


register_extractor(
    CATEGORY, extract, description="LGTM dashboards/alerts/datasources → KG"
)
