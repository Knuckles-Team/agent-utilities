"""Technitium DNS source extractor (CONCEPT:KG-2.9).

Maps DNS zones/records into the uniform ExtractionBatch: zones → :ConfigurationItem,
records → :ConfigurationItem (ci_class=dns_record) with a CONTAINS edge zone→record.
Stamped externalToolId + domain="technitium_dns". Client injected; tolerant.
"""

from __future__ import annotations

from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "technitium_dns"
_DOMAIN = "technitium_dns"


def _get(config: Any, key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else getattr(config, key, None)


def _call(client: Any, name: str, *args: Any) -> Any:
    m = getattr(client, name, None)
    if not callable(m):
        return None
    try:
        return m(*args)
    except Exception:  # noqa: BLE001
        return None


def _rows(res: Any, *keys: str) -> list[dict]:
    if isinstance(res, dict):
        for k in keys:
            v = res.get(k)
            if isinstance(v, list):
                return [r for r in v if isinstance(r, dict)]
        return []
    return [r for r in res if isinstance(r, dict)] if isinstance(res, list) else []


def extract(config: Any) -> ExtractionBatch:
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    for z in _rows(_call(client, "list_zones"), "zones", "value", "data"):
        zone = z.get("name") or z.get("zone")
        if not zone:
            continue
        zid = f"dnszone:{zone}"
        nodes.append(
            GraphNode(
                id=zid,
                type="ConfigurationItem",
                props={
                    "name": zone,
                    "ci_class": "dns_zone",
                    "externalToolId": zone,
                    "domain": _DOMAIN,
                },
            )
        )
        for r in _rows(_call(client, "get_records", zone), "records", "value", "data"):
            rname = r.get("name") or zone
            rtype = r.get("type") or "A"
            rid = f"dnsrecord:{zone}:{rname}:{rtype}"
            nodes.append(
                GraphNode(
                    id=rid,
                    type="ConfigurationItem",
                    props={
                        "name": rname,
                        "ci_class": f"dns_{rtype.lower()}",
                        "record_type": rtype,
                        "value": (r.get("rData") or {}).get("value")
                        if isinstance(r.get("rData"), dict)
                        else r.get("value"),
                        "externalToolId": rid.split(":", 1)[1],
                        "domain": _DOMAIN,
                    },
                )
            )
            edges.append(EnrichmentEdge(source=zid, target=rid, rel_type="CONTAINS"))

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_extractor(CATEGORY, extract, description="Technitium DNS zones/records → KG")
