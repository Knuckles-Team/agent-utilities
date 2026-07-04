"""Twenty CRM source extractor (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Full-CRM bidirectional read: companies → :Customer, people → :Person,
opportunities → :SalesOrder (consistent with the Salesforce mapping), with
BELONGS_TO / PLACED_BY edges. Stamped externalToolId + domain="twenty". Client
(``twenty_mcp.auth.get_client()``) injected; tolerant of REST/GraphQL list shapes.
"""

from __future__ import annotations

from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "twenty"
_DOMAIN = "twenty"


def _get(config: Any, key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else getattr(config, key, None)


def _records(res: Any, plural: str) -> list[dict]:
    """Pull records from REST {data:[...]} or GraphQL {data:{plural:{edges:[{node}]}}}."""
    if not isinstance(res, dict):
        return [r for r in res if isinstance(r, dict)] if isinstance(res, list) else []
    data = res.get("data", res)
    if isinstance(data, dict) and plural in data:
        conn = data[plural]
        if isinstance(conn, dict) and isinstance(conn.get("edges"), list):
            return [
                e["node"]
                for e in conn["edges"]
                if isinstance(e, dict) and e.get("node")
            ]
        if isinstance(conn, list):
            return [r for r in conn if isinstance(r, dict)]
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    return []


def _call(client: Any, name: str) -> Any:
    m = getattr(client, name, None)
    try:
        return m() if callable(m) else None
    except Exception:  # noqa: BLE001
        return None


def _name(rec: dict, *keys: str) -> str | None:
    for k in keys:
        v = rec.get(k)
        if isinstance(v, dict):  # Twenty FullName {firstName,lastName}
            joined = " ".join(str(x) for x in v.values() if x)
            if joined.strip():
                return joined.strip()
        elif v:
            return str(v)
    return None


def extract(config: Any) -> ExtractionBatch:
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    for c in _records(_call(client, "get_companies"), "companies"):
        cid = c.get("id")
        if cid:
            nodes.append(
                GraphNode(
                    id=f"twcompany:{cid}",
                    type="Customer",
                    props={
                        "name": _name(c, "name"),
                        "externalToolId": str(cid),
                        "domain": _DOMAIN,
                    },
                )
            )
    for p in _records(_call(client, "get_people"), "people"):
        pid = p.get("id")
        if not pid:
            continue
        node_id = f"twperson:{pid}"
        nodes.append(
            GraphNode(
                id=node_id,
                type="Person",
                props={
                    "name": _name(p, "name", "displayName"),
                    "email": _name(p, "email"),
                    "externalToolId": str(pid),
                    "domain": _DOMAIN,
                },
            )
        )
        comp = p.get("companyId") or (p.get("company") or {}).get("id")
        if comp:
            edges.append(
                EnrichmentEdge(
                    source=node_id, target=f"twcompany:{comp}", rel_type="BELONGS_TO"
                )
            )
    for o in _records(_call(client, "get_opportunities"), "opportunities"):
        oid = o.get("id")
        if not oid:
            continue
        node_id = f"twopp:{oid}"
        nodes.append(
            GraphNode(
                id=node_id,
                type="SalesOrder",
                props={
                    "name": _name(o, "name"),
                    "stage": o.get("stage"),
                    "amount": (o.get("amount") or {}).get("amountMicros")
                    if isinstance(o.get("amount"), dict)
                    else o.get("amount"),
                    "externalToolId": str(oid),
                    "domain": _DOMAIN,
                },
            )
        )
        comp = o.get("companyId") or (o.get("company") or {}).get("id")
        if comp:
            edges.append(
                EnrichmentEdge(
                    source=node_id, target=f"twcompany:{comp}", rel_type="PLACED_BY"
                )
            )

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_extractor(
    CATEGORY, extract, description="Twenty CRM (companies/people/opps) → KG"
)
