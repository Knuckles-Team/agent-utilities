"""Portainer source extractor (CONCEPT:KG-2.9).

Endpoints → :Server, stacks → :Service, containers → :AssetInstance (RUNS_ON the
endpoint). Stamped externalToolId + domain="portainer". Client injected; tolerant.
"""

from __future__ import annotations

from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "portainer"
_DOMAIN = "portainer"


def _get(config: Any, key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else getattr(config, key, None)


def _call(client: Any, name: str, *a: Any) -> Any:
    m = getattr(client, name, None)
    try:
        return m(*a) if callable(m) else None
    except Exception:  # noqa: BLE001
        return None


def _rows(res: Any) -> list[dict]:
    if isinstance(res, dict):
        res = res.get("data") or res.get("value") or []
    return [r for r in res if isinstance(r, dict)] if isinstance(res, list) else []


def extract(config: Any) -> ExtractionBatch:
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    for ep in _rows(_call(client, "get_endpoints")):
        eid = ep.get("Id") or ep.get("id")
        if eid is None:
            continue
        ep_node = f"portainer_endpoint:{eid}"
        nodes.append(
            GraphNode(
                id=ep_node,
                type="Server",
                props={
                    "name": ep.get("Name") or ep.get("name") or f"endpoint-{eid}",
                    "externalToolId": str(eid),
                    "domain": _DOMAIN,
                },
            )
        )
        for ct in _rows(_call(client, "list_containers", eid)):
            names = ct.get("Names") or [ct.get("name")]
            cname = (
                names[0]
                if isinstance(names, list) and names
                else ct.get("Id") or "container"
            )
            cid = f"portainer_container:{eid}:{ct.get('Id') or cname}"
            nodes.append(
                GraphNode(
                    id=cid,
                    type="AssetInstance",
                    props={
                        "name": str(cname).lstrip("/"),
                        "image": ct.get("Image"),
                        "state": ct.get("State"),
                        "externalToolId": cid.split(":", 1)[1],
                        "domain": _DOMAIN,
                    },
                )
            )
            edges.append(EnrichmentEdge(source=cid, target=ep_node, rel_type="RUNS_ON"))

    for st in _rows(_call(client, "list_stacks")):
        sid = st.get("Id") or st.get("id")
        if sid is None:
            continue
        nodes.append(
            GraphNode(
                id=f"portainer_stack:{sid}",
                type="Service",
                props={
                    "name": st.get("Name") or st.get("name") or f"stack-{sid}",
                    "ci_class": "stack",
                    "externalToolId": str(sid),
                    "domain": _DOMAIN,
                },
            )
        )
    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_extractor(
    CATEGORY, extract, description="Portainer endpoints/stacks/containers → KG"
)
