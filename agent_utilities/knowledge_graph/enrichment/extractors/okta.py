"""Okta source extractor — users / groups / apps / assignments (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Maps the Okta identity estate into the uniform ExtractionBatch: users →
:IdentityUser, groups → :IdentityGroup, apps → :Application, with MEMBER_OF_GROUP
and ASSIGNED_APP edges. Every node carries ``externalToolId`` + ``domain="okta"``.
The client (``okta_agent.auth.get_client()``) is injected; calls are tolerant.
"""

from __future__ import annotations

from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "okta"
_DOMAIN = "okta"


def _get(config: Any, key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else getattr(config, key, None)


def _first(row: Any, *keys: str) -> Any:
    if not isinstance(row, dict):
        return None
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return v
    return None


def _call(client: Any, name: str, *args: Any) -> list:
    method = getattr(client, name, None)
    if not callable(method):
        return []
    try:
        res = method(*args)
    except Exception:  # noqa: BLE001
        return []
    if isinstance(res, list | tuple):
        return list(res)
    if isinstance(res, dict):
        for key in ("value", "results", "items", "data"):
            if isinstance(res.get(key), list):
                return res[key]
    return []


def _profile(row: dict, *keys: str) -> Any:
    prof = row.get("profile") if isinstance(row, dict) else None
    if isinstance(prof, dict):
        for k in keys:
            if prof.get(k):
                return prof[k]
    return _first(row, *keys)


def extract(config: Any) -> ExtractionBatch:
    """Extract Okta users/groups/apps into a uniform ExtractionBatch."""
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    for u in _call(client, "list_users"):
        uid = _first(u, "id")
        if not uid:
            continue
        nodes.append(
            GraphNode(
                id=f"okta_user:{uid}",
                type="IdentityUser",
                props={
                    "name": _profile(u, "login", "email", "displayName"),
                    "email": _profile(u, "email"),
                    "status": _first(u, "status"),
                    "externalToolId": str(uid),
                    "domain": _DOMAIN,
                },
            )
        )

    for g in _call(client, "list_groups"):
        gid = _first(g, "id")
        if not gid:
            continue
        gnode = f"okta_group:{gid}"
        nodes.append(
            GraphNode(
                id=gnode,
                type="IdentityGroup",
                props={
                    "name": _profile(g, "name"),
                    "externalToolId": str(gid),
                    "domain": _DOMAIN,
                },
            )
        )
        for m in _call(client, "list_group_members", gid):
            mid = _first(m, "id")
            if mid:
                edges.append(
                    EnrichmentEdge(
                        source=f"okta_user:{mid}",
                        target=gnode,
                        rel_type="MEMBER_OF_GROUP",
                    )
                )

    for a in _call(client, "list_apps"):
        aid = _first(a, "id")
        if not aid:
            continue
        nodes.append(
            GraphNode(
                id=f"okta_app:{aid}",
                type="Application",
                props={
                    "name": _first(a, "label", "name"),
                    "status": _first(a, "status"),
                    "externalToolId": str(aid),
                    "domain": _DOMAIN,
                },
            )
        )

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_extractor(
    CATEGORY, extract, description="Okta identity (users/groups/apps) → KG"
)
