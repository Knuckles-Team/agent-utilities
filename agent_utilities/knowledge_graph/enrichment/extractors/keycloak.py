"""Keycloak source extractor — realm users / groups / clients (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Maps a Keycloak realm into the uniform ExtractionBatch: users → :IdentityUser,
groups → :IdentityGroup, clients → :Application, stamped ``externalToolId`` +
``domain="keycloak"``. The realm comes from ``config['realm']`` (default
``master``). Client (``keycloak_agent.auth.get_client()``) is injected; tolerant.
"""

from __future__ import annotations

from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "keycloak"
_DOMAIN = "keycloak"


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


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
    return list(res) if isinstance(res, list | tuple) else []


def extract(config: Any) -> ExtractionBatch:
    """Extract a Keycloak realm's users/groups/clients into an ExtractionBatch."""
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)
    realm = _get(config, "realm", "master")

    for u in _call(client, "list_users", realm):
        uid = _first(u, "id", "username")
        if not uid:
            continue
        nodes.append(
            GraphNode(
                id=f"kc_user:{uid}",
                type="IdentityUser",
                props={
                    "name": _first(u, "username", "email"),
                    "email": _first(u, "email"),
                    "enabled": _first(u, "enabled"),
                    "realm": realm,
                    "externalToolId": str(uid),
                    "domain": _DOMAIN,
                },
            )
        )

    for g in _call(client, "list_groups", realm) or _call(client, "get_groups", realm):
        gid = _first(g, "id", "name")
        if not gid:
            continue
        nodes.append(
            GraphNode(
                id=f"kc_group:{gid}",
                type="IdentityGroup",
                props={
                    "name": _first(g, "name"),
                    "realm": realm,
                    "externalToolId": str(gid),
                    "domain": _DOMAIN,
                },
            )
        )

    for c in _call(client, "list_clients", realm) or _call(
        client, "get_clients", realm
    ):
        cid = _first(c, "id", "clientId")
        if not cid:
            continue
        nodes.append(
            GraphNode(
                id=f"kc_client:{cid}",
                type="Application",
                props={
                    "name": _first(c, "clientId", "name"),
                    "realm": realm,
                    "externalToolId": str(cid),
                    "domain": _DOMAIN,
                },
            )
        )

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_extractor(
    CATEGORY, extract, description="Keycloak realm (users/groups/clients) → KG"
)
