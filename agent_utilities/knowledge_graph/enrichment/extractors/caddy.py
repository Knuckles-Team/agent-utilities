"""Caddy reverse-proxy source extractor (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

HTTP server routes → :Service (the reverse-proxy front door for an app). Stamped
externalToolId + domain="caddy". Client injected; tolerant of the Caddy JSON config.
"""

from __future__ import annotations

from typing import Any

from ..models import ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "caddy"
_DOMAIN = "caddy"


def _get(config: Any, key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else getattr(config, key, None)


def _hosts(route: dict) -> list[str]:
    hosts: list[str] = []
    for m in route.get("match", []) or []:
        if isinstance(m, dict):
            hosts.extend(m.get("host", []) or [])
    return hosts


def extract(config: Any) -> ExtractionBatch:
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=[])
    getter = getattr(client, "get_config", None)
    try:
        cfg = getter("") if callable(getter) else None
    except Exception:  # noqa: BLE001
        cfg = None
    if not isinstance(cfg, dict):
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=[])

    servers = ((cfg.get("apps") or {}).get("http") or {}).get("servers") or {}
    for sname, srv in servers.items() if isinstance(servers, dict) else []:
        for i, route in enumerate(srv.get("routes", []) or []):
            if not isinstance(route, dict):
                continue
            hosts = _hosts(route)
            label = hosts[0] if hosts else f"{sname}-route-{i}"
            rid = f"caddy_route:{sname}:{label}"
            nodes.append(
                GraphNode(
                    id=rid,
                    type="Service",
                    props={
                        "name": label,
                        "ci_class": "reverse_proxy_route",
                        "hosts": ",".join(hosts) or None,
                        "server": sname,
                        "externalToolId": rid.split(":", 1)[1],
                        "domain": _DOMAIN,
                    },
                )
            )
    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=[])


register_extractor(CATEGORY, extract, description="Caddy reverse-proxy routes → KG")
