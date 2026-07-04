"""Microsoft 365 source extractor (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

The *entity* side of the M365 integration: calendar events → :CalendarEvent and
directory users → :Person (the same canonical classes the Nextcloud extractor
uses, so M365/Nextcloud calendars + people reconcile across sources). Stamped
externalToolId + domain="microsoft".

The Microsoft Graph client is async; it reaches this synchronous extractor via
:class:`~agent_utilities.knowledge_graph.enrichment.source_adapters.MicrosoftGraphSourceClient`
(injected by ``resolve_source_client``), which bridges each call to sync. All
calls are tolerant — a missing surface yields no nodes rather than an error.
"""

from __future__ import annotations

from typing import Any

from ...core import owl_bridge
from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "microsoft"
_DOMAIN = "microsoft"


def _get(config: Any, key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else getattr(config, key, None)


def _nested(row: dict, *path: str) -> Any:
    """Pull a value at a dotted Graph path, e.g. start.dateTime / location.displayName."""
    cur: Any = row
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _call(client: Any, name: str) -> list[dict]:
    m = getattr(client, name, None)
    try:
        return list(m() or []) if callable(m) else []
    except Exception:  # noqa: BLE001 - tolerant: degrade to no rows
        return []


def extract(config: Any) -> ExtractionBatch:
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    for ev in _call(client, "calendar_events"):
        eid = ev.get("id")
        if not eid:
            continue
        props = {
            "name": ev.get("subject"),
            "scheduledStart": _nested(ev, "start", "dateTime"),
            "scheduledEnd": _nested(ev, "end", "dateTime"),
            "eventLocation": _nested(ev, "location", "displayName"),
            "externalToolId": str(eid),
            "domain": _DOMAIN,
        }
        nodes.append(
            GraphNode(
                id=f"msevent:{eid}",
                type="CalendarEvent",
                props={k: v for k, v in props.items() if v is not None},
            )
        )

    for u in _call(client, "users"):
        uid = u.get("id")
        if not uid:
            continue
        nodes.append(
            GraphNode(
                id=f"msuser:{uid}",
                type="Person",
                props={
                    "name": u.get("displayName"),
                    "email": u.get("mail") or u.get("userPrincipalName"),
                    "externalToolId": str(uid),
                    "domain": _DOMAIN,
                },
            )
        )

    owl_bridge.register_promotable_node_types({n.type for n in nodes})
    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_extractor(
    CATEGORY, extract, description="Microsoft 365 (calendar events + users) → KG"
)
