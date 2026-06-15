"""Nextcloud source extractor — calendar events + contacts (CONCEPT:KG-2.9).

The *entity* side of the Nextcloud integration (documents flow through the
document pipeline via the ``nextcloud-files`` source preset, not here). Maps
CalDAV calendar events → :CalendarEvent and CardDAV contacts → :Person, each
stamped ``externalToolId`` + ``domain="nextcloud"`` for write-back resolution.

The client is **injected** (``nextcloud_agent.auth.get_client()`` via
``resolve_source_client``); all calls are tolerant — a missing surface yields no
nodes rather than an error. No network here beyond delegating to the client.
"""

from __future__ import annotations

from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "nextcloud"
_DOMAIN = "nextcloud"


def _get(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


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
        return list(method(*args) or [])
    except Exception:  # noqa: BLE001 - tolerant: degrade to no rows
        return []


def extract(config: Any) -> ExtractionBatch:
    """Extract Nextcloud calendar events + contacts into a uniform ExtractionBatch."""
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    seen: set[str] = set()

    # Calendars → events.
    for cal in _call(client, "list_calendars"):
        cal_url = _first(cal, "url", "href")
        if not cal_url:
            continue
        for ev in _call(client, "list_events", cal_url):
            uid = _first(ev, "uid", "id", "href")
            if not uid:
                continue
            nid = f"event:{uid}"
            if nid in seen:
                continue
            seen.add(nid)
            props = {
                "name": _first(ev, "summary", "title", "name"),
                "scheduledStart": _first(ev, "start", "dtstart", "DTSTART"),
                "scheduledEnd": _first(ev, "end", "dtend", "DTEND"),
                "eventLocation": _first(ev, "location", "LOCATION"),
                "calendar": cal_url,
                "externalToolId": str(uid),
                "domain": _DOMAIN,
            }
            nodes.append(
                GraphNode(
                    id=nid,
                    type="CalendarEvent",
                    props={k: v for k, v in props.items() if v is not None},
                )
            )

    # Address books → contacts.
    for ab in _call(client, "list_address_books"):
        ab_url = _first(ab, "url", "href")
        if not ab_url:
            continue
        for c in _call(client, "list_contacts", ab_url):
            uid = _first(c, "uid", "id", "href")
            name = _first(c, "fn", "full_name", "name")
            if not (uid and name):
                continue
            nid = f"contact:{uid}"
            if nid in seen:
                continue
            seen.add(nid)
            nodes.append(
                GraphNode(
                    id=nid,
                    type="Person",
                    props={
                        "name": name,
                        "email": _first(c, "email", "EMAIL"),
                        "externalToolId": str(uid),
                        "domain": _DOMAIN,
                    },
                )
            )

    return ExtractionBatch(
        category=CATEGORY,
        nodes=nodes,
        edges=edges,
    )


register_extractor(
    CATEGORY, extract, description="Nextcloud calendar events + contacts → KG"
)
