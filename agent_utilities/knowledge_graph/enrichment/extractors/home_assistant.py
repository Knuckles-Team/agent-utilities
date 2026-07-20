"""Home Assistant source extractor — device/entity inventory (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Maps HA entities into the uniform ExtractionBatch as :ConfigurationItem (a
homelab/IoT device registry that joins the same CMDB/asset plane as ServiceNow
CIs), stamped ``externalToolId`` (entity_id) + ``domain="homeassistant"``.
Client (``home_assistant_agent.auth.get_client()``) is injected; tolerant of
dict- or object-shaped states.
"""

from __future__ import annotations

from typing import Any

from ..models import ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "homeassistant"
_DOMAIN = "homeassistant"


def _get(config: Any, key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else getattr(config, key, None)


def _attr(obj: Any, *names: str) -> Any:
    for n in names:
        if isinstance(obj, dict) and obj.get(n) not in (None, ""):
            return obj[n]
        v = getattr(obj, n, None)
        if v not in (None, ""):
            return v
    return None


def extract(config: Any) -> ExtractionBatch:
    """Extract Home Assistant entities into a uniform ExtractionBatch."""
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=[])

    getter = getattr(client, "get_states", None)
    if not callable(getter):
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=[])
    try:
        states = getter() or []
    except Exception:  # noqa: BLE001
        states = []

    for st in states:
        eid = _attr(st, "entity_id")
        if not eid:
            continue
        attrs = _attr(st, "attributes") or {}
        name = (attrs.get("friendly_name") if isinstance(attrs, dict) else None) or eid
        nodes.append(
            GraphNode(
                id=f"ha:{eid}",
                type="ConfigurationItem",
                props={
                    "name": name,
                    "state": _attr(st, "state"),
                    "ci_class": str(eid).split(".", 1)[0],  # domain (light/sensor/…)
                    "externalToolId": str(eid),
                    "domain": _DOMAIN,
                },
            )
        )

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=[])


register_extractor(
    CATEGORY, extract, description="Home Assistant entities (device inventory) → KG"
)
