"""LeanIX Enterprise Architecture source extractor (CONCEPT:KG-2.9).

Ingests LeanIX factsheets (ArchiMate-style EA inventory) into the KG via the
uniform :class:`ExtractionBatch` shape, so this enterprise source needs **no**
edits to any shared hub file. The LeanIX API client is *injected* via
``config.client`` (duck-typed) — this module performs **no** network I/O.

Mapping (LeanIX factsheet → GraphNode)::

    Application         -> Application        id=app:{id}
    ITComponent         -> ITComponent        id=itcomponent:{id}
    BusinessCapability  -> BusinessCapability id=capability:{id}
    DataObject          -> DataObject         id=dataobject:{id}

Edges (derived from factsheet relation fields, tolerant of shape)::

    Application -SUPPORTS->   BusinessCapability  (relApplicationToBusinessCapability)
    Application -DEPENDS_ON-> ITComponent         (relApplicationToITComponent)

Contract: ``def extract(config) -> ExtractionBatch`` where ``config.client``
exposes ``client.factsheets(type=...)`` or ``client.factsheets()`` returning
lists of dicts. Each dict carries ``id``, ``name``, ``type`` and (optionally)
relation fields. The extractor self-registers at import time.
"""

from __future__ import annotations

from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_source

CATEGORY = "leanix"

# LeanIX factsheet type -> (node type label, id prefix)
_TYPE_MAP: dict[str, tuple[str, str]] = {
    "Application": ("Application", "app"),
    "ITComponent": ("ITComponent", "itcomponent"),
    "BusinessCapability": ("BusinessCapability", "capability"),
    "DataObject": ("DataObject", "dataobject"),
}

_FACTSHEET_TYPES = list(_TYPE_MAP.keys())


def _node_id(fs_type: str, raw_id: str) -> str:
    prefix = _TYPE_MAP.get(fs_type, (fs_type, fs_type.lower()))[1]
    return f"{prefix}:{raw_id}"


def _iter_related_ids(value: Any) -> list[str]:
    """Pull target factsheet ids from a tolerant relation field.

    Accepts a bare id string, a dict (id / factSheetId / target{id}), a list of
    any of those, or LeanIX-style ``{"edges": [{"node": {...}}]}``.
    """
    ids: list[str] = []

    def _one(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            if item:
                ids.append(item)
            return
        if isinstance(item, dict):
            # LeanIX GraphQL relation envelopes.
            if "edges" in item and isinstance(item["edges"], list):
                for edge in item["edges"]:
                    _one(edge)
                return
            if "node" in item:
                _one(item["node"])
                return
            target = item.get("target")
            if isinstance(target, dict):
                _one(target)
                return
            for key in ("factSheetId", "id", "targetId"):
                val = item.get(key)
                if isinstance(val, str) and val:
                    ids.append(val)
                    return
            return
        if isinstance(item, list | tuple):
            for sub in item:
                _one(sub)

    _one(value)
    # de-dupe, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _collect_factsheets(client: Any) -> list[dict]:
    """Fetch factsheets from the injected client, tolerant of its surface."""
    sheets: list[dict] = []
    factsheets = getattr(client, "factsheets", None)
    if not callable(factsheets):
        return sheets
    # Prefer per-type queries; fall back to a single unfiltered call.
    try:
        for fs_type in _FACTSHEET_TYPES:
            result = factsheets(type=fs_type)
            for item in result or []:
                if isinstance(item, dict):
                    item.setdefault("type", fs_type)
                    sheets.append(item)
        if sheets:
            return sheets
    except TypeError:
        pass  # client.factsheets() takes no `type` kwarg
    result = factsheets()
    for item in result or []:
        if isinstance(item, dict):
            sheets.append(item)
    return sheets


def extract(config: Any) -> ExtractionBatch:
    """Extract LeanIX EA factsheets into a uniform ExtractionBatch.

    ``config.client`` is the injected (duck-typed) LeanIX API client. No network
    access happens here beyond delegating to that client.
    """
    client = getattr(config, "client", None) or getattr(config, "get", lambda *_: None)(
        "client"
    )
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    seen_nodes: set[str] = set()
    for fs in _collect_factsheets(client):
        raw_id = fs.get("id")
        fs_type = fs.get("type")
        if not raw_id or fs_type not in _TYPE_MAP:
            continue
        label = _TYPE_MAP[fs_type][0]
        nid = _node_id(fs_type, raw_id)
        if nid not in seen_nodes:
            seen_nodes.add(nid)
            nodes.append(GraphNode(id=nid, type=label, props={"name": fs.get("name")}))

        if fs_type != "Application":
            continue

        # Application -SUPPORTS-> BusinessCapability
        for cap_id in _iter_related_ids(fs.get("relApplicationToBusinessCapability")):
            edges.append(
                EnrichmentEdge(
                    source=nid,
                    target=_node_id("BusinessCapability", cap_id),
                    rel_type="SUPPORTS",
                )
            )
        # Application -DEPENDS_ON-> ITComponent
        for comp_id in _iter_related_ids(fs.get("relApplicationToITComponent")):
            edges.append(
                EnrichmentEdge(
                    source=nid,
                    target=_node_id("ITComponent", comp_id),
                    rel_type="DEPENDS_ON",
                )
            )

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_source(CATEGORY, extract, description="LeanIX EA factsheets → KG")
