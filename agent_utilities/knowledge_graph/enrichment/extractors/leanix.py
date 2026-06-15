"""LeanIX Enterprise Architecture source extractor (CONCEPT:KG-2.9).

Mirrors the LeanIX fact-sheet graph into the KG via the uniform
:class:`ExtractionBatch` shape, so this enterprise source needs **no** edits to
any shared hub file. The LeanIX client is *injected* via ``config.client``
(duck-typed) — this module performs **no** network I/O.

The type/relation vocabulary is **discovered from the live metamodel** (reusing
the metamodel→OWL compiler), so every fact sheet type and every ``rel*`` relation
is mirrored — not just a hardcoded handful. When the metamodel is unavailable the
extractor falls back to the well-known core EA types so it still works offline.

Mapping (LeanIX fact sheet → GraphNode)::

    <FactSheetType>  ->  node type == <FactSheetType>, id = <prefix>:<leanix id>

Every node carries ``externalToolId`` (the LeanIX id) and ``domain="leanix"`` —
the federation key the write-back layer (Piece 4) resolves against. Edges are
derived from every relation field, typed ``UPPER_SNAKE`` of the relation name.

Contract: ``def extract(config) -> ExtractionBatch`` where ``config.client``
exposes ``meta_model()`` and ``factsheets(type=..., since=...)``. ``config.since``
(optional) narrows to a watermark for delta sync. The extractor self-registers
at import time.
"""

from __future__ import annotations

from typing import Any

from ...ontology.leanix_metamodel import _upper_snake, compile_leanix_metamodel
from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "leanix"

# Fallback when the live metamodel can't be discovered: the well-known core EA
# types (label, id-prefix) and their canonical relations (lpg_rel_type, target).
_FALLBACK_TYPE_MAP: dict[str, tuple[str, str]] = {
    "Application": ("Application", "app"),
    "ITComponent": ("ITComponent", "itcomponent"),
    "BusinessCapability": ("BusinessCapability", "capability"),
    "DataObject": ("DataObject", "dataobject"),
}
_FALLBACK_RELATION_MAP: dict[str, tuple[str, str]] = {
    "relApplicationToBusinessCapability": ("SUPPORTS", "BusinessCapability"),
    "relApplicationToITComponent": ("DEPENDS_ON", "ITComponent"),
}


def _discover_maps(
    client: Any,
) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]]]:
    """Build type/relation maps from the live metamodel, or fall back."""
    meta: dict[str, Any] = {}
    mm = getattr(client, "meta_model", None)
    if callable(mm):
        try:
            meta = mm() or {}
        except Exception:  # noqa: BLE001 - tolerant: degrade to fallback
            meta = {}
    if meta:
        spec = compile_leanix_metamodel(meta)
        if spec.type_map:
            return spec.type_map, spec.relation_map
    return dict(_FALLBACK_TYPE_MAP), dict(_FALLBACK_RELATION_MAP)


def _iter_related_targets(value: Any) -> list[tuple[str, str | None]]:
    """Pull ``(target_id, target_type)`` pairs from a tolerant relation field.

    Accepts a bare id, a dict (``factSheetId`` / ``id`` / ``factSheet{id,type}`` /
    ``target{...}``), a list of any of those, or LeanIX-style
    ``{"edges":[{"node":{"factSheet":{...}}}]}``.
    """
    out: list[tuple[str, str | None]] = []

    def _one(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            if item:
                out.append((item, None))
            return
        if isinstance(item, dict):
            edges = item.get("edges")
            if isinstance(edges, list):
                for edge in edges:
                    _one(edge)
                return
            if "node" in item:
                _one(item["node"])
                return
            fsd = item.get("factSheet")
            if isinstance(fsd, dict):
                fid = fsd.get("id")
                if fid:
                    out.append((str(fid), fsd.get("type")))
                return
            target = item.get("target")
            if isinstance(target, dict):
                _one(target)
                return
            for key in ("factSheetId", "id", "targetId"):
                val = item.get(key)
                if isinstance(val, str) and val:
                    out.append((val, item.get("type")))
                    return
            return
        if isinstance(item, list | tuple):
            for sub in item:
                _one(sub)

    _one(value)
    # de-dupe by id, preserve order
    seen: set[str] = set()
    deduped: list[tuple[str, str | None]] = []
    for tid, ttype in out:
        if tid not in seen:
            seen.add(tid)
            deduped.append((tid, ttype))
    return deduped


def _collect_factsheets(
    client: Any, fs_types: list[str], since: str | None
) -> list[dict]:
    """Fetch fact sheets per type from the injected client, tolerant of its surface."""
    sheets: list[dict] = []
    factsheets = getattr(client, "factsheets", None)
    if not callable(factsheets):
        return sheets
    try:
        for fs_type in fs_types:
            result = factsheets(type=fs_type, since=since)
            for item in result or []:
                if isinstance(item, dict):
                    item.setdefault("type", fs_type)
                    sheets.append(item)
        return sheets
    except TypeError:
        # Client without type/since kwargs → single unfiltered call.
        result = factsheets()
        return [item for item in (result or []) if isinstance(item, dict)]


def extract(config: Any) -> ExtractionBatch:
    """Extract the LeanIX EA fact-sheet graph into a uniform ExtractionBatch."""
    client = getattr(config, "client", None) or getattr(config, "get", lambda *_: None)(
        "client"
    )
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    since = getattr(config, "since", None)
    type_map, relation_map = _discover_maps(client)

    def _node_id(fs_type: str, raw_id: str) -> str:
        prefix = type_map.get(fs_type, (fs_type, fs_type.lower()))[1]
        return f"{prefix}:{raw_id}"

    seen_nodes: set[str] = set()
    for fs in _collect_factsheets(client, list(type_map), since):
        raw_id = fs.get("id")
        fs_type = fs.get("type")
        if not raw_id or fs_type not in type_map:
            continue
        label = type_map[fs_type][0]
        nid = _node_id(fs_type, raw_id)
        if nid not in seen_nodes:
            seen_nodes.add(nid)
            nodes.append(
                GraphNode(
                    id=nid,
                    type=label,
                    props={
                        "name": fs.get("name"),
                        "externalToolId": raw_id,
                        "domain": "leanix",
                    },
                )
            )

        # Every relation field becomes typed edges to the related fact sheets.
        for key, value in fs.items():
            if not isinstance(key, str) or not key.startswith("rel"):
                continue
            mapped = relation_map.get(key)
            rel_type = mapped[0] if mapped else _upper_snake(key)
            default_target_type = mapped[1] if mapped else None
            for tid, ttype in _iter_related_targets(value):
                target_type = ttype or default_target_type
                if not target_type:
                    continue
                edges.append(
                    EnrichmentEdge(
                        source=nid,
                        target=_node_id(target_type, tid),
                        rel_type=rel_type,
                    )
                )

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_extractor(CATEGORY, extract, description="LeanIX EA factsheets → KG")
