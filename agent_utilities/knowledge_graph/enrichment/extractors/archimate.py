"""ArchiMate model source extractor (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Reads an ArchiMate model (Open-Exchange, via the archimate-mcp ArchiApi) into the
KG: elements → their ArchiMate class node (BusinessProcess, ApplicationComponent,
Node, …), relationships → typed edges. Stamped externalToolId + domain="archimate".
The emitted element types are registered as OWL-promotable so DL reasoning treats
them as first-class (they already have classes in ontology_archimate.ttl). Client
injected; tolerant.
"""

from __future__ import annotations

from typing import Any

from ...ontology.leanix_metamodel import _upper_snake
from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_extractor

CATEGORY = "archimate"
_DOMAIN = "archimate"


def _get(config: Any, key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else getattr(config, key, None)


def _call(client: Any, name: str) -> list[dict]:
    m = getattr(client, name, None)
    try:
        res = m() if callable(m) else None
    except Exception:  # noqa: BLE001
        return []
    return [r for r in res if isinstance(r, dict)] if isinstance(res, list) else []


def extract(config: Any) -> ExtractionBatch:
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    emitted_types: set[str] = set()
    for el in _call(client, "list_elements"):
        eid = el.get("id")
        etype = el.get("type")
        if not (eid and etype):
            continue
        emitted_types.add(etype)
        nodes.append(
            GraphNode(
                id=f"archi:{eid}",
                type=etype,
                props={
                    "name": el.get("name"),
                    "documentation": el.get("documentation"),
                    "externalToolId": str(eid),
                    "domain": _DOMAIN,
                },
            )
        )

    for rel in _call(client, "list_relationships"):
        src, tgt = rel.get("source"), rel.get("target")
        if not (src and tgt):
            continue
        edges.append(
            EnrichmentEdge(
                source=f"archi:{src}",
                target=f"archi:{tgt}",
                rel_type=_upper_snake(rel.get("type") or "ASSOCIATION"),
            )
        )

    # Make the model's element classes OWL-promotable (they have ontology_archimate
    # classes); registering here keeps reasoning first-class without a hub edit.
    if emitted_types:
        from ...core.owl_bridge import register_promotable_node_types

        register_promotable_node_types(emitted_types)

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_extractor(
    CATEGORY, extract, description="ArchiMate model elements/relationships → KG"
)
