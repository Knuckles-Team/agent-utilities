"""Ontology schema graph — render the live interface + link-type registries as
an interactive node/edge graph, and as a Markdown summary.

CONCEPT:AU-KG.ontology.schema-graph-visualization — closes the gap analysed
against Microsoft's Ontology-Playground (open-source-libraries/Ontology-Playground):
that project's headline feature is "any ontology renders as an interactive
node-and-edge diagram" (a Cytoscape.js canvas) plus a one-click Markdown
export of the whole schema. We already have the *schema itself*, live, at
import (:class:`~agent_utilities.knowledge_graph.ontology.interfaces.InterfaceRegistry`
+ :class:`~agent_utilities.knowledge_graph.ontology.links.LinkTypeRegistry`) but no
rendering of it as a graph payload or a human-readable document. This module is
a pure, read-only PROJECTION over those existing registries — it adds no new
ontology storage and never mutates the registries (extends nothing; "extend
the canonical ontology, never sprawl a new .ttl" is respected because there is
no new ontology content here, only a new *view* of the existing one).

Two node kinds mirror Foundry's own Interface/Object-Type distinction (the
same distinction ``ontology_interface`` already exposes via its ``list`` /
``implementers`` actions):

* ``interface`` — an abstract shape contract (:class:`Interface`), with its
  effective (inheritance-resolved) properties.
* ``object_type`` — a concrete registry node type that either implements an
  interface or appears as a link's source/target. Concrete types have no
  separate property registry in this system (properties are schema-on-read on
  the live graph), so an ``object_type`` node carries only its id/label plus
  the interfaces it implements.

Edges come in three kinds: ``implements`` (object_type -> interface),
``extends`` (interface -> parent interface, inheritance), and
``relationship`` (object_type -> object_type, one registered
:class:`~agent_utilities.knowledge_graph.ontology.links.LinkType`, carrying its
cardinality).

The returned graph is shaped like a Cytoscape.js elements list
(``{"nodes": [{"data": {...}}], "edges": [{"data": {...}}]}``) so any
graph-visualization frontend (e.g. agent-webui's sigma.js-backed
``GraphView``/``VertexView``, see ``kg-webui-graphviz``) can render it with no
further transformation.

Wired onto the existing ``ontology_interface`` MCP tool (actions ``graph`` /
``summary`` / ``lint``) and the granular ``/api/ontology/schema-graph`` +
``/api/ontology/schema-summary`` REST routes — see
``agent_utilities/mcp/tools/ontology_tools.py`` and
``agent_utilities/gateway/ontology_api.py``.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .interfaces import InterfaceRegistry
from .links import LinkTypeRegistry

__all__ = ["build_schema_graph", "render_schema_markdown"]

# A small, deterministic, brand-neutral palette — swapped per-deployment theme
# by a consuming frontend if desired; this only needs to give visually distinct
# nodes out of the box, mirroring Ontology-Playground's per-entity color dots.
_PALETTE: tuple[str, ...] = (
    "#0078D4",
    "#107C10",
    "#5C2D91",
    "#D83B01",
    "#00A9E0",
    "#B4009E",
    "#008272",
    "#986F0B",
)


def _color_for(name: str) -> str:
    """Deterministic palette color for a node id (stable across renders)."""
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return _PALETTE[int(digest, 16) % len(_PALETTE)]


def build_schema_graph(
    interface_registry: InterfaceRegistry,
    link_registry: LinkTypeRegistry,
) -> dict[str, Any]:
    """Compose the registered interfaces + link types into a node/edge graph.

    Pure and read-only: iterates ``interface_registry.list_interfaces()`` and
    ``link_registry.list_links()`` and never touches the live KG engine, so it
    works offline/registry-only exactly like the sibling ``list``/``owl``
    actions on ``ontology_interface``.
    """
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def _ensure_object_type(type_name: str) -> None:
        if type_name in nodes:
            return
        nodes[type_name] = {
            "data": {
                "id": type_name,
                "label": type_name,
                "kind": "object_type",
                "color": _color_for(type_name),
            }
        }

    for iface in interface_registry.list_interfaces():
        effective_props = iface.all_properties(interface_registry)
        nodes[iface.name] = {
            "data": {
                "id": iface.name,
                "label": iface.name,
                "kind": "interface",
                "description": iface.description,
                "color": _color_for(iface.name),
                "properties": [
                    {
                        "name": prop.name,
                        "type": prop.type_ref,
                        "required": prop.required,
                        "description": prop.description,
                    }
                    for prop in effective_props.values()
                ],
            }
        }
        for parent_name in iface.extends:
            edges.append(
                {
                    "data": {
                        "id": f"extends::{iface.name}::{parent_name}",
                        "source": iface.name,
                        "target": parent_name,
                        "kind": "extends",
                        "label": "extends",
                    }
                }
            )
        for implementer in interface_registry.find_implementers(iface.name):
            _ensure_object_type(implementer)
            edges.append(
                {
                    "data": {
                        "id": f"implements::{implementer}::{iface.name}",
                        "source": implementer,
                        "target": iface.name,
                        "kind": "implements",
                        "label": "implements",
                    }
                }
            )

    for link in link_registry.list_links():
        source, target = str(link.source_type), str(link.target_type)
        _ensure_object_type(source)
        _ensure_object_type(target)
        edges.append(
            {
                "data": {
                    "id": f"relationship::{link.name}",
                    "source": source,
                    "target": target,
                    "kind": "relationship",
                    "label": link.name,
                    "cardinality": str(link.cardinality),
                    "edge_type": str(link.edge_type),
                }
            }
        )

    interface_count = sum(1 for n in nodes.values() if n["data"]["kind"] == "interface")
    object_type_count = len(nodes) - interface_count
    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "counts": {
            "interfaces": interface_count,
            "object_types": object_type_count,
            "edges": len(edges),
        },
    }


def render_schema_markdown(
    graph: dict[str, Any], *, title: str = "Ontology Schema"
) -> str:
    """Render a :func:`build_schema_graph` payload as a Markdown document.

    Mirrors Ontology-Playground's "Ontology Summary" export (a copy-pasteable
    Markdown doc of every entity + relationship) — useful for docs, PR
    descriptions, or priming an LLM's context with the platform's own schema.
    """
    interfaces = [n["data"] for n in graph["nodes"] if n["data"]["kind"] == "interface"]
    object_types = [
        n["data"] for n in graph["nodes"] if n["data"]["kind"] == "object_type"
    ]
    relationships = [
        e["data"] for e in graph["edges"] if e["data"]["kind"] == "relationship"
    ]
    extensions = [e["data"] for e in graph["edges"] if e["data"]["kind"] == "extends"]

    lines: list[str] = [f"# {title}", ""]
    lines.append(
        f"{len(interfaces)} interfaces, {len(object_types)} object types, "
        f"{len(relationships)} relationships."
    )
    lines.append("")

    lines.append("## Interfaces")
    lines.append("")
    for iface in sorted(interfaces, key=lambda i: i["label"]):
        lines.append(f"### {iface['label']}")
        if iface.get("description"):
            lines.append(iface["description"])
        parents = sorted(e["target"] for e in extensions if e["source"] == iface["id"])
        if parents:
            lines.append("")
            lines.append(f"_Extends: {', '.join(parents)}_")
        properties = iface.get("properties") or []
        if properties:
            lines.append("")
            lines.append("**Properties:**")
            for prop in properties:
                marker = "" if prop["required"] else " (optional)"
                desc = f": {prop['description']}" if prop.get("description") else ""
                lines.append(f"- **{prop['name']}** ({prop['type']}){marker}{desc}")
        lines.append("")

    lines.append("## Relationships")
    lines.append("")
    if relationships:
        for rel in sorted(relationships, key=lambda r: r["label"]):
            lines.append(
                f"- **{rel['label']}**: {rel['source']} → {rel['target']} "
                f"({rel['cardinality']})"
            )
    else:
        lines.append("_No registered link types._")
    lines.append("")

    return "\n".join(lines)
