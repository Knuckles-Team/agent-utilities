"""Shortest-path finder between two arbitrary ontology objects.

CONCEPT:AU-KG.ontology.object-path-finder — closes a gap analysed against
Microsoft's Ontology-Playground (open-source-libraries/Ontology-Playground):
its "Path Finder" panel (``src/lib/pathFinder.ts`` +
``src/components/PathFinderPanel.tsx``) runs a BFS shortest path between two
selected entities and renders the hop-by-hop relationship chain. This
platform already has the identical underlying primitive —
:meth:`~agent_utilities.knowledge_graph.core.graph_compute.GraphComputeEngine.get_shortest_path`
— with exactly one caller (``query_tools.code_connects``), scoped to
``:Code`` symbols only. This module is the same technique made
object-type-agnostic: given any two object ids (e.g. resolved via
``object_set`` search), find the shortest path and annotate each hop with its
connecting relationship type and a friendly label, so any two ontology
objects — not just code — can be traced. See also ``object_set``'s
``search_around``/``pivot`` actions, which answer "what's near X" (bounded-hop
neighborhood); this answers "how does X connect to Y" (the specific chain).

Wired onto the existing ``object_set`` MCP tool as ``action='path'``
(``source_id`` + ``target_id``) and the granular
``GET /api/objects/{source_id}/path/{target_id}`` REST route — see
``agent_utilities/mcp/tools/ontology_tools.py`` and
``agent_utilities/gateway/ontology_api.py``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["find_object_path"]


def find_object_path(engine: Any, source_id: str, target_id: str) -> dict[str, Any]:
    """Find the shortest path between two objects and annotate each hop.

    Tries ``source_id -> target_id`` then the reverse, so an edge that only
    runs one direction in the property graph is still found (mirrors
    ``code_connects``'s undirected-connectivity behavior). Never raises on a
    missing path — returns ``connected: False`` instead, so callers can
    render a "no path found" state exactly like Ontology-Playground's
    PathFinderPanel.
    """
    if source_id == target_id:
        return {
            "source": source_id,
            "target": target_id,
            "connected": False,
            "error": "source and target are the same object",
        }

    path = engine.get_shortest_path(source_id, target_id) or engine.get_shortest_path(
        target_id, source_id
    )
    if not path:
        return {
            "source": source_id,
            "target": target_id,
            "connected": False,
            "path": [],
        }

    # Friendly type/name for every node on the path, resolved in one query.
    labels: dict[str, dict[str, Any]] = {}
    try:
        rows = engine.query_cypher(
            "MATCH (n) WHERE n.id IN $ids RETURN n.id AS id, n.type AS type, n.name AS name",
            {"ids": path},
        )
        for row in rows or []:
            node_id = row.get("id")
            if node_id:
                labels[node_id] = {"type": row.get("type"), "name": row.get("name")}
    except Exception:  # noqa: BLE001 — labeling is best-effort, the path still returns
        pass

    hops: list[dict[str, Any]] = []
    for a, b in zip(path, path[1:], strict=False):
        rel, confidence = None, None
        try:
            erows = engine.query_cypher(
                "MATCH (x {id: $a})-[r]-(y {id: $b}) "
                "RETURN type(r) AS rel, r.confidence AS confidence LIMIT 1",
                {"a": a, "b": b},
            )
            if erows:
                rel = erows[0].get("rel")
                confidence = erows[0].get("confidence")
        except Exception:  # noqa: BLE001 — annotation is best-effort
            pass
        hops.append({"from": a, "to": b, "rel": rel, "confidence": confidence})

    return {
        "source": source_id,
        "target": target_id,
        "connected": True,
        "length": len(path) - 1,
        "path": [{"id": node_id, **labels.get(node_id, {})} for node_id in path],
        "hops": hops,
    }
