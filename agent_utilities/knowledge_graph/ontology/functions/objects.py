#!/usr/bin/python
from __future__ import annotations

"""Functions-on-Objects (CONCEPT:KG-2.41).

Palantir Foundry ``functions/overview`` defines *Functions-on-Objects*: typed
functions that read an object's properties, traverse its links, and aggregate
over sets of objects in the ontology. This module provides the real graph-read
primitives those functions stand on, composed over the live
:class:`~agent_utilities.knowledge_graph.facade.KnowledgeGraph` facade.

The single store interface every backend exposes is ``execute(cypher, params)``
(see ``knowledge_graph/backends/base.py``), so every helper here is expressed as
a parameterized, read-only Cypher query routed through the facade's guarded
:meth:`KnowledgeGraph.query` (permission-filtered + audited when enforcement is
on). When no backend is reachable the facade returns ``[]`` and these helpers
degrade to empty results — the logic is real, the failure is graceful.
"""

import logging
import re
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent_utilities.knowledge_graph.facade import KnowledgeGraph

# Cypher relationship types cannot be passed as a bind parameter — they are
# part of the query structure — so a relationship label is admitted into the
# query string only if it matches a strict identifier allowlist (letters,
# digits and underscores, not starting with a digit). Anything else is
# rejected, which removes the string-built-Cypher injection surface while
# leaving every valid label working exactly as before.
_REL_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _rel_label(rel_type: str | None) -> str:
    """Return a safe ``:LABEL`` fragment for ``rel_type`` (or ``""`` if unset).

    Raises:
        ValueError: if ``rel_type`` is not a valid Cypher identifier.
    """
    if not rel_type:
        return ""
    if not _REL_TYPE_RE.match(rel_type):
        raise ValueError(f"invalid relationship type {rel_type!r}")
    return f":{rel_type}"


def _facade(graph: Any) -> Any:
    """Return a usable KnowledgeGraph facade, constructing a default if needed."""
    if graph is not None:
        return graph
    try:
        from agent_utilities.knowledge_graph.facade import KnowledgeGraph

        return KnowledgeGraph()
    except Exception as exc:  # noqa: BLE001 — offline/no-backend degrades cleanly
        logger.debug("ObjectFunctionContext: no facade available: %s", exc)
        return None


def _run(graph: Any, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Run a read query through the facade's guarded path; ``[]`` when offline."""
    if graph is None:
        return []
    try:
        return graph.query(cypher, params) or []
    except Exception as exc:  # noqa: BLE001 — degrade gracefully, never raise
        logger.debug("Functions-on-Objects read failed: %s", exc)
        return []


class ObjectFunctionContext:
    """Read/traverse/aggregate primitives for Functions-on-Objects. CONCEPT:KG-2.41.

    Constructed over a :class:`KnowledgeGraph` facade (or, when omitted, a fresh
    default facade — which still degrades cleanly with no backend). Every method
    is read-only and returns plain data so a :class:`FunctionSpec` handler can be
    a pure callable over these results.

    Args:
        graph: The live :class:`KnowledgeGraph` facade, or ``None`` to build a
            default one lazily.
    """

    def __init__(self, graph: KnowledgeGraph | None = None) -> None:
        self.graph = _facade(graph)

    # ── Property reads ─────────────────────────────────────────────────

    def get_object(self, object_id: str) -> dict[str, Any]:
        """Return the full property map of one object, or ``{}`` if not found."""
        rows = _run(
            self.graph,
            "MATCH (n {id: $id}) RETURN n",
            {"id": object_id},
        )
        if not rows:
            return {}
        return _node_props(rows[0])

    def get_property(self, object_id: str, name: str, default: Any = None) -> Any:
        """Return a single named property of an object (or ``default``)."""
        props = self.get_object(object_id)
        return props.get(name, default)

    # ── Link traversal ─────────────────────────────────────────────────

    def neighbors(
        self,
        object_id: str,
        *,
        rel_type: str | None = None,
        direction: str = "out",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return the 1-hop linked objects of ``object_id``.

        Args:
            rel_type: Optional relationship label to filter on (e.g. ``"DEPENDS_ON"``).
            direction: ``"out"`` (default), ``"in"``, or ``"both"``.
            limit: Maximum number of neighbors returned.
        """
        try:
            rel = _rel_label(rel_type)
        except ValueError as exc:
            logger.debug("neighbors: %s", exc)
            return []
        if direction == "in":
            pattern = f"(n {{id: $id}})<-[{rel}]-(m)"
        elif direction == "both":
            pattern = f"(n {{id: $id}})-[{rel}]-(m)"
        else:
            pattern = f"(n {{id: $id}})-[{rel}]->(m)"
        rows = _run(
            self.graph,
            f"MATCH {pattern} RETURN m LIMIT {int(limit)}",
            {"id": object_id},
        )
        return [_node_props(r) for r in rows]

    def traverse(
        self,
        object_id: str,
        *,
        hops: int = 1,
        rel_type: str | None = None,
        direction: str = "out",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Return objects reachable within ``hops`` of ``object_id`` (N-hop).

        A real variable-length traversal: walks 1..``hops`` edges (optionally
        filtered to ``rel_type`` and a direction) and returns the distinct
        endpoint objects. Degrades to ``[]`` with no backend.
        """
        hops = max(1, int(hops))
        try:
            rel = _rel_label(rel_type)
        except ValueError as exc:
            logger.debug("traverse: %s", exc)
            return []
        varlen = f"[{rel}*1..{hops}]"
        if direction == "in":
            pattern = f"(n {{id: $id}})<-{varlen}-(m)"
        elif direction == "both":
            pattern = f"(n {{id: $id}})-{varlen}-(m)"
        else:
            pattern = f"(n {{id: $id}})-{varlen}->(m)"
        rows = _run(
            self.graph,
            f"MATCH {pattern} RETURN DISTINCT m LIMIT {int(limit)}",
            {"id": object_id},
        )
        return [_node_props(r) for r in rows]

    # ── Aggregation over object sets ───────────────────────────────────

    def object_set(
        self,
        *,
        object_type: str | None = None,
        ids: list[str] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Materialize a set of objects by type and/or explicit id list."""
        if ids:
            rows = _run(
                self.graph,
                f"MATCH (n) WHERE n.id IN $ids RETURN n LIMIT {int(limit)}",
                {"ids": list(ids)},
            )
        elif object_type:
            rows = _run(
                self.graph,
                f"MATCH (n {{type: $t}}) RETURN n LIMIT {int(limit)}",
                {"t": object_type},
            )
        else:
            rows = _run(
                self.graph,
                f"MATCH (n) RETURN n LIMIT {int(limit)}",
                {},
            )
        return [_node_props(r) for r in rows]

    def aggregate(
        self,
        objects: list[dict[str, Any]],
        property_name: str,
        op: str = "sum",
    ) -> float:
        """Run a numeric aggregate over one property across an object set.

        ``op`` is one of ``sum``/``mean``/``min``/``max``/``count``. Non-numeric
        and missing values are ignored (``count`` counts numeric occurrences).
        Reuses the registry's built-in numeric aggregate so the math lives in
        exactly one place.
        """
        from .registry import _numeric_aggregate

        values = [o.get(property_name) for o in objects]
        return _numeric_aggregate(values, op=op)

    def aggregate_links(
        self,
        object_id: str,
        property_name: str,
        *,
        rel_type: str | None = None,
        direction: str = "out",
        op: str = "sum",
    ) -> float:
        """Aggregate a property over the objects linked to ``object_id``.

        The composition that makes this a Functions-on-Objects primitive:
        traverse 1-hop links, then run :meth:`aggregate` over the neighbors.
        """
        neighbors = self.neighbors(object_id, rel_type=rel_type, direction=direction)
        return self.aggregate(neighbors, property_name, op=op)


def _node_props(row: dict[str, Any]) -> dict[str, Any]:
    """Extract a flat property dict from a single-column ``RETURN n`` row.

    Backends return rows shaped either ``{"n": {...props...}}`` (node bound to a
    variable) or already-flattened ``{...props...}``. Both are normalized to a
    plain property mapping.
    """
    if not isinstance(row, dict):
        return {}
    if len(row) == 1:
        (only,) = row.values()
        if isinstance(only, dict):
            inner = only.get("properties")
            if isinstance(inner, dict):
                return dict(inner)
            return dict(only)
    # Already-flat row, or a multi-column row — return as-is.
    return dict(row)


__all__ = ["ObjectFunctionContext"]
