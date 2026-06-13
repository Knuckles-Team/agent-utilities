#!/usr/bin/python
from __future__ import annotations

"""Object Set Service — search / filter / aggregate / traverse over object sets.

CONCEPT:KG-2.45 — Object Set Service

Palantir provenance: the Foundry *object-backend* **Object Set Service** and the
*object-explorer* surface. An *object set* is a first-class, composable handle on
a collection of ontology objects that you can:

  - **materialize** three ways: a :attr:`ObjectSetKind.STATIC` set of fixed ids; a
    :attr:`ObjectSetKind.DYNAMIC` set defined by a *filter/predicate* that is
    re-evaluated against the live graph so the membership auto-updates as the
    graph changes; or a :attr:`ObjectSetKind.TEMPORARY` set whose snapshot is
    bounded by a TTL (Foundry's temporary object sets);
  - **search** (``search(query, filters)``) via hybrid/semantic retrieval and
    **filter** (``filter(predicate)``) by typed property/type predicates,
    each returning a *new* :class:`ObjectSet`;
  - **search-around** (``search_around(link_type, hops)``) — traverse a typed
    link N hops to the *related* object set (Foundry's SEARCH_AROUND);
  - **pivot** (``pivot(link_type, group_by)``) — follow a link type to the linked
    set and group/pivot it by a target property;
  - **aggregate** (``aggregate(group_by, metric, ...)``) — real
    count / sum / avg / min / max over object properties, grouped or global;
  - compose with **set algebra** (``union`` / ``intersect`` / ``subtract``).

The set is an abstraction *over existing fabric* — it never reinvents storage or
retrieval. It binds to a live
:class:`~agent_utilities.knowledge_graph.facade.KnowledgeGraph` facade and reads
through that facade's already-built layers:

  - **traversal / property access** through the L1 compute graph
    (:pyattr:`KnowledgeGraph.compute` — a ``GraphComputeEngine``-shaped object
    exposing ``node_ids`` / ``_get_node_properties`` / ``get_successors`` /
    ``out_edges(data=True)`` / ``_get_edge_properties``);
  - **property/full scans** preferentially through the L0 store's Cypher
    (:pyattr:`KnowledgeGraph.store` ``.execute``) when present, falling back to
    the compute graph;
  - **semantic / hybrid search** through
    :class:`~agent_utilities.knowledge_graph.retrieval.hybrid_retriever.HybridRetriever`
    (lazily constructed over an ``IntelligenceGraphEngine`` when one is reachable),
    degrading to a deterministic substring scan when no embedding model / engine
    is available so a search always returns *something*.

Interface-typed sets: where a set should target an *interface* rather than a
concrete object type, callers pass the interface name to :meth:`of_type` /
:meth:`search`; the interface is resolved to its implementing concrete types via
``ontology.interfaces.find_implementers`` (soft import — absence degrades to
treating the name as a concrete type, so this module never hard-breaks).

Wire-First: :class:`ObjectSet` is reached through the ontology facade — the
``object_set`` / ``of_type`` factories on :class:`OntologySystem` (wired by the
integrator) and the ``kg_object_set`` MCP tool — not a free-floating helper.
"""

import time
from collections.abc import Callable, Iterable, Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


__all__ = [
    "ObjectSetKind",
    "Predicate",
    "PropertyFilter",
    "AggregationResult",
    "PivotResult",
    "ObjectSet",
    "GraphView",
    "object_set_from_ids",
    "object_set_of_type",
    "dynamic_object_set",
]


# ── default traversal cap (Foundry SEARCH_AROUND result ceiling) ───────────────
DEFAULT_SEARCH_AROUND_CAP = 100_000


class ObjectSetKind(StrEnum):
    """The three materialization kinds of a Foundry object set (CONCEPT:KG-2.45).

    - ``STATIC``: a fixed, explicit collection of object ids. Membership never
      changes unless a new set is derived from it.
    - ``DYNAMIC``: membership is *defined by a predicate* re-evaluated against the
      live graph on every read, so the set auto-updates as objects are
      added/changed/removed (a saved filter / dynamic object set).
    - ``TEMPORARY``: a snapshot whose validity is bounded by a TTL; after the TTL
      elapses the snapshot is considered expired and is re-materialized from its
      source on next read.
    """

    STATIC = "static"
    DYNAMIC = "dynamic"
    TEMPORARY = "temporary"


# A predicate is any callable over an object's property mapping.
Predicate = Callable[[Mapping[str, Any]], bool]


_COMPARATORS: dict[str, Callable[[Any, Any], bool]] = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "gt": lambda a, b: a is not None and b is not None and a > b,
    "gte": lambda a, b: a is not None and b is not None and a >= b,
    "lt": lambda a, b: a is not None and b is not None and a < b,
    "lte": lambda a, b: a is not None and b is not None and a <= b,
    "in": lambda a, b: a in b if _is_container(b) else False,
    "contains": lambda a, b: (b in a) if _is_container(a) else (str(b) in str(a)),
    "exists": lambda a, b: (a is not None) == bool(b),
}


def _is_container(v: Any) -> bool:
    return isinstance(v, list | tuple | set | frozenset | dict | str)


class PropertyFilter:
    """A single typed property/type predicate on an object's properties.

    CONCEPT:KG-2.45 — the building block of :meth:`ObjectSet.filter` and the
    ``filters`` argument of :meth:`ObjectSet.search`. A filter names a property
    ``field``, a comparison ``op`` (one of eq/ne/gt/gte/lt/lte/in/contains/
    exists), and a comparison ``value``. The special field ``"type"`` matches a
    node's object type (``type``/``_type``/``label`` property), enabling
    type-scoped sets.

    Examples::

        PropertyFilter("type", "eq", "document")
        PropertyFilter("amount", "gte", 1000)
        PropertyFilter("tags", "contains", "urgent")
    """

    __slots__ = ("field", "op", "value")

    def __init__(self, field: str, op: str = "eq", value: Any = None) -> None:
        if op not in _COMPARATORS:
            raise ValueError(
                f"unsupported filter op {op!r}; expected one of {sorted(_COMPARATORS)}"
            )
        self.field = field
        self.op = op
        self.value = value

    def matches(self, props: Mapping[str, Any]) -> bool:
        """Whether ``props`` satisfies this filter (real evaluation, no raise)."""
        actual = _prop(props, self.field)
        try:
            return _COMPARATORS[self.op](actual, self.value)
        except TypeError:
            return False

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"PropertyFilter({self.field!r}, {self.op!r}, {self.value!r})"


def _prop(props: Mapping[str, Any], field: str) -> Any:
    """Read a logical field from an object's property mapping.

    The ``"type"`` field is resolved across the common object-type aliases the
    graph layers use (``type`` / ``_type`` / ``node_type`` / ``label``). The
    ingestion path (``engine_ingestion``) writes the canonical label under
    ``node_type`` (matching ``add_node(node_type=...)``), so it must be in the
    chain or type-filtered object sets miss every ingested node.
    """
    if field == "type":
        return (
            props.get("type")
            or props.get("_type")
            or props.get("node_type")
            or props.get("label")
        )
    return props.get(field)


def _coalesce_filters(
    filters: Iterable[PropertyFilter] | None,
) -> Predicate | None:
    """Compose property filters into one AND-predicate, or ``None`` if empty."""
    flist = list(filters or [])
    if not flist:
        return None

    def _pred(props: Mapping[str, Any]) -> bool:
        return all(f.matches(props) for f in flist)

    return _pred


class AggregationResult:
    """Result of an :meth:`ObjectSet.aggregate` (CONCEPT:KG-2.45).

    Attributes:
        metric: The aggregation metric applied (count/sum/avg/min/max).
        field: The numeric property aggregated (``None`` for a bare ``count``).
        group_by: The grouping property (``None`` for a global aggregation).
        groups: ``{group_value: metric_value}`` when grouped; for a global
            aggregation the single key is ``None``.
        total_objects: Number of objects considered.
    """

    __slots__ = ("metric", "field", "group_by", "groups", "total_objects")

    def __init__(
        self,
        metric: str,
        field: str | None,
        group_by: str | None,
        groups: dict[Any, float],
        total_objects: int,
    ) -> None:
        self.metric = metric
        self.field = field
        self.group_by = group_by
        self.groups = groups
        self.total_objects = total_objects

    @property
    def value(self) -> float | None:
        """The single scalar value for a *global* (ungrouped) aggregation."""
        if self.group_by is not None:
            return None
        return self.groups.get(None)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"AggregationResult(metric={self.metric!r}, field={self.field!r}, "
            f"group_by={self.group_by!r}, groups={self.groups!r})"
        )


class PivotResult:
    """Result of an :meth:`ObjectSet.pivot` (CONCEPT:KG-2.45).

    A pivot follows a link type from the base set to the linked set, then groups
    the linked objects by a target property.

    Attributes:
        link_type: The link/edge type traversed.
        group_by: The target property the linked objects were grouped by.
        groups: ``{group_value: [object_id, ...]}``.
        linked_set: The full related :class:`ObjectSet` (pre-grouping).
    """

    __slots__ = ("link_type", "group_by", "groups", "linked_set")

    def __init__(
        self,
        link_type: str,
        group_by: str,
        groups: dict[Any, list[str]],
        linked_set: ObjectSet,
    ) -> None:
        self.link_type = link_type
        self.group_by = group_by
        self.groups = groups
        self.linked_set = linked_set

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"PivotResult(link_type={self.link_type!r}, group_by={self.group_by!r}, "
            f"groups={ {k: len(v) for k, v in self.groups.items()}!r})"
        )


class GraphView:
    """A uniform read view over whatever graph the facade exposes (CONCEPT:KG-2.45).

    Normalizes the small read surface :class:`ObjectSet` needs across the three
    shapes the facade may hand us — a ``GraphComputeEngine`` (the facade's L1
    :pyattr:`KnowledgeGraph.compute`), an ``IntelligenceGraphEngine`` (exposing a
    ``.graph`` compute engine and optional ``.backend`` store), and a minimal
    duck-typed in-memory graph used in tests. Every method degrades gracefully
    (returns empty / falls through) rather than raising, so a set is usable in
    degraded environments.

    The required duck-typed surface of the underlying graph is small:
    ``node_ids() -> list[str]``, ``_get_node_properties(id) -> dict``,
    ``has_node(id) -> bool``, ``get_successors(id) -> list[str]``,
    ``get_predecessors(id) -> list[str]``, and ``out_edges(id, data=True)`` /
    ``in_edges(id, data=True)`` yielding ``(src, tgt, props)`` triples (or, as a
    fallback, ``_get_edge_properties(src, tgt) -> dict``).
    """

    def __init__(self, graph: Any, store: Any = None) -> None:
        # Unwrap an IntelligenceGraphEngine -> its compute graph + backend store.
        compute = getattr(graph, "graph", None)
        if compute is not None and hasattr(compute, "node_ids"):
            self._g = compute
            self._store = (
                store if store is not None else getattr(graph, "backend", None)
            )
        else:
            self._g = graph
            self._store = store

    @property
    def store(self) -> Any:
        """The L0 Cypher store, if reachable (else ``None``)."""
        return self._store

    # ── nodes / properties ───────────────────────────────────────────────────
    def node_ids(self) -> list[str]:
        """All object ids known to the graph (empty on any failure)."""
        g = self._g
        try:
            if hasattr(g, "node_ids"):
                return list(g.node_ids())
            nodes = getattr(g, "nodes", None)
            if nodes is not None:
                return [str(n) for n in nodes]
        except Exception:
            return []
        return []

    def props(self, node_id: str) -> dict[str, Any]:
        """Property mapping of ``node_id`` (``{}`` when unknown), id always set."""
        g = self._g
        out: dict[str, Any] = {}
        try:
            if hasattr(g, "_get_node_properties"):
                out = dict(g._get_node_properties(node_id) or {})
            elif hasattr(g, "get_node"):
                out = dict(g.get_node(node_id) or {})
        except Exception:
            out = {}
        out.setdefault("id", node_id)
        return out

    def has_node(self, node_id: str) -> bool:
        g = self._g
        try:
            if hasattr(g, "has_node"):
                return bool(g.has_node(node_id))
        except Exception:
            return False
        return node_id in set(self.node_ids())

    # ── typed traversal ──────────────────────────────────────────────────────
    def out_neighbors(self, node_id: str, link_type: str | None) -> list[str]:
        """Successor ids of ``node_id`` whose edge type matches ``link_type``.

        ``link_type`` of ``None`` returns *all* successors. Edge type is read from
        the edge's ``type``/``_type``/``label``/``edge_type`` property.
        """
        return self._typed_neighbors(node_id, link_type, outgoing=True)

    def in_neighbors(self, node_id: str, link_type: str | None) -> list[str]:
        """Predecessor ids of ``node_id`` whose edge type matches ``link_type``."""
        return self._typed_neighbors(node_id, link_type, outgoing=False)

    def _typed_neighbors(
        self, node_id: str, link_type: str | None, *, outgoing: bool
    ) -> list[str]:
        g = self._g
        out: list[str] = []
        seen: set[str] = set()
        edge_fn = "out_edges" if outgoing else "in_edges"
        # Preferred: edge views with data so we can read the type.
        try:
            if hasattr(g, edge_fn):
                for triple in getattr(g, edge_fn)(node_id, data=True):
                    src, tgt, props = _unpack_edge(triple)
                    other = tgt if outgoing else src
                    if other is None or other in seen:
                        continue
                    if link_type is None or _edge_type(props) == link_type:
                        seen.add(other)
                        out.append(other)
                return out
        except Exception:
            out = []
            seen = set()
        # Fallback: plain successor/predecessor ids + per-edge property lookup.
        try:
            succ_fn = "get_successors" if outgoing else "get_predecessors"
            neigh = getattr(g, succ_fn)(node_id) if hasattr(g, succ_fn) else []
            for other in neigh:
                if other in seen:
                    continue
                if link_type is None:
                    seen.add(other)
                    out.append(other)
                    continue
                props = self._edge_props(node_id, other, outgoing=outgoing)
                if _edge_type(props) == link_type:
                    seen.add(other)
                    out.append(other)
        except Exception:
            return out
        return out

    def _edge_props(
        self, node_id: str, other: str, *, outgoing: bool
    ) -> dict[str, Any]:
        g = self._g
        src, tgt = (node_id, other) if outgoing else (other, node_id)
        try:
            if hasattr(g, "_get_edge_properties"):
                return dict(g._get_edge_properties(src, tgt) or {})
            if hasattr(g, "get_edge_data"):
                data = g.get_edge_data(src, tgt) or {}
                inner = data.get(0, data) if isinstance(data, dict) else {}
                return dict(inner or {})
        except Exception:
            return {}
        return {}


def _unpack_edge(triple: Any) -> tuple[str | None, str | None, dict[str, Any]]:
    """Normalize an edge triple to ``(src, tgt, props)``."""
    try:
        if len(triple) >= 3:
            src, tgt, props = triple[0], triple[1], triple[2]
            return src, tgt, dict(props or {}) if isinstance(props, Mapping) else {}
        if len(triple) == 2:
            return triple[0], triple[1], {}
    except TypeError:
        pass
    return None, None, {}


def _edge_type(props: Mapping[str, Any]) -> Any:
    return (
        props.get("type")
        or props.get("_type")
        or props.get("edge_type")
        # The live L1 compute graph stores the edge type under ``rel_type``
        # (matching backend.add_edge(..., rel_type=...)); recognise it so typed
        # SEARCH_AROUND / pivot work on the real graph, not just test duck-graphs.
        or props.get("rel_type")
        or props.get("label")
    )


class ObjectSet:
    """A composable handle on a set of ontology objects (CONCEPT:KG-2.45).

    Construct via the factories — :func:`object_set_from_ids` (STATIC),
    :func:`dynamic_object_set` (DYNAMIC predicate over the live graph), or
    :func:`object_set_of_type` — or via :class:`OntologySystem` accessors. Derive
    new sets with :meth:`search`, :meth:`filter`, :meth:`search_around`, and the
    set-algebra methods; read membership lazily with :meth:`ids`, :meth:`objects`,
    :meth:`count`; and summarize with :meth:`aggregate` / :meth:`pivot`.

    Membership semantics by :attr:`kind`:

      - ``STATIC``: the explicit ``ids`` (validated against the graph lazily).
      - ``DYNAMIC``: every object whose properties satisfy ``predicate``,
        recomputed against the live graph on each :meth:`ids` call — so the set
        reflects graph changes automatically.
      - ``TEMPORARY``: a STATIC-like snapshot whose ``ids`` are re-materialized
        from ``source`` once the ``ttl_seconds`` window elapses.

    Args:
        graph: A live :class:`KnowledgeGraph` facade (preferred) or any
            duck-typed graph/engine the :class:`GraphView` can read.
        kind: The :class:`ObjectSetKind`.
        ids: Explicit ids for a STATIC set / initial TEMPORARY snapshot.
        predicate: Membership predicate for a DYNAMIC set.
        ttl_seconds: TTL for a TEMPORARY set (re-materialize after expiry).
        source: A zero-arg callable returning the live id list a TEMPORARY set
            re-snapshots from on expiry.
        name: Optional human label.
    """

    def __init__(
        self,
        graph: Any,
        *,
        kind: ObjectSetKind = ObjectSetKind.STATIC,
        ids: Iterable[str] | None = None,
        predicate: Predicate | None = None,
        ttl_seconds: float | None = None,
        source: Callable[[], Iterable[str]] | None = None,
        name: str | None = None,
    ) -> None:
        self._kg = graph
        self.kind = kind
        self.name = name
        self._ids: list[str] = list(ids or [])
        self._predicate = predicate
        self._ttl_seconds = ttl_seconds
        self._source = source
        self._snapshot_at = time.monotonic()
        # Resolve the underlying read view from the facade (or a raw engine).
        self._view = _view_for(graph)
        if kind is ObjectSetKind.DYNAMIC and predicate is None:
            raise ValueError("a DYNAMIC object set requires a predicate")

    # ── membership ───────────────────────────────────────────────────────────
    def ids(self) -> list[str]:
        """The current member ids, evaluated per :attr:`kind` (real evaluation)."""
        if self.kind is ObjectSetKind.DYNAMIC:
            pred = self._predicate
            assert pred is not None  # enforced in __init__
            out: list[str] = []
            for nid in self._view.node_ids():
                try:
                    if pred(self._view.props(nid)):
                        out.append(nid)
                except Exception:  # nosec B112 — a bad predicate skips that node, not the query
                    continue
            return out
        if self.kind is ObjectSetKind.TEMPORARY and self._is_expired():
            if self._source is not None:
                self._ids = list(self._source())
            self._snapshot_at = time.monotonic()
        return list(self._ids)

    def _is_expired(self) -> bool:
        if self._ttl_seconds is None:
            return False
        return (time.monotonic() - self._snapshot_at) >= self._ttl_seconds

    def is_expired(self) -> bool:
        """Whether a TEMPORARY set's snapshot window has elapsed."""
        return self.kind is ObjectSetKind.TEMPORARY and self._is_expired()

    def objects(self) -> list[dict[str, Any]]:
        """The member objects as property dicts (id always present)."""
        return [self._view.props(nid) for nid in self.ids()]

    def count(self) -> int:
        """Number of current members."""
        return len(self.ids())

    def __len__(self) -> int:
        return self.count()

    def __iter__(self):
        return iter(self.ids())

    def __contains__(self, node_id: object) -> bool:
        return node_id in set(self.ids())

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"ObjectSet(kind={self.kind.value!r}, name={self.name!r}, "
            f"count={self.count()})"
        )

    # ── derivation: filter ───────────────────────────────────────────────────
    def filter(
        self,
        predicate: Predicate | None = None,
        *,
        filters: Iterable[PropertyFilter] | None = None,
    ) -> ObjectSet:
        """Return a new STATIC set of members satisfying the filter (CONCEPT:KG-2.45).

        Accepts either a free ``predicate`` callable over an object's properties
        and/or a list of typed :class:`PropertyFilter` (AND-combined). At least
        one must be supplied. Evaluation is real and against the *current*
        members of this set.
        """
        prop_pred = _coalesce_filters(filters)
        if predicate is None and prop_pred is None:
            raise ValueError("filter() requires a predicate and/or filters")

        def _keep(props: Mapping[str, Any]) -> bool:
            if predicate is not None and not predicate(props):
                return False
            if prop_pred is not None and not prop_pred(props):
                return False
            return True

        kept = [nid for nid in self.ids() if _keep(self._view.props(nid))]
        return ObjectSet(
            self._kg,
            kind=ObjectSetKind.STATIC,
            ids=kept,
            name=f"{self.name or 'set'}::filtered",
        )

    def of_type(self, type_or_interface: str) -> ObjectSet:
        """Return the members whose object type matches ``type_or_interface``.

        An *interface* name is expanded to its concrete implementing types via
        the ontology interface registry (soft import — if unavailable the name is
        treated as a single concrete type). CONCEPT:KG-2.38 / KG-2.45.
        """
        types = set(_resolve_target_types(self._kg, type_or_interface))

        def _is_type(props: Mapping[str, Any]) -> bool:
            return _prop(props, "type") in types

        return self.filter(_is_type)

    # ── derivation: search ───────────────────────────────────────────────────
    def search(
        self,
        query: str,
        *,
        filters: Iterable[PropertyFilter] | None = None,
        limit: int = 50,
    ) -> ObjectSet:
        """Search this set semantically/lexically, then apply property filters.

        CONCEPT:KG-2.45. The query is resolved through the hybrid/semantic
        retriever when an embedding-backed engine is reachable; otherwise it
        degrades to a deterministic substring scan over each object's textual
        fields (name/title/content/text/description/summary). Either way the hits
        are intersected with this set's current membership and any ``filters``
        (typed property predicates) are applied. Returns a new STATIC set ranked
        by the search.

        An empty/whitespace query short-circuits to a plain :meth:`filter` (or a
        copy when no filters are given) — "search nothing" means "everything".
        """
        member_ids = self.ids()
        if not query or not query.strip():
            base = self
            return base.filter(filters=filters) if filters else base._copy_static()

        ranked = self._search_ids(query, limit=max(limit, 1))
        member_set = set(member_ids)
        # Keep search order; restrict to this set's membership.
        hit_ids = [i for i in ranked if i in member_set]
        # If retrieval found nothing usable, fall back to a substring scan over
        # this set's own members so a search always returns something it can.
        if not hit_ids:
            hit_ids = _substring_scan(
                query, ((nid, self._view.props(nid)) for nid in member_ids)
            )
        prop_pred = _coalesce_filters(filters)
        if prop_pred is not None:
            hit_ids = [i for i in hit_ids if prop_pred(self._view.props(i))]
        return ObjectSet(
            self._kg,
            kind=ObjectSetKind.STATIC,
            ids=hit_ids[:limit],
            name=f"{self.name or 'set'}::search",
        )

    def _search_ids(self, query: str, *, limit: int) -> list[str]:
        """Resolve query → ranked ids via the hybrid retriever, else substring."""
        retr = _hybrid_retriever_for(self._kg)
        if retr is not None:
            try:
                nodes = retr.retrieve_hybrid(
                    query, context_window=limit, skip_quality_gate=True
                )
                ids = [str(n.get("id")) for n in nodes if n.get("id") is not None]
                if ids:
                    return ids
            except Exception:
                pass
        # Deterministic substring scan over the whole graph (degraded mode).
        return _substring_scan(
            query,
            ((nid, self._view.props(nid)) for nid in self._view.node_ids()),
        )[:limit]

    # ── SEARCH-AROUND: related-object traversal ──────────────────────────────
    def search_around(
        self,
        link_type: str | None = None,
        *,
        hops: int = 1,
        direction: str = "out",
        cap: int = DEFAULT_SEARCH_AROUND_CAP,
        include_seed: bool = False,
    ) -> ObjectSet:
        """Traverse ``link_type`` ``hops`` edges to the related object set.

        CONCEPT:KG-2.45 — Foundry SEARCH_AROUND. Starting from this set's members,
        walk the typed link ``hops`` times (BFS) and return the reachable object
        set. ``direction`` is ``"out"`` (successors), ``"in"`` (predecessors), or
        ``"both"``. ``link_type=None`` follows links of *any* type. The frontier
        is capped at ``cap`` distinct objects (default 100k) so a fan-out on a hub
        cannot blow up. By default the seed objects are excluded from the result
        (the *related* set); set ``include_seed=True`` to keep them.

        Returns a new STATIC :class:`ObjectSet`.
        """
        if hops < 1:
            raise ValueError("hops must be >= 1")
        seeds = self.ids()
        seed_set = set(seeds)
        discovered: dict[str, None] = {}  # ordered set
        frontier: list[str] = list(seeds)
        visited: set[str] = set(seeds)
        for _ in range(hops):
            nxt: list[str] = []
            for nid in frontier:
                for other in self._neighbors(nid, link_type, direction):
                    if other in visited:
                        continue
                    visited.add(other)
                    discovered[other] = None
                    nxt.append(other)
                    if len(discovered) >= cap:
                        break
                if len(discovered) >= cap:
                    break
            frontier = nxt
            if not frontier or len(discovered) >= cap:
                break
        result_ids = list(discovered.keys())
        if include_seed:
            result_ids = list(seed_set) + [i for i in result_ids if i not in seed_set]
        return ObjectSet(
            self._kg,
            kind=ObjectSetKind.STATIC,
            ids=result_ids[:cap],
            name=f"{self.name or 'set'}::around[{link_type or '*'}x{hops}]",
        )

    def _neighbors(
        self, node_id: str, link_type: str | None, direction: str
    ) -> list[str]:
        if direction == "out":
            return self._view.out_neighbors(node_id, link_type)
        if direction == "in":
            return self._view.in_neighbors(node_id, link_type)
        if direction == "both":
            out = self._view.out_neighbors(node_id, link_type)
            ins = self._view.in_neighbors(node_id, link_type)
            seen = set(out)
            return out + [i for i in ins if i not in seen]
        raise ValueError(f"direction must be out|in|both, got {direction!r}")

    # ── PIVOT: linked set grouped by a target property ───────────────────────
    def pivot(
        self,
        link_type: str | None,
        group_by: str,
        *,
        direction: str = "out",
        cap: int = DEFAULT_SEARCH_AROUND_CAP,
    ) -> PivotResult:
        """Follow ``link_type`` to the linked set and group it by ``group_by``.

        CONCEPT:KG-2.45 — pivot across a link type. Computes the 1-hop related set
        (:meth:`search_around`) and buckets those linked objects by the value of
        their ``group_by`` property. Returns a :class:`PivotResult` carrying the
        ``{group_value: [ids]}`` buckets plus the full linked :class:`ObjectSet`.
        """
        linked = self.search_around(link_type, hops=1, direction=direction, cap=cap)
        groups: dict[Any, list[str]] = {}
        for nid in linked.ids():
            key = _prop(self._view.props(nid), group_by)
            groups.setdefault(key, []).append(nid)
        return PivotResult(
            link_type=link_type or "*",
            group_by=group_by,
            groups=groups,
            linked_set=linked,
        )

    # ── aggregations ─────────────────────────────────────────────────────────
    def aggregate(
        self,
        metric: str = "count",
        *,
        field: str | None = None,
        group_by: str | None = None,
    ) -> AggregationResult:
        """Aggregate over the set's objects (CONCEPT:KG-2.45).

        Real ``count`` / ``sum`` / ``avg`` / ``min`` / ``max``:

          - ``count`` needs no ``field`` (counts objects per group / overall).
          - ``sum`` / ``avg`` / ``min`` / ``max`` require a numeric ``field``;
            non-numeric / missing values are skipped (an empty bucket yields
            ``0.0`` for ``sum`` and ``None`` for the others — represented as the
            absence of that group when nothing aggregated).

        When ``group_by`` is given, results are bucketed by that property's value;
        otherwise a single global value is returned (key ``None`` in
        :attr:`AggregationResult.groups`, also exposed as
        :attr:`AggregationResult.value`).
        """
        metric = metric.lower()
        if metric not in ("count", "sum", "avg", "min", "max"):
            raise ValueError(
                f"unsupported metric {metric!r}; expected count|sum|avg|min|max"
            )
        if metric != "count" and not field:
            raise ValueError(f"metric {metric!r} requires a numeric field")

        buckets: dict[Any, list[float]] = {}
        counts: dict[Any, int] = {}
        objects = self.objects()
        for props in objects:
            key = _prop(props, group_by) if group_by is not None else None
            counts[key] = counts.get(key, 0) + 1
            if metric != "count":
                val = _as_number(props.get(field)) if field is not None else None
                if val is not None:
                    buckets.setdefault(key, []).append(val)

        groups: dict[Any, float] = {}
        if metric == "count":
            groups = {k: float(v) for k, v in counts.items()}
        else:
            for key, vals in buckets.items():
                groups[key] = _reduce_metric(metric, vals)
            # Groups that had members but no numeric values: emit a neutral 0.0
            # for sum so the group is still represented; skip for min/max/avg.
            if metric == "sum":
                for key in counts:
                    groups.setdefault(key, 0.0)

        return AggregationResult(
            metric=metric,
            field=field,
            group_by=group_by,
            groups=groups,
            total_objects=len(objects),
        )

    # ── set algebra ──────────────────────────────────────────────────────────
    def union(self, other: ObjectSet) -> ObjectSet:
        """Return the union of this set and ``other`` (CONCEPT:KG-2.45)."""
        a = self.ids()
        seen = set(a)
        merged = a + [i for i in other.ids() if i not in seen]
        return self._derive(merged, "union", other)

    def intersect(self, other: ObjectSet) -> ObjectSet:
        """Return the intersection of this set and ``other`` (CONCEPT:KG-2.45)."""
        b = set(other.ids())
        kept = [i for i in self.ids() if i in b]
        return self._derive(kept, "intersect", other)

    def subtract(self, other: ObjectSet) -> ObjectSet:
        """Return this set minus ``other`` (set difference, CONCEPT:KG-2.45)."""
        b = set(other.ids())
        kept = [i for i in self.ids() if i not in b]
        return self._derive(kept, "subtract", other)

    # operator sugar
    __or__ = union
    __and__ = intersect
    __sub__ = subtract

    def _derive(self, ids: list[str], op: str, other: ObjectSet) -> ObjectSet:
        return ObjectSet(
            self._kg,
            kind=ObjectSetKind.STATIC,
            ids=ids,
            name=f"({self.name or 'set'} {op} {other.name or 'set'})",
        )

    def _copy_static(self) -> ObjectSet:
        return ObjectSet(
            self._kg,
            kind=ObjectSetKind.STATIC,
            ids=self.ids(),
            name=self.name,
        )

    def as_temporary(self, ttl_seconds: float) -> ObjectSet:
        """Snapshot the current membership as a TTL-bounded TEMPORARY set.

        The snapshot re-materializes from *this set's* live membership once the
        TTL elapses — so a TEMPORARY view of a DYNAMIC set refreshes on expiry.
        """
        snapshot = self.ids()
        src = self.ids  # bound method → live re-snapshot source
        return ObjectSet(
            self._kg,
            kind=ObjectSetKind.TEMPORARY,
            ids=snapshot,
            ttl_seconds=ttl_seconds,
            source=src,
            name=f"{self.name or 'set'}::temp",
        )


# ── facade-aware helpers ──────────────────────────────────────────────────────


def _view_for(graph: Any) -> GraphView:
    """Build a :class:`GraphView` from a facade or raw engine/graph."""
    # KnowledgeGraph facade: prefer its compute graph + store.
    compute = getattr(graph, "compute", None)
    store = getattr(graph, "store", None)
    if compute is not None and hasattr(compute, "node_ids"):
        return GraphView(compute, store=store)
    # IntelligenceGraphEngine: has .graph (+ optional .backend) — GraphView unwraps.
    if hasattr(graph, "graph") and hasattr(graph.graph, "node_ids"):
        return GraphView(graph)
    # Raw GraphComputeEngine / in-memory duck graph.
    return GraphView(graph, store=store)


def _resolve_target_types(graph: Any, type_or_interface: str) -> list[str]:
    """Expand an interface name to implementing types via the ontology (soft)."""
    ont = getattr(graph, "ontology", None)
    try:
        if ont is not None and getattr(ont, "interfaces", None) is not None:
            return list(ont.interfaces.resolve_target(type_or_interface))
    except Exception:
        pass
    # Soft import of the interface registry as a last resort.
    try:
        from .interfaces import DEFAULT_INTERFACE_REGISTRY

        return list(DEFAULT_INTERFACE_REGISTRY.resolve_target(type_or_interface))
    except Exception:
        return [type_or_interface]


def _hybrid_retriever_for(graph: Any) -> Any:
    """Lazily build a HybridRetriever over a reachable engine, else ``None``."""
    # An IntelligenceGraphEngine carries its own retriever.
    retr = getattr(graph, "hybrid_retriever", None)
    if retr is not None:
        return retr
    engine = graph if hasattr(graph, "hybrid_retriever") else None
    # If the facade hands us a store that is itself engine-backed, reuse it.
    store = getattr(graph, "store", None)
    if store is not None and hasattr(store, "hybrid_retriever"):
        return store.hybrid_retriever
    if engine is not None:
        return getattr(engine, "hybrid_retriever", None)
    return None


def _substring_scan(
    query: str, items: Iterable[tuple[str, Mapping[str, Any]]]
) -> list[str]:
    """Deterministic, embedding-free relevance scan over object text fields.

    Scores each object by the count of distinct query tokens present in its
    textual fields and returns ids in descending score order (ties broken by id
    for determinism). Objects with zero matches are dropped.
    """
    tokens = [t for t in _tokenize(query) if t]
    if not tokens:
        return []
    scored: list[tuple[int, str]] = []
    for nid, props in items:
        text = _object_text(props).lower()
        score = sum(1 for t in set(tokens) if t in text)
        if score > 0:
            scored.append((score, nid))
    scored.sort(key=lambda sc: (-sc[0], sc[1]))
    return [nid for _, nid in scored]


_TEXT_FIELDS = ("name", "title", "content", "text", "description", "summary", "label")


def _object_text(props: Mapping[str, Any]) -> str:
    parts = [str(props.get(k, "")) for k in _TEXT_FIELDS]
    body = " ".join(p for p in parts if p)
    return body or " ".join(str(v) for v in props.values() if isinstance(v, str))


def _tokenize(text: str) -> list[str]:
    import re

    return [t.lower() for t in re.findall(r"[A-Za-z0-9_]{2,}", text)]


def _as_number(value: Any) -> float | None:
    """Coerce a value to float for aggregation, or ``None`` if non-numeric."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _reduce_metric(metric: str, vals: list[float]) -> float:
    if not vals:
        return 0.0
    if metric == "sum":
        return float(sum(vals))
    if metric == "avg":
        return float(sum(vals) / len(vals))
    if metric == "min":
        return float(min(vals))
    if metric == "max":
        return float(max(vals))
    raise ValueError(metric)  # pragma: no cover - guarded by caller


# ── factories (the public construction surface) ───────────────────────────────


def object_set_from_ids(
    graph: Any, ids: Iterable[str], *, name: str | None = None
) -> ObjectSet:
    """Construct a STATIC :class:`ObjectSet` over fixed ``ids`` (CONCEPT:KG-2.45)."""
    return ObjectSet(graph, kind=ObjectSetKind.STATIC, ids=ids, name=name)


def dynamic_object_set(
    graph: Any,
    predicate: Predicate | None = None,
    *,
    filters: Iterable[PropertyFilter] | None = None,
    name: str | None = None,
) -> ObjectSet:
    """Construct a DYNAMIC :class:`ObjectSet` from a predicate / typed filters.

    The set membership is re-evaluated against the live graph on every read, so
    it auto-updates as objects are added, changed, or removed. CONCEPT:KG-2.45.
    """
    prop_pred = _coalesce_filters(filters)
    if predicate is None and prop_pred is None:
        raise ValueError("dynamic_object_set requires a predicate and/or filters")

    if predicate is not None and prop_pred is not None:

        def _pred(props: Mapping[str, Any]) -> bool:
            return predicate(props) and prop_pred(props)

        merged: Predicate = _pred
    else:
        merged = predicate or prop_pred  # type: ignore[assignment]

    return ObjectSet(graph, kind=ObjectSetKind.DYNAMIC, predicate=merged, name=name)


def object_set_of_type(
    graph: Any, type_or_interface: str, *, name: str | None = None
) -> ObjectSet:
    """Construct a DYNAMIC set of all live objects of a type/interface.

    An interface name is resolved to its implementing concrete types; the set
    auto-updates as matching objects appear/disappear. CONCEPT:KG-2.45 / KG-2.38.
    """
    types = set(_resolve_target_types(graph, type_or_interface))

    def _is_type(props: Mapping[str, Any]) -> bool:
        return _prop(props, "type") in types

    return ObjectSet(
        graph,
        kind=ObjectSetKind.DYNAMIC,
        predicate=_is_type,
        name=name or f"of_type:{type_or_interface}",
    )
