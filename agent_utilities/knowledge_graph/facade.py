#!/usr/bin/python
from __future__ import annotations

"""Knowledge Graph facade — the single object the execution plane talks to.

Plan 04 (Multi-Layer Knowledge Graph) splits the system into layers:

* **L0 — store**: the configured persistent graph backend (LadybugDB, Neo4j,
  FalkorDB, in-memory epistemic engine, …) that holds the labelled property
  graph.
* **L1 — compute**: the Rust-native ``epistemic-graph`` compute client used for
  in-process graph algorithms (when available).
* **L2 — semantic / retrieval**: OWL reasoning (``owl_bridge``) and the
  capability-aware designation index (:class:`CapabilityIndex`) that turn the
  graph into actionable routing/designation decisions.

Layer contract (strictly one-directional)::

    graph/*  ->  facade (KnowledgeGraph)  ->  { L0 store, L1 compute, L2 semantic/retrieval }

The execution plane (``graph/*`` — routing, planning, orchestration) depends on
this facade. The facade depends downward on L0/L1/L2. **Nothing below the
facade imports the execution plane**, and the facade itself imports its layers
lazily and defensively so that *constructing* a :class:`KnowledgeGraph` never
requires a running service, an installed optional backend, or a network
connection. Each layer is materialised on first access and any import/connect
failure is tolerated (the attribute resolves to ``None``), keeping the facade
usable in tests, edge deployments, and degraded environments.
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .retrieval.capability_index import CapabilityIndex, Designation

logger = logging.getLogger(__name__)

__all__ = ["KnowledgeGraph"]


class KnowledgeGraph:
    """Composition root over the multi-layer knowledge graph.

    Exposes four lazily-initialised layer attributes — :attr:`compute`,
    :attr:`store`, :attr:`semantic`, :attr:`retrieval` — plus the
    :meth:`designate` convenience that the execution plane uses for
    capability-aware routing.

    Construction is cheap and side-effect free: no layer is touched until its
    property is accessed, and any failure to import or initialise a layer is
    logged and surfaced as ``None`` rather than raised.

    Args:
        backend_type: Optional explicit store backend (``"memory"``,
            ``"ladybug"``, …). Falls back to ``GRAPH_BACKEND`` env / default.
        retrieval: Optional pre-built :class:`CapabilityIndex`. When omitted, a
            fresh empty index is created on first access.
        embedding_dim: Optional embedding dimensionality for a freshly created
            retrieval index.
        **store_kwargs: Forwarded to the backend factory (``db_path``, ``host``,
            …) when the store is created.
    """

    def __init__(
        self,
        *,
        backend_type: str | None = None,
        retrieval: CapabilityIndex | None = None,
        embedding_dim: int | None = None,
        **store_kwargs: Any,
    ) -> None:
        self._backend_type = backend_type
        self._store_kwargs = store_kwargs
        self._embedding_dim = embedding_dim

        # Lazy slots — sentinel ``...`` means "not yet initialised".
        self._store: Any = ...
        self._compute: Any = ...
        self._semantic: Any = ...
        self._retrieval: CapabilityIndex | None = retrieval

    # ------------------------------------------------------------------
    # L0 — store
    # ------------------------------------------------------------------
    @property
    def store(self) -> Any:
        """The configured persistent graph backend (L0), or ``None``.

        Created lazily via the backend factory. Any failure (missing optional
        package, unreachable service) is tolerated and yields ``None``.
        """
        if self._store is ...:
            self._store = None
            try:
                from .backends import create_backend

                self._store = create_backend(
                    backend_type=self._backend_type, **self._store_kwargs
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("KnowledgeGraph: store backend unavailable: %s", exc)
                self._store = None
        return self._store

    # ------------------------------------------------------------------
    # L1 — compute
    # ------------------------------------------------------------------
    @property
    def compute(self) -> Any:
        """The Rust-native graph compute client (L1), or ``None``.

        Prefers the store backend's own compute engine when present; otherwise
        instantiates a standalone ``GraphComputeEngine``. Tolerates absence of
        the compiled engine.
        """
        if self._compute is ...:
            self._compute = None
            try:
                store = self.store
                graph = getattr(store, "graph", None)
                if graph is not None:
                    self._compute = graph
                else:
                    from .core.graph_compute import GraphComputeEngine

                    self._compute = GraphComputeEngine(backend_type="rust")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("KnowledgeGraph: compute engine unavailable: %s", exc)
                self._compute = None
        return self._compute

    # ------------------------------------------------------------------
    # L2 — semantic (OWL bridge)
    # ------------------------------------------------------------------
    @property
    def semantic(self) -> Any:
        """The OWL reasoning bridge (L2 semantics), or ``None``.

        Built lazily over the active compute/store layers. Requires an OWL
        backend; if none can be constructed the attribute resolves to ``None``.
        """
        if self._semantic is ...:
            self._semantic = None
            try:
                from .core.owl_bridge import OWLBridge

                owl_backend = self._make_owl_backend()
                if owl_backend is not None and self.compute is not None:
                    self._semantic = OWLBridge(
                        graph=self.compute,
                        owl_backend=owl_backend,
                        backend=self.store,
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("KnowledgeGraph: semantic layer unavailable: %s", exc)
                self._semantic = None
        return self._semantic

    @staticmethod
    def _make_owl_backend() -> Any:
        """Best-effort construction of an OWL backend; ``None`` on failure."""
        try:
            from .backends.owl import create_owl_backend  # type: ignore

            return create_owl_backend()
        except Exception:
            pass
        try:
            from .backends.owl.owlready2_backend import (  # type: ignore
                Owlready2Backend,
            )

            return Owlready2Backend()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("KnowledgeGraph: no OWL backend available: %s", exc)
            return None

    # ------------------------------------------------------------------
    # L2 — retrieval (capability index)
    # ------------------------------------------------------------------
    @property
    def retrieval(self) -> CapabilityIndex:
        """The capability-aware designation index (L2 retrieval).

        Unlike the other layers this always resolves to a usable object: a
        fresh empty :class:`CapabilityIndex` is created on first access if one
        was not supplied to the constructor.
        """
        if self._retrieval is None:
            from .retrieval.capability_index import CapabilityIndex

            self._retrieval = CapabilityIndex(dim=self._embedding_dim)
        return self._retrieval

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def designate(
        self,
        prompt_embedding: Any,
        required_caps: Any = None,
        k: int = 5,
    ) -> list[Designation]:
        """Designate the top-``k`` entities for a task.

        Delegates to :meth:`CapabilityIndex.designate`, then (when
        ``KG_BRAIN_ENFORCE`` is on) drops designations the current actor is not
        permitted to read and records a read audit. No-op filtering otherwise.

        Args:
            prompt_embedding: The task/query embedding vector.
            required_caps: Optional iterable of required capabilities.
            k: Maximum number of designations to return.

        Returns:
            A list of :class:`Designation` objects, ranked by similarity.
        """
        results = self.retrieval.designate(
            prompt_embedding, required_caps=required_caps, k=k
        )
        from .core.secured_reads import audit_read, permit

        ids: list[str] = [
            i for d in results if isinstance((i := getattr(d, "id", None)), str)
        ]
        if ids:
            allowed = set(permit(ids))
            results = [d for d in results if getattr(d, "id", None) in allowed]
            audit_read(ids, summary="designate")
        # Apply learned/asserted governance rules so corrections-turned-rules
        # change behaviour (CONCEPT:KG-2.8). Best-effort; never blocks retrieval.
        try:
            from .retrieval.governance_rules import (
                apply_governance_rules,
                load_active_rules,
            )

            rules = load_active_rules(self.store)
            if rules:
                results = apply_governance_rules(results, rules)
        except Exception as exc:  # pragma: no cover - enhancement only
            logger.debug("governance rule application skipped: %s", exc)
        return results

    def query(self, cypher: str, params: Any = None) -> list[dict[str, Any]]:
        """Run a tenant-scoped, permission-filtered, audited Cypher read.

        The guarded counterpart to ``store.execute``: applies tenant scoping to
        the query, filters ACL-denied rows, and records a read audit — all
        no-ops unless ``KG_BRAIN_ENFORCE`` is on. Internal/unscoped callers may
        still use ``store.execute`` directly.
        """
        store = self.store
        if store is None:
            return []
        from .core.secured_reads import audit_read, filter_rows, scope

        rows = store.execute(scope(cypher), params or {}) or []
        rows = filter_rows(rows)
        audit_read([], summary="query")
        return rows

    def populate_capability_index(self, nodes: Any) -> int:
        """Populate the L2 retrieval index from graph nodes (Plan 08 Synergy 1).

        This is the bridge that lets the live router call :meth:`designate`
        once the knowledge graph feeds real nodes into the capability index. It
        is intentionally lazy and defensive: it touches only the
        :attr:`retrieval` layer (which always resolves to a usable
        :class:`CapabilityIndex`), so it never requires a running store,
        compute, or semantic backend.

        Each node is a mapping shaped::

            {
                "id": "tool-or-agent-id",
                "embedding": [float, ...],      # required; node skipped if absent
                "capabilities": ["cap", ...],   # optional
                "swappable_with": ["other", ...],  # optional
            }

        Missing keys are tolerated. Nodes without an ``id`` or without an
        ``embedding`` (or with an empty embedding) are skipped — an embedding is
        mandatory for similarity ranking.

        Args:
            nodes: An iterable of node mappings (or objects exposing the same
                attributes — ``capabilities``/``provides``/``providesCapability``
                and ``swappable_with``/``swappableWith`` aliases are accepted,
                matching :meth:`CapabilityIndex.build_from_edges`).

        Returns:
            The number of nodes actually added to the index.
        """
        index = self.retrieval
        added = 0
        for node in nodes:
            if isinstance(node, dict):
                getter = node.get
            else:

                def getter(key: str, default: Any = None, _n: Any = node) -> Any:
                    return getattr(_n, key, default)

            nid = getter("id")
            if nid is None:
                continue
            emb = getter("embedding")
            if emb is None:
                continue
            # Skip empty embeddings without letting CapabilityIndex.add raise.
            try:
                if len(emb) == 0:
                    continue
            except TypeError:
                # Not sized (e.g. a scalar) — let add() validate it.
                pass
            caps = (
                getter("capabilities")
                or getter("provides")
                or getter("providesCapability")
                or []
            )
            swap = getter("swappable_with") or getter("swappableWith") or None
            try:
                index.add(str(nid), emb, caps, swappable_with=swap)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "KnowledgeGraph.populate_capability_index: skipping node %r: %s",
                    nid,
                    exc,
                )
                continue
            added += 1
        return added
