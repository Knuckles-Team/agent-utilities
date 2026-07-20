#!/usr/bin/python
from __future__ import annotations

"""Knowledge Graph facade — the single object the execution plane talks to.

The **epistemic-graph engine is the ONE authority** (compute + in-memory cache +
semantic + durable persistence); the facade composes it with the semantic + retrieval
layers behind one object. There is no numeric graph-storage hierarchy — the facade exposes three
collaborating concerns, all served by the one authority:

* **store** — the epistemic engine, automatically wrapped with any configured
  durable external projections, holding the labelled property graph.
* **compute** — the Rust-native ``epistemic-graph`` compute client for in-process graph
  algorithms (when available).
* **semantic / retrieval** — OWL reasoning (``owl_bridge``) and the capability-aware
  designation index (:class:`CapabilityIndex`) that turn the graph into actionable
  routing/designation decisions.

Dependency contract (strictly one-directional)::

    graph/*  ->  facade (KnowledgeGraph)  ->  { store, compute, semantic/retrieval }

The execution plane (``graph/*`` — routing, planning, orchestration) depends on
this facade. The facade depends downward on those concerns. **Nothing below the
facade imports the execution plane**, and the facade itself imports them
lazily so that *constructing* a :class:`KnowledgeGraph` remains side-effect
free. Accessing its operational store or compute layer requires the already
active process-owned engine and fails closed when that authority is absent.
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .core.session import GraphSession
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
    property is accessed. Operational layers bind to the active process-owned
    engine rather than constructing an overlapping backend or transport.

    Args:
        retrieval: Optional pre-built :class:`CapabilityIndex`. When omitted, a
            fresh empty index is created on first access.
        embedding_dim: Optional embedding dimensionality for a freshly created
            retrieval index.
    """

    def __init__(
        self,
        *,
        retrieval: CapabilityIndex | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        self._embedding_dim = embedding_dim

        # Lazy slots — sentinel ``...`` means "not yet initialised".
        self._store: Any = ...
        self._compute: Any = ...
        self._semantic: Any = ...
        self._retrieval: CapabilityIndex | None = retrieval
        self._ontology: Any = ...
        # Object Index Funnel (CONCEPT:AU-KG.ontology.batch-incremental-sync-live) — lazily built over self.retrieval.
        self._object_funnel: Any = None

    @classmethod
    def from_engine(cls, engine: Any) -> KnowledgeGraph:
        """Bind the guarded facade to an existing live engine.

        Served entrypoints use this constructor instead of reaching through an
        ``IntelligenceGraphEngine`` to its backend.  The facade remains the one
        policy/audit boundary, while the backend-bound compute client preserves
        the exact named graph selected for the request.
        """
        facade = cls()
        store = getattr(engine, "backend", None)
        if store is None:
            raise RuntimeError("The selected graph engine has no active store")
        facade._store = store
        facade._compute = getattr(store, "graph", None) or getattr(
            engine, "graph_compute", None
        )
        return facade

    def cache_snapshot_token(self, session: GraphSession | None = None) -> str | None:
        """Return a trusted data-snapshot token suitable for a read-cache key.

        A short TTL is not a substitute for snapshot identity: property updates
        can change a result without changing node/edge counts.  This method only
        accepts explicit revision primitives exposed by a server-owned
        transaction, compute client, or store.  When none exists it returns
        ``None`` and the governed query boundary bypasses caching.
        """
        candidates = [getattr(session, "txn", None), self._compute, self._store]
        for candidate in candidates:
            if candidate is None or candidate is ...:
                continue
            for name in (
                "cache_snapshot_token",
                "snapshot_token",
                "snapshot_id",
                "data_revision",
                "commit_revision",
                "generation",
            ):
                value = getattr(candidate, name, None)
                if callable(value):
                    try:
                        value = value()
                    except Exception:  # pragma: no cover - optional primitive
                        continue
                if isinstance(value, str | int) and str(value):
                    return f"{type(candidate).__name__}:{name}:{value}"
        return None

    # ------------------------------------------------------------------
    # store — the authority's graph backend
    # ------------------------------------------------------------------
    @property
    def store(self) -> Any:
        """Return the active process authority's persistent graph backend.

        The facade never constructs a backend. GraphOS owns engine lifecycle;
        callers must bind an explicit engine with :meth:`from_engine` or start
        the one process authority before accessing this lazy property.
        """
        if self._store is ...:
            from .core.engine import IntelligenceGraphEngine

            active = IntelligenceGraphEngine.get_active()
            if active is None:
                raise RuntimeError(
                    "KnowledgeGraph requires the active process-owned graph engine"
                )
            store = getattr(active, "backend", None)
            if store is None:
                raise RuntimeError(
                    "The active process-owned graph engine has no store authority"
                )
            self._store = store
        return self._store

    # ------------------------------------------------------------------
    # compute — the Rust-native graph client
    # ------------------------------------------------------------------
    @property
    def compute(self) -> Any:
        """Return the active process authority's Rust-native compute client.

        The facade never creates a second ``GraphComputeEngine``. An explicit
        :meth:`from_engine` binding remains available for dependency-injected
        tests and graph-scoped views.
        """
        if self._compute is ...:
            from .core.engine import IntelligenceGraphEngine

            active = IntelligenceGraphEngine.get_active()
            if active is None:
                raise RuntimeError(
                    "KnowledgeGraph requires the active process-owned graph engine"
                )
            compute = getattr(active, "graph_compute", None)
            if compute is None:
                raise RuntimeError(
                    "The active process-owned graph engine has no compute authority"
                )
            self._compute = compute
        return self._compute

    # ------------------------------------------------------------------
    # semantic — the OWL reasoning bridge
    # ------------------------------------------------------------------
    @property
    def semantic(self) -> Any:
        """The OWL reasoning bridge (semantic layer), or ``None``.

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
    # retrieval — the capability designation index
    # ------------------------------------------------------------------
    @property
    def retrieval(self) -> CapabilityIndex:
        """The capability-aware designation index (retrieval layer).

        Unlike the other layers this always resolves to a usable object: a
        fresh empty :class:`CapabilityIndex` is created on first access if one
        was not supplied to the constructor.
        """
        if self._retrieval is None:
            from .retrieval.capability_index import CapabilityIndex

            self._retrieval = CapabilityIndex(dim=self._embedding_dim)
        return self._retrieval

    # ------------------------------------------------------------------
    # Ontology layer — Palantir Foundry parity (KG-2.26/2.38/2.39/2.40/2.41/2.47)
    # ------------------------------------------------------------------
    @property
    def ontology(self) -> Any:
        """The first-class ontology system bound to this graph, or ``None``.

        Composes the property-type / value-type / interface / link / function /
        derived-property registries (:class:`OntologySystem`) and binds them to
        *this* facade so Functions-on-Objects, derived-property compute, and
        interface targeting resolve against the live store/semantic/retrieval
        layers. Built lazily and defensively — any import failure resolves to
        ``None`` rather than raising, matching the other layer accessors.
        """
        if self._ontology is ...:
            self._ontology = None
            try:
                from .ontology import OntologySystem

                self._ontology = OntologySystem(graph=self)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("KnowledgeGraph: ontology layer unavailable: %s", exc)
                self._ontology = None
        return self._ontology

    # ------------------------------------------------------------------
    # Object Index Lifecycle — the Object Data Funnel (CONCEPT:AU-KG.ontology.batch-incremental-sync-live)
    # ------------------------------------------------------------------
    @property
    def object_index_funnel(self) -> Any:
        """The Object Data Funnel driving *this* facade's live search index.

        Cached :class:`ObjectIndexFunnel` constructed once over
        :pyattr:`retrieval` — it drives the SAME :class:`CapabilityIndex`
        ``designate`` ranks against (never a second index), so batch/incremental
        sync and staleness tracking apply to the live retrieval plane.
        """
        if self._object_funnel is None:
            from .ontology.indexing import ObjectIndexFunnel

            self._object_funnel = ObjectIndexFunnel(index=self.retrieval)
        return self._object_funnel

    def sync_object_index(self, nodes: Any) -> Any:
        """Batch-rebuild the live object index from source-of-truth nodes.

        Routes through :meth:`ObjectIndexFunnel.batch_sync` so staleness is
        tracked. Returns the funnel's :class:`SyncResult`.
        """
        return self.object_index_funnel.batch_sync(nodes)

    def reindex_stale_objects(self, source_nodes: Any) -> Any:
        """Reconcile the live index against ``source_nodes`` (CONCEPT:AU-KG.ontology.batch-incremental-sync-live).

        Computes content-hash drift via the staleness ledger and applies exactly
        the needed upsert/delete delta. Returns the funnel's :class:`SyncResult`.
        """
        return self.object_index_funnel.reconcile(source_nodes)

    # ------------------------------------------------------------------
    # Object permissioning — fine-grained read enforcement (CONCEPT:AU-KG.ontology.redact-object-materialize-restricted)
    # ------------------------------------------------------------------
    def restricted_view(
        self,
        objects: list[dict[str, Any]],
        actor: Any = None,
        *,
        mask: bool = False,
    ) -> list[dict[str, Any]]:
        """Materialize a permission-filtered VIEW of ``objects`` for an actor.

        Delegates to :func:`ontology.permissioning.restricted_view` (row-drop of
        marked/ACL-denied objects composed with column redaction).
        """
        from .ontology.permissioning import restricted_view as _rv

        return _rv(objects, actor, mask=mask)

    def apply_marking(self, node_id: str, marking: Any) -> None:
        """Attach a mandatory marking to a node (CONCEPT:AU-KG.ontology.redact-object-materialize-restricted)."""
        from .ontology.permissioning import apply_marking as _am

        _am(node_id, marking)

    def process_document(self, document: Any, **kwargs: Any) -> dict[str, Any]:
        """Process ``document`` into Document + Chunk objects (CONCEPT:AU-KG.ingest.chunk-overlap-stage).

        Convenience that runs the document-processing pipeline through this live
        facade's write path; returns ``{document_node, chunk_nodes, edges}``.
        """
        from .ontology.document_processing import process_document as _pd

        return _pd(document, self, **kwargs)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def designate(
        self,
        prompt_embedding: Any,
        required_caps: Any = None,
        k: int = 5,
        *,
        session: GraphSession | None = None,
    ) -> list[Designation]:
        """Designate the top-``k`` entities for a task.

        Delegates to :meth:`CapabilityIndex.designate`, then (when
        mandatory Company Brain enforcement drops designations the current actor is not
        permitted to read and records a read audit. No-op filtering otherwise.

        Args:
            prompt_embedding: The task/query embedding vector.
            required_caps: Optional iterable of required capabilities.
            k: Maximum number of designations to return.
            session: Explicit :class:`~.core.session.GraphSession` this read
                runs under (CONCEPT:AU-P0-1 — the one currency). When omitted, one is
                inherited from the verified ambient context in production.

        Returns:
            A list of :class:`Designation` objects, ranked by similarity.
        """
        from .core.session import resolve_session

        session = resolve_session(session, required_scope="kg:read")

        results = self.retrieval.designate(
            prompt_embedding, required_caps=required_caps, k=k
        )
        from .core.secured_reads import audit_read, permit

        ids: list[str] = [
            i for d in results if isinstance((i := getattr(d, "id", None)), str)
        ]
        if ids:
            allowed = set(permit(ids, session.actor))
            results = [d for d in results if getattr(d, "id", None) in allowed]
            audit_read(ids, summary="designate", actor=session.actor)
        # Apply learned/asserted governance rules so corrections-turned-rules
        # change behaviour (CONCEPT:EG-KG.storage.nonblocking-checkpoint). Best-effort; never blocks retrieval.
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

    def reason(self, task: Any) -> Any:
        """Route a reasoning task to the best-learned paradigm and run it (KG-2.68).

        First-class entry to the pluggable-paradigm router: a :class:`ReasoningTask`
        is dispatched to the inductive (KG-2.69) / model-based (KG-2.67) / deductive /
        generative paradigm whose learned reward EMA + capability tags best fit, and
        the scored result trains the routing so paradigm selection self-improves.
        """
        from .core.reasoner import ReasoningTask, get_reasoner_router

        if not isinstance(task, ReasoningTask):
            task = ReasoningTask(goal=str(task))
        return get_reasoner_router().reason(task)

    def query(
        self,
        cypher: str,
        params: Any = None,
        *,
        session: GraphSession | None = None,
        include_epistemic: bool = False,
    ) -> list[dict[str, Any]]:
        """Run a tenant-scoped, permission-filtered, audited Cypher read.

        The guarded counterpart to ``store.execute_read``: applies tenant scoping to
        the query, filters ACL-denied rows, and records a read audit — all
        mandatory current-contract boundaries. Internal graph reads inherit the
        same verified session and may not bypass these controls.

        Args:
            cypher: The read query.
            params: Optional query parameters.
            session: Explicit :class:`~.core.session.GraphSession` this read
                runs under (CONCEPT:AU-P0-1 — the one currency: identity + policy +
                trace in one object). Production inherits the middleware-minted
                ambient session and rejects any out-of-band replacement.
            include_epistemic: Opt-in (CONCEPT:AU-KB-CURRENCY, Seam 1 — the
                ``KnowledgeBatch`` currency). Default ``False`` — existing callers
                get the byte-for-byte SAME plain ``list[dict]`` rows as before this
                parameter existed. When ``True``, this method instead returns
                ``list[EpistemicRow]``: the SAME rows widened with the engine's
                per-row epistemic envelope (score, confidence, the bitemporal
                valid/tx window, evidence provenance, policy labels) — resolved
                server-side via ``Method::ExplainProvenanceByIds``, never
                fabricated AU-side. Requires an ``EpistemicGraphBackend`` store
                (a ``compute.explain_provenance_by_ids`` primitive); any other
                backend degrades to an empty list rather than raising. See
                ``docs/architecture/epistemic-columns-currency.md`` for the full
                seam and how another read path adopts the same pattern.
        """
        from .core.session import resolve_session

        session = resolve_session(session, required_scope="kg:read")
        store = self.store
        if store is None:
            raise PermissionError("Graph read store is unavailable")
        from .core.secured_reads import audit_read, filter_rows, scope, visible

        rows = store.execute_read(scope(cypher, session.actor), params or {}) or []
        rows = visible(filter_rows(rows, session.actor), session.actor)
        # Fine-grained object permissioning is mandatory: rows without governed
        # markings/ACLs are denied and enforcement failures propagate.
        from .ontology.permissioning import enforce as enforce_fine_grained

        rows = enforce_fine_grained(rows, session.actor)
        audit_read([], summary="query", actor=session.actor)
        if include_epistemic:
            return self._attach_epistemic(rows)  # type: ignore[return-value]
        # Light epistemic layer (CONCEPT:AU-KB-CURRENCY, Native by default):
        # additive confidence/source_refs/evidence_refs/policy_labels/
        # provenance columns merged onto the SAME plain rows, default-on —
        # distinct from the heavy `include_epistemic=True` path above, which
        # stays opt-in (type-changing to `EpistemicRow`, an extra round trip
        # a caller must explicitly ask for). Never changes the return type,
        # so every existing `list[dict]` caller is unaffected.
        from agent_utilities.core.config import config as _app_config

        from .core.epistemic_row import (
            attach_epistemic_columns,
            should_attach_epistemic_columns,
        )

        if should_attach_epistemic_columns(
            rows, default=_app_config.epistemic_light_default
        ):
            rows = attach_epistemic_columns(
                rows, getattr(self.compute, "explain_provenance_by_ids", None)
            )
        return rows

    def _attach_epistemic(self, rows: list[dict[str, Any]]) -> list[Any]:
        """CONCEPT:AU-KB-CURRENCY (Seam 1) — currency-upgrade plain ``rows`` into
        :class:`~.core.epistemic_row.EpistemicRow` results.

        Extracts the distinct node ids ``rows`` project (see
        :func:`~.core.epistemic_row.row_ids_from_plain_rows`), resolves their
        epistemic envelope in ONE round-trip via
        ``compute.explain_provenance_by_ids`` (``Method::ExplainProvenanceByIds``),
        and zips each engine row back with the plain row's own properties so no
        information is lost by opting in. Degrades to ``[]`` (never raises, never
        fabricates) when no compute engine is bound or nothing in ``rows`` carries
        a resolvable id.
        """
        from .core.epistemic_row import attach_epistemic_rows

        compute = self.compute
        fetch = getattr(compute, "explain_provenance_by_ids", None)
        if fetch is None:
            logger.debug(
                "KnowledgeGraph.query(include_epistemic=True): no compute engine "
                "exposing explain_provenance_by_ids — returning no epistemic rows"
            )
            return []
        return attach_epistemic_rows(rows, fetch)

    def tenant_graph(self, tenant: str | None = None, base: str | None = None) -> str:
        """Resolve the per-tenant named graph (CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw).

        The naming discipline that makes engine sharding tenant-partitioned:
        ``tenant → named graph → authoritative engine placement``. ``tenant`` defaults to the
        ambient :class:`ActorContext` tenant; ``base`` to the configured
        default graph (``KG_DEFAULT_GRAPH``). With no tenant in scope the base
        is returned unchanged, so single-tenant deployments are unaffected.
        """
        from agent_utilities.security.brain_context import current_actor

        from .core.shard_topology import default_graph_name, tenant_graph_name

        effective_tenant = tenant if tenant is not None else current_actor().tenant_id
        return tenant_graph_name(effective_tenant, base or default_graph_name())

    def populate_capability_index(self, nodes: Any) -> int:
        """Populate the retrieval (capability designation) index from graph nodes (Plan 08 Synergy 1).

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
