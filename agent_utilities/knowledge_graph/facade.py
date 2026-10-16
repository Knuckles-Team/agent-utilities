#!/usr/bin/python
from __future__ import annotations

"""Knowledge Graph facade — the single object the execution plane talks to.

The **epistemic-graph engine is the ONE authority** (compute + in-memory cache +
semantic + durable persistence); the facade composes its storage, native RDF/OWL
compute methods, and EG designation port behind one object. There is no numeric
graph-storage hierarchy — these concerns are served by the one authority:

* **store** — the epistemic engine, automatically wrapped with any configured
  durable external projections (e.g. ``backends/`` mirrors when
  ``GRAPH_MIRROR_TARGETS`` names one or more mirrors), holding the labelled
  property graph.
* **compute** — the Rust-native ``epistemic-graph`` client for graph algorithms,
  SPARQL, and read-only OWL reasoning.
* **designation** — forwarded to the EG capability-designation port (EG-DECISION-ENGINE-R172).

Dependency contract (strictly one-directional)::

    graph/*  ->  facade (KnowledgeGraph)  ->  { store, compute }

The execution plane (``graph/*`` — routing, planning, orchestration) depends on
this facade. The facade depends downward on those concerns. **Nothing below the
facade imports the execution plane**, and the facade itself imports them
lazily so that *constructing* a :class:`KnowledgeGraph` remains side-effect
free — it never requires a running service, an installed optional backend, or
a network connection. Accessing its operational store or compute layer
requires the already active process-owned engine and fails closed when that
authority is absent. The ontology registry remains lazily constructed and
best-effort so the facade can still be created in tests and degraded environments.
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .core.session import GraphSession
    from .retrieval.capability_designation_port import (
        CapabilityDesignation,
        CapabilityDesignationPort,
    )

logger = logging.getLogger(__name__)

__all__ = ["KnowledgeGraph"]


class KnowledgeGraph:
    """Composition root over the multi-layer knowledge graph.

    Exposes lazily-initialised :attr:`compute` and :attr:`store` layers, native :meth:`sparql` and :meth:`owl_reason`
    operations, the :attr:`ontology` registry, and the :meth:`designate`
    convenience used by capability-aware routing.

    Construction is cheap and side-effect free: no layer is touched until its
    property is accessed. Operational layers bind to the active process-owned
    engine rather than constructing an overlapping backend or transport.

    Args:
        designator: Optional EG capability-designation port
            (``EG-DECISION-ENGINE-R172``). :meth:`designate` forwards to it;
            without one, designation returns no candidates. AU keeps no index.
    """

    def __init__(
        self,
        *,
        designator: CapabilityDesignationPort | None = None,
    ) -> None:
        self._designator: CapabilityDesignationPort | None = designator

        # Lazy slots — sentinel ``...`` means "not yet initialised".
        self._store: Any = ...
        self._compute: Any = ...
        self._ontology: Any = ...

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
    # semantic — native epistemic-graph RDF/OWL operations
    # ------------------------------------------------------------------
    def sparql(
        self,
        query: str,
        *,
        base_iri: str = "",
        type_convention: str = "",
    ) -> list[dict[str, str | None]]:
        """Run SPARQL through the graph authority's native RDF engine."""
        return self.compute.sparql(
            query, base_iri=base_iri, type_convention=type_convention
        )

    def owl_reason(
        self,
        *,
        ontology: str | None = None,
        target_class: str | None = None,
        class_base: str = "http://agent-utilities.dev/ontology#",
    ) -> dict[str, Any]:
        """Run the graph authority's native, read-only OWL reasoner."""
        return self.compute.owl_reason(
            ontology=ontology,
            target_class=target_class,
            class_base=class_base,
        )

    # ------------------------------------------------------------------
    # Ontology layer — Palantir Foundry parity (KG-2.26/2.38/2.39/2.40/2.41/2.47)
    # ------------------------------------------------------------------
    @property
    def ontology(self) -> Any:
        """The first-class ontology system bound to this graph, or ``None``.

        Composes the property-type / value-type / interface / link / function /
        derived-property registries (:class:`OntologySystem`) and binds them to
        *this* facade so Functions-on-Objects, derived-property compute, and
        interface targeting resolve against the live store/native compute/retrieval
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
    ) -> list[CapabilityDesignation]:
        """Designate the top-``k`` entities for a task.

        Forwards to the EG capability-designation port (EG owns ranking,
        ``EG-DECISION-ENGINE-R172``; no port means no candidates), then (when
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
            A list of :class:`CapabilityDesignation` objects in EG's order.
        """
        from .core.session import resolve_session

        session = resolve_session(session, required_scope="kg:read")

        from .retrieval.capability_designation_port import (
            CapabilityDesignationRequest,
        )

        if self._designator is None:
            return []
        results = self._designator.designate(
            CapabilityDesignationRequest(
                query_vector=[float(v) for v in prompt_embedding],
                required_capabilities=tuple(str(c) for c in (required_caps or ())),
                k=int(k),
            )
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
        except Exception as exc:  # pragma: no cover - enhancement only  # noqa: BLE001 — governance-rule re-ranking is an enhancement over `results`, which is already fully computed and returned unconditionally on the next line
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
        from .core.tenant_sharing import push_down_visibility

        # D-W2T-2: `scope()` returns (query, extra_params) — the tenant id is a
        # bound `$_tenant_scope_id` Cypher parameter now, not a string-literal
        # splice. Merge it into this call's own params dict.
        scoped_cypher, tenant_params = scope(cypher, session.actor)
        merged_params = {**(params or {}), **tenant_params}
        # Best-effort owner/scope visibility pushdown INTO the query text
        # (mirrors `QueryMixin.query_cypher`'s identical non-aggregate-branch
        # fix): when it demonstrably succeeds (or the actor is privileged),
        # `pushed_down` lets a row with no governed id — a plain
        # `RETURN n.name AS name` projection, which carries no `id` column at
        # all — be trusted by the post-hoc classifiers below instead of
        # rejecting the WHOLE read (the bug this closes: every non-aggregate
        # projection lacking an `id` column raised `PermissionError`, which
        # `read_union`/callers upstream had been silently swallowing into an
        # empty `[]`). Best-effort and fails closed on any pushdown failure —
        # the post-hoc row filters below remain the enforcement backstop
        # either way.
        scoped_cypher, visibility_params, pushed_down = push_down_visibility(
            scoped_cypher, session.actor
        )
        merged_params.update(visibility_params)
        rows = store.execute_read(scoped_cypher, merged_params) or []
        rows = visible(
            filter_rows(rows, session.actor, trust_pushdown=pushed_down),
            session.actor,
        )
        # Fine-grained object permissioning is mandatory: rows without governed
        # markings/ACLs are denied and enforcement failures propagate — unless
        # `pushed_down`, see `push_down_visibility`/`filter_rows` above.
        from .ontology.permissioning import enforce as enforce_fine_grained

        rows = enforce_fine_grained(rows, session.actor, trust_pushdown=pushed_down)
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

    def stream_epistemic_by_label(
        self, label: str, *, batch_size: int = 512, limit: int = 0
    ) -> Any | None:
        """CONCEPT:AU-KG.query.knowledge-stream-consumer (report §9 #3) — bulk, Arrow-columnar,
        cursor-resumable :class:`~.core.epistemic_row.EpistemicRow` sweep over up to
        ``limit`` nodes of ``label`` (``limit=0`` ⇒ uncapped), via
        ``Method::KnowledgeStream``. The label-scoped, streaming sibling of
        :meth:`query`'s ``include_epistemic=True`` path (which is id-seeded and
        answers in one non-streaming round trip) — use this for a population-level
        epistemic sweep (a bulk claim/evidence review, a confidence-distribution
        audit) where the caller wants EVERY/MANY node of a label, not a specific
        id list. Never materializes the whole result set in one frame: each Arrow
        page is bounded by ``batch_size`` and the cursor resumes the next.

        Returns ``None`` (never raises) when the compute engine has no
        ``.knowledge`` streaming surface reachable, or ``pyarrow`` isn't installed
        — a caller falls back to its existing bulk-read path.
        """
        from .core.epistemic_row import stream_epistemic_rows_by_label

        return stream_epistemic_rows_by_label(
            self.compute, label, batch_size=batch_size, limit=limit
        )

    def query_batches(
        self,
        query: str,
        *,
        family: str = "cross_modal",
        batch_size: int = 512,
        limit: int = 0,
        params: bytes = b"",
        base_iri: str = "",
        type_convention: str = "",
    ) -> Any | None:
        """CONCEPT:AU-KG.query.knowledge-stream-consumer (Seam 1, the ``KnowledgeBatch``
        bulk-throughput currency) — stream ``query``'s result as bounded, epistemic
        Arrow ``RecordBatch`` pages over ``Method::KnowledgeStream``.

        The COLUMNAR sibling of :meth:`query`'s ``include_epistemic=True`` /
        :meth:`stream_epistemic_by_label`: instead of Python
        :class:`~.core.epistemic_row.EpistemicRow` objects, it yields raw
        ``pyarrow.RecordBatch`` pages (each ≤ ``batch_size`` rows) carrying the
        engine's full epistemic schema — ``id``/``kind``/``score_*``/``confidence``/
        ``evidence_*``/the bitemporal valid/tx window/``source_refs``/
        ``policy_labels``/``transformation_ids``/``proof_ids``/``alternative_ids``/
        ``contradiction_ids``/``blob_handle`` — so a caller hands results straight to
        anything Arrow-speaking (Polars/pandas, DataFusion, a Parquet/distillation
        export, a vectorized aggregation) and RSS stays FLAT across an arbitrarily
        large scan, the whole reason the columnar currency exists. Never fabricates a
        column the wire didn't carry.

        ``family`` selects the ``KnowledgeStreamQuery`` family and how ``query`` is
        read: ``"cross_modal"`` (default — ``query`` is UQL text, which subsumes
        graph-pattern ``MATCH … |> …`` queries), ``"sql"`` (read-only SQL, with
        optional msgpack ``params``), ``"rdf"``/``"sparql"`` (SPARQL, with optional
        ``base_iri``/``type_convention``), or ``"graph"`` (``query`` is a node label
        swept up to ``limit`` rows, ``limit=0`` ⇒ uncapped).

        Returns ``None`` (never raises) when the compute engine exposes no
        ``.knowledge`` streaming surface or ``pyarrow`` isn't installed (an optional
        extra) — a caller falls back to its existing non-columnar read path. Degrades
        cleanly to a short/empty stream on a non-epistemic backend; it never
        fabricates rows.
        """
        from .core.knowledge_stream import pull_record_batches

        selected = family.lower()
        if selected in ("cross_modal", "uql", "cypher"):
            wire_query: dict[str, Any] = {"family": "cross_modal", "text": str(query)}
        elif selected == "sql":
            wire_query = {
                "family": "sql",
                "query": str(query),
                "params_msgpack": bytes(params),
            }
        elif selected in ("rdf", "sparql"):
            wire_query = {
                "family": "rdf",
                "query": str(query),
                "base_iri": str(base_iri),
                "type_convention": str(type_convention),
            }
        elif selected == "graph":
            wire_query = {"family": "graph", "label": str(query), "limit": int(limit)}
        else:
            raise ValueError(
                f"query_batches: unknown family {family!r} "
                "(use cross_modal/uql/cypher, sql, rdf/sparql, or graph)"
            )
        return pull_record_batches(self.compute, wire_query, batch_size=batch_size)

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
