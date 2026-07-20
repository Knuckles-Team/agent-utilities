#!/usr/bin/python
from __future__ import annotations

"""Unified Intelligence Graph Engine.

This module provides the high-level interface for querying the unified knowledge graph,
supporting structural Cypher queries, topological impact analysis, and hybrid search.

GraphComputeEngine is the single operational authority for storage, retrieval,
and native graph algorithms. Optional backends are explicit interoperability or
mirror targets; they do not introduce a second read authority.

The engine is composed of focused mixins for maintainability:
- ``engine_query.py``: Query, search, and retrieval methods.
- ``engine_memory.py``: Memory CRUD operations.
- ``engine_ingestion.py``: Episode, MCP, A2A, and skill ingestion.
- ``engine_registry.py``: Identity, prompt, resource, and codemap management.
"""

import asyncio
import contextlib
import json
import logging
import math
import threading
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .session import GraphSession

from ...core.registry.kg_adapter import FocusedSubgraph, RegistryMixin
from ..backends import create_backend, get_active_backend
from ..backends.base import GraphBackend
from ..orchestration.engine_ahe import AHEMixin
from ..orchestration.engine_enterprise import EnterpriseEngineMixin
from ..orchestration.engine_federation import FederationMixin
from ..orchestration.engine_finance import FinanceEngineMixin
from ..orchestration.engine_infra import InfrastructureEngineMixin
from ..orchestration.engine_ml_rlm import MachineLearningEngineMixin
from ..orchestration.engine_query import QueryMixin
from .engine_ingestion import IngestionMixin
from .engine_mcp_discovery import MCPDiscoveryMixin
from .engine_memory import MemoryMixin
from .engine_tasks import TaskManagerMixin
from .graph_compute import GraphComputeEngine

logger = logging.getLogger(__name__)

__all__ = [
    "IntelligenceGraphEngine",
    "FocusedSubgraph",
    "cosine_similarity",
]


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2, strict=False))
    magnitude1 = math.sqrt(sum(a * a for a in v1))
    magnitude2 = math.sqrt(sum(a * a for a in v2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


# implements core.execution.ExecutionEngine
class IntelligenceGraphEngine(
    QueryMixin,
    MemoryMixin,
    IngestionMixin,
    MCPDiscoveryMixin,
    RegistryMixin,
    TaskManagerMixin,
    FederationMixin,
    AHEMixin,
    InfrastructureEngineMixin,
    # CONCEPT:AU-KG.domains.lazy-symbol-loading — domain methods are composed onto
    # the engine so current surfaces invoke one real engine capability directly.
    EnterpriseEngineMixin,
    FinanceEngineMixin,
    MachineLearningEngineMixin,
):
    """Engine for querying the unified intelligence graph (Agents, Tools, Code, Memory).

    Composed of focused mixins for maintainability. All 49+ existing importers
    continue to work since IntelligenceGraphEngine is still the single public class.

    ``self.graph`` and ``self.graph_compute`` name the same native graph
    authority. Optional mirrors are owned and reconciled by ``FanOutBackend``.
    """

    _ACTIVE_ENGINE: IntelligenceGraphEngine | None = None
    _ACTIVE_ENGINE_LOCK = threading.RLock()

    def __init__(
        self,
        backend: GraphBackend | None = None,
        db_path: str | None = None,
        graph: Any = None,
        schema_pack: Any = None,
    ):
        if IntelligenceGraphEngine._ACTIVE_ENGINE is not None:
            raise RuntimeError(
                "A process-owned graph engine already exists; production code "
                "must acquire it through IntelligenceGraphEngine.get_or_create()"
            )
        # Use provided backend, or check for an active one, or create one from factory
        if backend is not None:
            self.backend = backend
        else:
            active_backend = get_active_backend()
            if active_backend is not None:
                self.backend = active_backend
            elif db_path:
                self.backend = create_backend(db_path=db_path)
            else:
                created_backend = create_backend()
                if created_backend is not None:
                    self.backend = created_backend
                else:
                    raise RuntimeError(
                        "A persistent graph backend is required. Memory-only mode is no longer supported."
                    )

        # Reuse the authority backend's own compute client when it has one.  An
        # EpistemicGraphBackend (including one wrapped by FanOutBackend) already
        # owns a GraphComputeEngine; constructing another here opened a second
        # socket and every write was then applied twice to the same authority.
        # Non-engine stores still receive one bounded compute scratch client.
        backend_graph = getattr(self.backend, "graph", None)
        self.graph_compute = (
            graph
            if graph is not None
            else (
                backend_graph
                if backend_graph is not None
                else GraphComputeEngine.get_or_create(backend_type="epistemic_graph")
            )
        )
        self.graph = self.graph_compute
        self._compute_is_authority = self.graph_compute is backend_graph
        self._process_owned = True

        # CONCEPT:AU-KG.backend.schedule-on-control-graph — bind the sole native
        # WorkItem authority and :Schedule store. The single-client production
        # profile shares one process-owned engine authority; graph-scoped profiles
        # bind its ``__control__`` view. Construction fails if neither contract is
        # available, preventing a silent second work-state location.
        self.control_backend = self._build_control_backend()

        super().__init__()

        # Start workers when native WorkItems report an ingestion backlog.
        if self.backend:
            try:
                if self.ingest_queue_depth() > 0:
                    self.start_task_workers()
            except Exception:
                logger.debug(
                    "Failed to start task workers on initialization", exc_info=True
                )

        with IntelligenceGraphEngine._ACTIVE_ENGINE_LOCK:
            active = IntelligenceGraphEngine._ACTIVE_ENGINE
            if active is None:
                IntelligenceGraphEngine._ACTIVE_ENGINE = self
            elif active is not self:
                raise RuntimeError(
                    "Concurrent duplicate graph engine construction was rejected"
                )
        # Model transport is evidence-governed process-wide. Register the
        # operational authority at the same lifecycle boundary as _ACTIVE_ENGINE.
        from agent_utilities.core.contextual_model import set_context_compiler_engine

        set_context_compiler_engine(self)
        self._bind_policy_stores()

        from ..retrieval.hybrid_retriever import HybridRetriever  # type: ignore
        from .inference_engine import InferenceEngine  # type: ignore

        # Resolve the active Schema Pack (explicit > env > config > core) and build
        # the retriever pack-aware so pack-driven retrieval signals (recency,
        # source-trust, autocut, relational-intent) are reachable (CONCEPT:AU-KG.ontology.schema-pack-lifecycle-audit).
        if schema_pack is None:
            try:
                from agent_utilities.models.schema_pack_loader import (
                    get_active_pack,
                    register_listener,
                )

                schema_pack = get_active_pack()
                register_listener(self._on_schema_pack_change)
            except Exception:  # pragma: no cover - never block engine construction
                schema_pack = None
        self.active_schema_pack = schema_pack

        self.hybrid_retriever = HybridRetriever(self, schema_pack=schema_pack)
        self.inference_engine = InferenceEngine(self)

        # CONCEPT:AU-ORCH.adapter.kg-graph-materialization — Auto-register service registry
        self._services_registered = False

    def _build_control_backend(self) -> GraphBackend:
        """Return the operational backend that owns native WorkItems.

        Native WorkItem lifecycle methods remain the only writable work-state
        There is no separate control store, graph client, or fallback authority.
        """
        return self.backend

    def _bind_policy_stores(self) -> None:
        """Bind mandatory policy state to the one authoritative graph backend."""

        from ..ontology.permissioning import set_marking_store

        set_marking_store(self.backend)

    def register_services(self) -> int:
        """Register all services with the KG for orchestrator discovery.

        CONCEPT:AU-ORCH.adapter.kg-graph-materialization — Unified Service Discovery

        Lazily initializes the ServiceRegistry and registers all concept
        modules as CallableResource nodes in the KG, enabling the
        TopologyEngine and KGTeamComposer to discover and invoke them.

        Returns:
            Number of services registered.
        """
        if self._services_registered:
            return 0

        try:
            from ...core.registry.service_adapter import ServiceRegistry

            registry = ServiceRegistry.instance()
            registry.initialize()
            count = registry.register_with_kg(self)
            self._services_registered = True
            logger.info(
                "[CONCEPT:AU-ORCH.adapter.kg-graph-materialization] Registered %d services with KG engine",
                count,
            )
            return count
        except Exception as e:
            logger.debug("Service registration failed: %s", e)
            return 0

    @property
    def _is_memory_only(self) -> bool:
        """True when no persistent backend exists (NX is both storage and compute)."""
        return False

    @property
    def memory(self):
        """Lazy-initialized MemoryEngine for the full memory lifecycle.

        Provides a single ergonomic entry point for:
          startup → active context → compaction → synthesis → retrieval
        """
        if not hasattr(self, "_memory_manager"):
            from ..memory import MemoryEngine

            self._memory_manager = MemoryEngine(engine=self)
        return self._memory_manager

    @classmethod
    def get_active(cls) -> IntelligenceGraphEngine | None:
        """Retrieve the currently active engine instance."""
        return cls._ACTIVE_ENGINE

    @classmethod
    def get_or_create(
        cls,
        factory: Any | None = None,
        **kwargs: Any,
    ) -> IntelligenceGraphEngine:
        """Acquire the one process-owned operational graph engine.

        ``factory`` is an explicit dependency-injection seam for tests and
        launchers. It executes at most once under the process lock. Public
        routes and background services use this method instead of constructing
        overlapping clients ad hoc.
        """
        active = cls._ACTIVE_ENGINE
        if active is not None:
            return active
        with cls._ACTIVE_ENGINE_LOCK:
            active = cls._ACTIVE_ENGINE
            if active is not None:
                return active
            created = factory() if factory is not None else cls(**kwargs)
            registered = cls._ACTIVE_ENGINE
            if registered is None:
                cls._ACTIVE_ENGINE = created
                registered = created
            elif registered is not created:
                raise RuntimeError(
                    "The graph client factory attempted to replace the process authority"
                )
            return registered

    def for_graph(self, graph_name: str) -> IntelligenceGraphEngine:
        """Return a graph-scoped facade over the one process client.

        The returned object is a lightweight view: no backend factory, socket,
        event-loop thread, mirror drainer, worker, or schema listener is
        created.  It is therefore safe for unified read fan-out and routed
        ingestion while preserving the one-client-per-process invariant.
        """
        target = str(graph_name or getattr(self.graph_compute, "graph_name", ""))
        if target == getattr(self.graph_compute, "graph_name", ""):
            return self

        backend_view_factory = getattr(self.backend, "for_graph", None)
        if not callable(backend_view_factory):
            authority = getattr(self.backend, "_authority", None)
            backend_view_factory = getattr(authority, "for_graph", None)
        if not callable(backend_view_factory):
            raise RuntimeError(
                f"backend {type(self.backend).__name__} has no named-graph view"
            )

        view = object.__new__(type(self))
        view.__dict__ = self.__dict__.copy()
        view.backend = backend_view_factory(target)
        # The backend view already owns the graph-compute view. Reuse that
        # exact object so authority identity remains true and facade writes are
        # not mirrored into a second wrapper over the same engine graph.
        view.graph_compute = getattr(view.backend, "graph", None)
        if view.graph_compute is None:
            view.graph_compute = self.graph_compute.for_graph(target)
        view.graph = view.graph_compute
        view._compute_is_authority = (
            getattr(view.backend, "graph", None) is view.graph_compute
        )
        view._process_owned = False
        view._process_root = getattr(self, "_process_root", self)

        from ..retrieval.hybrid_retriever import HybridRetriever
        from .inference_engine import InferenceEngine

        view.hybrid_retriever = HybridRetriever(
            view, schema_pack=getattr(self, "active_schema_pack", None)
        )
        view.inference_engine = InferenceEngine(view)
        return view

    @classmethod
    def set_active(cls, engine: IntelligenceGraphEngine | None):
        """Explicit dependency-injection/reset seam (primarily for tests)."""
        with cls._ACTIVE_ENGINE_LOCK:
            cls._ACTIVE_ENGINE = engine
        from agent_utilities.core.contextual_model import set_context_compiler_engine

        set_context_compiler_engine(engine)
        if engine is None:
            from ..ontology.permissioning import clear_markings

            clear_markings()
        else:
            engine._bind_policy_stores()

    def _normalize_label(self, label: str) -> str:
        """Find canonical case for a label from the schema. Delegates to the one
        materialization helper (CONCEPT:AU-KG.ingest.enterprise-source-extractor) — single source of truth."""
        from .materialization import normalize_label

        return normalize_label(label)

    def _get_allowed_columns(self, label: str) -> list[str]:
        """Get the list of allowed columns for a given node label from the schema."""
        label = self._normalize_label(label)
        try:
            from ...models.schema_definition import SCHEMA

            for node_def in SCHEMA.nodes:
                if node_def.name == label:
                    return list(node_def.columns.keys())
        except ImportError:
            pass
        return []

    def _serialize_node(self, node: Any, label: str | None = None) -> dict[str, Any]:
        """Serialize a Pydantic node for backend storage, handling Enums and JSON fields."""
        data = node.model_dump() if hasattr(node, "model_dump") else dict(node)
        clean_data = {}

        # Define fields that Ladybug supports as native arrays
        ARRAY_FIELDS = [
            "capabilities",
            "tags",
            "tool_ids",
            "success_criteria_met",
            "embedding",
            "issues",
        ]

        # Filter by schema if label is provided
        allowed_cols = self._get_allowed_columns(label) if label else None

        for k, v in data.items():
            if v is None:
                continue
            if allowed_cols is not None and k not in allowed_cols:
                continue

            if isinstance(v, Enum):
                clean_data[k] = v.value
            elif isinstance(v, dict | list) and k not in ARRAY_FIELDS:
                clean_data[k] = json.dumps(v)
            else:
                clean_data[k] = v
        return clean_data

    # Backends with a fixed, column-typed schema: writing a property that is
    # not a declared column is an error, so props must be filtered (extras are
    # routed to the catch-all ``metadata`` column). Schemaless backends
    # (epistemic_graph/neo4j/falkordb) accept arbitrary properties as-is.
    _SCHEMA_BACKED = {"LadybugBackend", "PostgreSQLBackend"}

    # Drivers that reject map/list-of-map property values (openCypher property
    # values must be primitives or arrays of primitives). Nested dict/list props
    # are JSON-encoded for these so a write remains portable across explicit mirrors
    # (CONCEPT:AU-KG.backend.mirror-health-repair).
    _NESTED_UNSAFE = {"Neo4jBackend", "FalkorDBBackend"}

    # Fields stored as native arrays (not JSON-encoded) across backends.
    _ARRAY_FIELDS = frozenset(
        {
            "capabilities",
            "tags",
            "tool_ids",
            "success_criteria_met",
            "embedding",
            "issues",
            # CONCEPT:AU-KG.ontology.capability-node-aliases-lexical — capability-node aliases the lexical gate matches;
            # kept native (not JSON-encoded) so it round-trips as a list on every
            # backend, including the nested-unsafe ones (neo4j/falkordb).
            "synonyms",
        }
    )

    def _schema_valid_keys(self, label: str | None) -> set[str] | None:
        """Declared columns for ``label`` on a schema-backed backend, else None.

        For an UNKNOWN label on LadybugDB/Kuzu (fixed typed tables), fall back to the
        canonical ``GENERIC_NODE_COLUMNS`` — the backend auto-creates a matching
        generic table, so the SET clause filters to those columns and ad-hoc props
        fold into ``metadata`` (otherwise the write references undeclared columns →
        Binder error → the node is dropped). The Postgres transpiler creates per-label
        tables dynamically, so it stays schemaless (None) for unknown labels.
        """
        from .materialization import schema_valid_keys

        return schema_valid_keys(self.backend, label)

    def _get_set_clause(
        self, data: dict[str, Any], alias: str = "n", label: str | None = None
    ) -> str:
        """Generate a SET clause for a Cypher query from a dictionary. Delegates to
        the one materialization helper (CONCEPT:AU-KG.ingest.enterprise-source-extractor) so SET-clause column
        filtering has a single implementation."""
        from .materialization import set_clause

        return set_clause(data, self.backend, alias, label)

    def _prepare_node_props(
        self, label: str | None, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Backend-aware serialization of node props for ``backend.execute``.

        Guarantees no property is silently dropped on a configured mirror and that no
        write throws on a map-rejecting driver:

        * schema-backed mirrors (Ladybug/PostgreSQL): keep declared columns and fold
          any ad-hoc keys into the ``metadata`` JSON column; JSON-encode nested
          declared values. Applied on BOTH update and create (previously only on
          create — re-writes silently dropped extras).
        * schemaless map-rejecting drivers (Neo4j/FalkorDB): JSON-encode nested
          dict/list props so the openCypher write never throws.
        * schemaless map-safe (epistemic_graph): pass through (native dicts ok).

        The full native property set still reaches the graph authority.
        """
        import json

        valid_keys = self._schema_valid_keys(label)
        backend_name = self.backend.__class__.__name__ if self.backend else ""
        nested_unsafe = backend_name in self._NESTED_UNSAFE

        def _enc(key: str, value: Any) -> Any:
            if isinstance(value, dict | list) and key not in self._ARRAY_FIELDS:
                return json.dumps(value, default=str)
            return value

        if valid_keys is None:
            # Schemaless backend: keep every property. Encode nested values only
            # for drivers that reject map properties.
            if not nested_unsafe:
                return dict(data)
            return {k: _enc(k, v) for k, v in data.items()}

        # Schema-backed: declared columns pass through (nested ones JSON-encoded);
        # everything else folds into the ``metadata`` catch-all column.
        prepared: dict[str, Any] = {}
        extras: dict[str, Any] = {}
        for k, v in data.items():
            if k == "id" or k in valid_keys:
                prepared[k] = _enc(k, v)
            elif k != "metadata":
                extras[k] = v

        if extras and "metadata" in valid_keys:
            meta: dict[str, Any] = {}
            existing = prepared.get("metadata")
            if isinstance(existing, str) and existing:
                try:
                    meta = json.loads(existing)
                except Exception:
                    meta = {"_": existing}
            meta.update(extras)
            prepared["metadata"] = json.dumps(meta, default=str)
        return prepared

    def _upsert_node(self, label: str, node_id: str, data: dict[str, Any]):
        """Perform an idempotent node upsert through the backend's native seam.

        Properties are folded/serialized ONCE (``_prepare_node_props``) and applied
        through the typed mutation API when Epistemic Graph is the authority.  This
        preserves arrays and nested values without compiling them into the native
        Cypher subset (whose ``SET`` grammar intentionally accepts scalar bare-name
        properties only).  Native proxy backends such as the governance guard and
        ``FanOutBackend`` expose the same typed seam; FanOut records the structured
        mutation in its mirror outbox.

        External Cypher stores retain the portable ``MERGE`` + ``SET`` statement.
        """
        if not self.backend:
            return None

        label = self._normalize_label(label)
        prepared = self._prepare_node_props(label, data)

        typed_support = getattr(self.backend, "typed_mutation_support", "")
        if typed_support == "native":
            typed_add = getattr(self.backend, "add_node", None)
            if not callable(typed_add):
                raise RuntimeError(
                    "native graph authority does not expose typed node mutations"
                )
            typed_add(
                node_id,
                **{
                    **prepared,
                    "id": node_id,
                    "node_type": label,
                },
            )
            return None
        if getattr(self.backend, "cypher_support", "full") == "native":
            raise RuntimeError(
                "native graph authority did not declare lossless typed mutations"
            )

        set_clause = self._get_set_clause(prepared, label=label)
        merge_query = f"MERGE (n:{label} {{id: $id}}) {set_clause}".rstrip()
        return self.backend.execute(merge_query, prepared)

    def link_nodes(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
        *,
        session: GraphSession | None = None,
    ):
        """Create a relationship between two nodes in the graph.

        Write ordering follows the configured authority contract. When a native
        graph view is distinct from that backend, it is updated after the
        authoritative write succeeds.

        Args:
            session: Explicit :class:`~.session.GraphSession` this write runs
                under (CONCEPT:AU-P0-1). See :meth:`add_node` for the ambient-actor
                scoping this enables downstream.
        """
        from agent_utilities.security.brain_context import use_actor

        from .session import resolve_session

        session = resolve_session(session, required_scope="kg:write")

        if rel_type:
            # Flag edge types outside an EXCLUSIVE pack before normalising case
            # (observe-only; no-op under the default core pack) (CONCEPT:AU-KG.ontology.schema-pack-lifecycle-audit).
            self._audit_candidate_type("edge", str(rel_type))
            rel_type = rel_type.upper()
        props = properties or {}
        # Inject lightweight provenance/confidence tags for structural memory
        if "confidence" not in props:
            props["confidence"] = 1.0
        if "source" not in props:
            props["source"] = "system"
        # CONCEPT:AU-KG.temporal.bi-temporal-memory-layers — Bi-Temporal Memory. Stamp event_time / storage_time /
        # valid_from / valid_to so edges support as-of queries and event-time contradiction
        # precedence (extends the prior Graphiti-inspired valid_from). A caller-supplied
        # event_time (e.g. a narrative date resolved by the learner) is preserved.
        from agent_utilities.knowledge_graph.core.bitemporal import stamp_bitemporal

        stamp_bitemporal(props, event_time=props.get("event_time"))

        with use_actor(session.actor):
            if self.backend and not ephemeral:
                # The configured backend is the write authority for this operation.
                self._upsert_edge(source_id, target_id, rel_type, props)

            # An EpistemicGraphBackend write already passed through this exact
            # compute client. Other durable stores retain a bounded scratchpad.
            if ephemeral or not self._compute_is_authority:
                self.graph_compute.add_edge(
                    source_id, target_id, {"relationship": rel_type, **props}
                )

    def _upsert_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        props: dict[str, Any],
        *,
        source_label: str | None = None,
        target_label: str | None = None,
    ) -> None:
        """Idempotent edge upsert via MERGE — writes ONLY to ``self.backend`` (no
        graph_compute), the portable counterpart to :meth:`_upsert_node`. Reused by
        the cross-backend migration so every durable backend gets a dialect-correct
        edge write. The caller normalises ``rel_type`` (upper) and stamps ``props``.

        Kuzu/Ladybug REL tables carry a single JSON ``properties`` column (other
        backends store edge props as native columns/JSONB), so fold all edge props
        into it there — otherwise Kuzu drops them.

        ``source_label`` / ``target_label``: when the caller already knows the
        endpoint labels (the bulk migration does — it just wrote the nodes), pass
        them to skip the two ``MATCH (n) WHERE n.id=$id`` label-lookup round-trips
        AND to emit a *labelled* ``MATCH`` an ``:Label(id)`` index can serve. Without
        them the unlabelled match full-scans — O(n) per edge, fatal at scale. Omitted
        on the live single-write path, which still looks the labels up.
        """
        if not self.backend:
            return

        # Epistemic Graph's typed mutation carries the complete property value
        # domain (including arrays/nested values) and avoids two label-lookup
        # queries plus a scalar-only Cypher ``SET``.  Native proxy backends retain
        # governance and mirror-outbox behavior through this same method.
        typed_support = getattr(self.backend, "typed_mutation_support", "")
        if typed_support == "native":
            typed_add = getattr(self.backend, "add_edge", None)
            if not callable(typed_add):
                raise RuntimeError(
                    "native graph authority does not expose typed edge mutations"
                )
            typed_add(
                source_id,
                target_id,
                **{
                    **props,
                    "relationship": rel_type,
                },
            )
            return
        if getattr(self.backend, "cypher_support", "full") == "native":
            raise RuntimeError(
                "native graph authority did not declare lossless typed mutations"
            )

        _backend_name = self.backend.__class__.__name__
        if _backend_name == "LadybugBackend":
            set_clause = " SET r.`properties` = $properties"
            edge_params: dict[str, Any] = {"properties": json.dumps(props, default=str)}
        else:
            set_clause = self._get_set_clause(props, alias="r")
            edge_params = dict(props)

        # Portable label lookup. Dialect is chosen from the backend's *declared
        # capability* (``cypher_support``), NEVER its class name: full-openCypher
        # stores (Neo4j/FalkorDB/Apache AGE) expose ``labels(n)`` (a LIST) and reject
        # the singular ``label(n)`` ("Unknown function 'label'"); the bounded-subset
        # stores (epistemic-graph / pggraph regex transpiler) use ``label(n)``. Using
        # the capability — the same signal ``migration.py`` uses — is correct THROUGH
        # a wrapper (a FanOutBackend delegates ``cypher_support`` to its authority), so
        # a Neo4j/FalkorDB mirror or authority reached via the fan-out no longer gets a
        # broken ``label(n)`` query that retries forever (CONCEPT:AU-KG.backend.mirror-health-repair). The
        # class-name ``_NESTED_UNSAFE`` set stays for nested-prop JSON encoding only.
        # Skipped entirely when the caller supplies the labels (bulk migration) — the
        # labels are normalised the same way the nodes were written so the indexed
        # ``MATCH (s:Label {id})`` resolves.
        if source_label is not None and target_label is not None:
            s_label = f":{self._normalize_label(source_label)}"
            t_label = f":{self._normalize_label(target_label)}"
        else:
            _full_cypher = getattr(self.backend, "cypher_support", "full") == "full"
            _lbl_expr = "labels(n)[0]" if _full_cypher else "label(n)"
            s_label_res = self.backend.execute(
                f"MATCH (n) WHERE n.id = $id RETURN {_lbl_expr} as lbl",
                {"id": source_id},
            )
            t_label_res = self.backend.execute(
                f"MATCH (n) WHERE n.id = $id RETURN {_lbl_expr} as lbl",
                {"id": target_id},
            )
            s_label = (
                f":{s_label_res[0]['lbl']}"
                if s_label_res and s_label_res[0].get("lbl")
                else ""
            )
            t_label = (
                f":{t_label_res[0]['lbl']}"
                if t_label_res and t_label_res[0].get("lbl")
                else ""
            )
        query = (
            f"MATCH (s{s_label} {{id: $sid}}) MATCH (t{t_label} {{id: $tid}}) "
            f"MERGE (s)-[r:{rel_type}]->(t){set_clause}"
        )
        params = {"sid": source_id, "tid": target_id}
        params.update(edge_params)
        self.backend.execute(query, params)

    def resolve_and_link(
        self,
        source_name: str,
        target_name: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
        *,
        session: GraphSession | None = None,
    ) -> bool:
        """Lightweight cross-entity relationship resolution.

        Attempts to resolve source and target names to existing node IDs using
        string matching before linking them. If backend is present, this is pushed
        down to Cypher to avoid O(N) memory scans on large enterprise graphs.
        """

        from agent_utilities.security.brain_context import use_actor

        from .session import resolve_session

        session = resolve_session(session, required_scope="kg:write")
        if self.backend and not ephemeral:
            # Push-down resolution to backend via CONTAINS to avoid O(N) memory scan
            props = properties or {}
            set_clause = self._get_set_clause(props, alias="r")
            q = f"""
            MATCH (s) WHERE toLower(s.name) CONTAINS toLower($source) OR toLower($source) CONTAINS toLower(s.name)
            MATCH (t) WHERE toLower(t.name) CONTAINS toLower($target) OR toLower($target) CONTAINS toLower(t.name)
            WITH s, t LIMIT 1
            MERGE (s)-[r:{rel_type}]->(t){set_clause}
            RETURN s.id AS sid, t.id AS tid
            """
            params = {
                "source": source_name,
                "target": target_name,
            }
            params.update(props)
            with use_actor(session.actor):
                res = self.backend.execute(q, params)
            return len(res) > 0

        return False

    def _on_schema_pack_change(self, pack: Any) -> None:
        """Rewire the engine when the active Schema Pack changes (CONCEPT:AU-KG.ontology.schema-pack-lifecycle-audit).

        Rebuilds the retriever so the new pack's retrieval signals take effect
        immediately; the fresh retriever carries the new ``pack.signature()`` so a
        prior pack's boosted/cut results can never be served after a switch.
        """
        self.active_schema_pack = pack
        try:
            from ..retrieval.hybrid_retriever import HybridRetriever

            self.hybrid_retriever = HybridRetriever(self, schema_pack=pack)
        except Exception:  # pragma: no cover - best-effort rewire
            logger.debug(
                "Failed to rewire retriever for new schema pack", exc_info=True
            )

    def _audit_candidate_type(
        self, kind: Literal["node", "edge"], type_name: str
    ) -> None:
        """Record an out-of-pack node/edge type under an EXCLUSIVE pack (KG-2.35).

        Observe-only: never raises, never blocks the write. A no-op under the
        default ADDITIVE ``core`` pack (where every type is active).
        """
        pack = getattr(self, "active_schema_pack", None)
        if pack is None:
            return
        try:
            from agent_utilities.models.schema_pack import SchemaPackMode

            if pack.mode != SchemaPackMode.EXCLUSIVE:
                return
            if kind == "node":
                active = {str(t).lower() for t in pack.get_active_node_types()}
            else:
                active = {str(t).lower() for t in pack.get_active_edge_types()}
            if type_name.lower() not in active:
                from agent_utilities.models.schema_pack_audit import (
                    SchemaCandidateAuditor,
                )

                SchemaCandidateAuditor.instance().record(kind, type_name, pack.name)
        except Exception:  # pragma: no cover - audit must never break writes
            pass

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
        *,
        session: GraphSession | None = None,
    ):
        """Add a generic node to the graph.

        This is a convenience method for code that doesn't have a typed
        Pydantic model (e.g. council verdicts, ad-hoc decision nodes).

        Write ordering: backend-first, then graph_compute.

        Args:
            session: Explicit :class:`~.session.GraphSession` this write runs
                under (CONCEPT:AU-P0-1 — the one currency). When omitted, one is derived
                from today's ambient actor via ``GraphSession.from_ambient()``.
                The write executes with that session's actor scoped ambiently
                (``use_actor``) for its duration, so any provenance/ownership
                stamping downstream that still reads the ambient actor
                (e.g. the write-path Company Brain guard) attributes correctly
                — existing callers that never pass ``session`` see identical
                behaviour.
        """
        from agent_utilities.security.brain_context import use_actor

        from .session import resolve_session

        session = resolve_session(session, required_scope="kg:write")

        node_type = self._normalize_label(node_type)
        props = dict(properties or {})
        if "type" in props:
            raise ValueError(
                "node property 'type' is retired; use the node_type argument"
            )
        props["node_type"] = node_type
        # Flag types outside an EXCLUSIVE pack, observe-only (CONCEPT:AU-KG.ontology.schema-pack-lifecycle-audit).
        self._audit_candidate_type("node", node_type)

        with use_actor(session.actor):
            result: Any = None
            if self.backend and not ephemeral:
                # The configured backend is the write authority for this operation.
                data = {"id": node_id, **props}
                result = self._upsert_node(node_type, node_id, data)

            if ephemeral or not self._compute_is_authority:
                self.graph_compute.add_node(node_id, props)
            return result if result is not None else {"id": node_id, **props}

    def add_edge(
        self,
        source: str,
        target: str,
        rel_type: str = "",
        ephemeral: bool = False,
        *,
        session: GraphSession | None = None,
        **properties: Any,
    ) -> None:
        """Add a generic edge between two nodes (backend-first, then graph_compute).

        Convenience for ad-hoc relationships (e.g. provenance links like
        ``RunTrace -[:HAS_CONTEXT]-> ContextBlob``, CONCEPT:AU-ORCH.session.invoker-agent-handoff) where there is no typed
        model. Best-effort: a missing backend/compute method is tolerated.

        Args:
            session: Explicit :class:`~.session.GraphSession` this write runs
                under (CONCEPT:AU-P0-1). See :meth:`add_node` for the ambient-actor
                scoping this enables downstream.
        """
        from agent_utilities.security.brain_context import use_actor

        from .session import resolve_session

        session = resolve_session(session, required_scope="kg:write")

        if not str(rel_type).strip():
            raise ValueError("rel_type is required")

        aliases = {"type", "rel_type", "relationship_type", "relation"}.intersection(
            properties
        )
        if aliases:
            names = ", ".join(sorted(aliases))
            raise ValueError(
                f"edge relationship aliases are retired ({names}); "
                "use the rel_type argument"
            )

        with use_actor(session.actor):
            if self.backend and not ephemeral:
                _be = getattr(self.backend, "add_edge", None)
                if callable(_be):
                    with contextlib.suppress(Exception):
                        _be(source, target, rel_type, **properties)
            _ge = getattr(self.graph_compute, "add_edge", None)
            if callable(_ge) and (ephemeral or not self._compute_is_authority):
                with contextlib.suppress(Exception):
                    _ge(source, target, {"relationship": rel_type, **properties})

    def get_blast_radius(
        self,
        node_id: str,
        depth: int,
        *,
        session: GraphSession | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the blast radius (dependencies) from a starting node.

        Uses the high-performance GraphComputeEngine with compiled Rust and
        epistemic-graph backend for fast traversals.
        """
        from agent_utilities.security.brain_context import use_actor

        from .session import resolve_session

        session = resolve_session(session, required_scope="kg:read")
        if self.backend:
            # Cypher-powered query: handles millions of nodes directly in the database
            query = f"""
            MATCH (s {{id: $node_id}})-[*1..{depth}]->(t)
            WITH s, t, shortestPath((s)-[*1..{depth}]->(t)) as p
            RETURN distinct t.id as id, labels(t)[0] as node_type, length(p) as depth
            """
            try:
                with use_actor(session.actor):
                    results = self.backend.execute(query, {"node_id": node_id})
                return [
                    {
                        "id": r["id"],
                        "node_type": r.get("node_type", "Node"),
                        "depth": r["depth"],
                    }
                    for r in results
                ]
            except Exception as e:
                logger.warning(
                    f"Cypher blast radius query failed: {e}. Falling back to compute engine."
                )

        return self.graph_compute.get_blast_radius(node_id, depth)

    def register_materialization(self, derived_id: str) -> dict[str, Any]:
        """Register ``derived_id`` as a live engine-side TruthMaintenance
        materialization (CONCEPT:EG-KG.epistemic.truth-maintenance, Seam 3 — X-6
        across the storage boundary): the engine reads ``derived_id``'s OWN
        already-stored provenance (its ``invalidation_deps`` property plus any
        outgoing ``:DerivedFrom``/``:GeneratedBy`` edge) into a dependency set and
        tracks it so ANY subsequent committed change to a dependency (through the
        normal write path) automatically marks it stale, with no polling. Call
        this ONCE, right after writing a derived node (a mined claim, a computed
        capability index entry, ...) plus its provenance edges. Thin passthrough
        to :meth:`GraphComputeEngine.register_materialization`; requires an engine
        built with the ``epistemic-tms`` feature (opt-in, not part of ``full``).
        """
        return self.graph_compute.register_materialization(derived_id)

    def materialization_status(self, id: str) -> str | None:
        """Current status (``"Fresh"``/``"Stale"``/``"Retracted"``, or ``None`` if
        never registered) of a materialization tracked on the same index
        :meth:`register_materialization` writes to. Thin passthrough to
        :meth:`GraphComputeEngine.materialization_status`."""
        return self.graph_compute.materialization_status(id)

    # --- Background Analysis Methods ---

    def execute_deep_analysis(self, query: str, max_depth: int = 2) -> dict[str, Any]:
        """Perform a native background deep analysis of a concept.

        Architecture (native signals plus free-text synthesis):
            - **Structured discovery**: ``discover_innovations`` provides
              structured domain signals, scores, biomimicry mappings, and
              innovation claims natively — no LLM needed.
            - **Free-text synthesis**: The LLM generates a natural-language
              synthesis summary. This plays to any model's strength (text gen)
              and eliminates JSON schema validation failures entirely.
            - **KG writeback**: Native domain recommendations are written as
              ``ANALOGOUS_TO`` edges. The LLM summary is stored as a semantic
              ``Memory`` node for future retrieval.
        """

        from agent_utilities.core.config import (
            DEFAULT_KG_MODEL_ID,
            DEFAULT_LLM_PROVIDER,
        )
        from agent_utilities.core.contextual_model import create_context_agent
        from agent_utilities.core.model_factory import create_model

        # Structured discovery (no LLM).
        l1_results = self.discover_innovations(query, top_k=10)
        enriched = l1_results.get("results", [])
        domain_recs = l1_results.get("domain_recommendations", [])
        if not enriched:
            return {"status": "skipped", "reason": "No initial concepts found"}

        # Build compact context for the LLM from native signals.
        match_lines = []
        for r in enriched[:7]:
            match_lines.append(
                f"- **{r.get('name', r.get('id', ''))}** "
                f"(score={r.get('score', 0):.3f}, signals={r.get('total_signal_count', 0)})"
            )
            for claim in r.get("innovation_claims", [])[:2]:
                match_lines.append(f"  > {claim[:250]}")
            for sig in r.get("tech_signals", [])[:3]:
                match_lines.append(
                    f"  ↳ {sig['keyword']}: {sig['analogy']} → {sig['domain']}"
                )

        domain_lines = []
        for d in domain_recs[:10]:
            domain_lines.append(
                f"- **{d['domain']}** ({d['analogy']}) — "
                f"{d['source_count']} signals, priority={d['priority']}"
            )

        prompt = (
            f"## Deep Analysis: {query}\n\n"
            f"### Top Matches from Knowledge Graph\n"
            + "\n".join(match_lines)
            + "\n\n### Domain Recommendations (by signal frequency)\n"
            + "\n".join(domain_lines)
            + "\n\n---\n\n"
            "Based on these research paper matches and domain signals, write a "
            "detailed synthesis covering:\n"
            "1. **Key Features to Implement**: Name each feature, explain what it does, "
            "and which domain(s) it maps to.\n"
            "2. **Implementation Priorities**: Rank features by expected impact.\n"
            "3. **Cross-Domain Connections**: Identify non-obvious connections between "
            "different research papers or domains.\n"
            "4. **Architectural Recommendations**: Suggest concrete integration points.\n\n"
            "Write in clear, structured markdown. Be specific and actionable."
        )

        # Free-text LLM synthesis.
        llm_summary = ""
        try:
            from ...core.event_loop import allow_nested_run_sync

            allow_nested_run_sync()

            model = create_model(
                provider=DEFAULT_LLM_PROVIDER, model_id=DEFAULT_KG_MODEL_ID
            )
            agent = create_context_agent(
                model,
                system_prompt=(
                    "You are an expert software architect analyzing research papers "
                    "and codebases. Produce clear, actionable synthesis reports in "
                    "structured markdown. Focus on practical implementation guidance."
                ),
            )

            result = agent.run_sync(prompt)
            llm_summary = str(result.output)
            logger.info("Synthesis complete: %d chars generated", len(llm_summary))
        except Exception as e:
            logger.warning("LLM synthesis failed (non-fatal): %s", e)
            llm_summary = (
                f"[LLM synthesis unavailable — native signals preserved]\n\n"
                f"Query: {query}\n"
                f"Matches: {len(enriched)}\n"
                f"Top domains: {', '.join(d['domain'] for d in domain_recs[:5])}"
            )

        # ── KG Writeback: Domain edges + Memory node ─────────────────
        source_id = (
            query if "-" in query else (enriched[0].get("id") if enriched else query)
        )

        new_concepts = []
        # Write ANALOGOUS_TO edges from native domain recommendations.
        for d in domain_recs:
            if d.get("priority") in ("high", "medium"):
                success = self.resolve_and_link(
                    source_name=source_id,
                    target_name=d["domain"],
                    rel_type="ANALOGOUS_TO",
                    properties={
                        "source": "deep_analysis",
                        "feature": d["analogy"],
                        "signal_count": d.get("source_count", 0),
                        "priority": d["priority"],
                    },
                )
                if success:
                    new_concepts.append(d["domain"])

        # Store synthesis as a semantic memory for future recall
        try:
            self.add_memory(
                content=llm_summary,
                category="deep_analysis",
                tags=["synthesis", query],
            )
        except Exception as mem_e:
            logger.debug(f"Memory store skipped: {mem_e}")

        return {
            "status": "success",
            "features_extracted": len(domain_recs),
            "new_analogies": len(new_concepts),
            "discovered_targets": new_concepts,
            "llm_summary_length": len(llm_summary),
            "llm_summary": llm_summary[:2000],
        }

    async def run(self, manifest: Any) -> Any:
        """Unified ExecutionEngine contract entrypoint.

        Plan 03 Step 5 — conforms to ``core.execution.ExecutionEngine``.
        Additive adapter: normalises ``manifest`` to a query string and runs
        the engine's native deep-analysis pipeline (:meth:`execute_deep_analysis`),
        returning a canonical ``ExecutionResult``. Existing behaviour and all
        other public methods are unchanged.
        """
        from agent_utilities.core.execution.models import ExecutionResult

        query = manifest if isinstance(manifest, str) else ""
        manifest_id = ""
        if not query:
            query = getattr(manifest, "query", "") or ""
            manifest_id = getattr(manifest, "manifest_id", "") or ""

        analysis = await asyncio.to_thread(self.execute_deep_analysis, query)
        return ExecutionResult(
            manifest_id=manifest_id,
            synthesis_output=analysis.get("llm_summary", ""),
            success=analysis.get("status") == "success",
        )

    def delete_node(
        self,
        node_id: str,
        ephemeral: bool = False,
        *,
        session: GraphSession | None = None,
    ) -> None:
        """Remove a node through the session-owned graph authority."""
        from agent_utilities.security.brain_context import use_actor

        from .session import resolve_session

        session = resolve_session(session, required_scope="kg:write")
        with use_actor(session.actor):
            if self.backend and not ephemeral:
                try:
                    self.backend.execute(
                        "MATCH (n {id: $id}) DETACH DELETE n",
                        {"id": node_id},
                    )
                except Exception as e:
                    logger.warning(f"Backend delete_node failed: {e}")

            if ephemeral or not self._compute_is_authority:
                try:
                    self.graph_compute.remove_node(node_id)
                except Exception as e:
                    logger.debug(
                        f"graph_compute remove_node failed or node not found: {e}"
                    )

    def remove_node(
        self,
        node_id: str,
        ephemeral: bool = False,
        *,
        session: GraphSession | None = None,
    ) -> None:
        """Remove a node — delegates to :meth:`delete_node`."""
        self.delete_node(node_id, ephemeral, session=session)

    def delete_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str | None = None,
        ephemeral: bool = False,
        *,
        session: GraphSession | None = None,
    ) -> None:
        """Remove a relationship through the session-owned graph authority."""
        from agent_utilities.security.brain_context import use_actor

        from .session import resolve_session

        session = resolve_session(session, required_scope="kg:write")
        with use_actor(session.actor):
            if self.backend and not ephemeral:
                try:
                    if rel_type:
                        rel_type = rel_type.upper()
                        query = (
                            f"MATCH (s {{id: $sid}})-[r:{rel_type}]->(t {{id: $tid}}) "
                            "DELETE r"
                        )
                    else:
                        query = "MATCH (s {id: $sid})-[r]->(t {id: $tid}) DELETE r"
                    self.backend.execute(
                        query, {"sid": source_id, "tid": target_id}
                    )
                except Exception as e:
                    logger.warning(f"Backend delete_edge failed: {e}")

            if ephemeral or not self._compute_is_authority:
                try:
                    self.graph_compute.remove_edge(source_id, target_id)
                except Exception as e:
                    logger.debug(
                        f"graph_compute remove_edge failed or edge not found: {e}"
                    )
