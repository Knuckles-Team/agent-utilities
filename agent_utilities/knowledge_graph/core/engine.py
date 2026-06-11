#!/usr/bin/python
from __future__ import annotations

"""Unified Intelligence Graph Engine — Tiered Architecture.

This module provides the high-level interface for querying the unified knowledge graph,
supporting structural Cypher queries, topological impact analysis, and hybrid search.

Architecture (Two-Tier Graph Engine):
    - **Tier 1 (Source of Truth)**: A persistent Cypher-capable backend
      (LadybugDB/Neo4j/PostgreSQL) handles all CRUD, schema enforcement,
      vector indexing, and filtered queries.
    - **Tier 2 (Compute Scratchpad)**: GraphComputeEngine (Rust-native) is
      loaded on-demand via ``load_subgraph()`` for graph algorithms (PageRank,
      VF2, spectral clustering, causal reasoning) that databases cannot
      perform natively.

    When no persistent backend is available (``GRAPH_BACKEND=memory``),
    the engine falls back to using ``self.graph`` (GraphComputeEngine) as
    both storage and compute — suitable for testing and small graphs only.

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
import os
from enum import Enum, StrEnum
from typing import Any, Literal

from ...core.registry.kg_adapter import FocusedSubgraph, RegistryMixin
from ..backends import create_backend, get_active_backend
from ..backends.base import GraphBackend
from ..orchestration.engine_ahe import AHEMixin
from ..orchestration.engine_federation import FederationMixin
from ..orchestration.engine_infra import InfrastructureEngineMixin
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


class RoutingStrategy(StrEnum):
    """Controls how queries are routed between the persistent backend and the
    Rust compute engine.

    Set via ``GRAPH_ROUTING_STRATEGY`` environment variable.
    """

    BACKEND = "backend"  # All ops → LadybugDB only
    COMPUTE = "compute"  # All ops → Rust only (testing/ephemeral)
    HYBRID = "hybrid"  # Writes → backend-first, reads → compute-first


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
):
    """Engine for querying the unified intelligence graph (Agents, Tools, Code, Memory).

    Composed of focused mixins for maintainability. All 49+ existing importers
    continue to work since IntelligenceGraphEngine is still the single public class.

    Tiered Architecture:
        - Writes go to the persistent backend (Tier 1) when available.
        - ``self.graph`` (GraphComputeEngine, Rust-native) is the fallback
          when no backend exists, AND the compute scratchpad for graph
          algorithms via ``load_subgraph()``.
        - Dual-writes are avoided to prevent OOM at enterprise scale (100K+ nodes).
    """

    _ACTIVE_ENGINE: IntelligenceGraphEngine | None = None

    def __init__(
        self,
        backend: GraphBackend | None = None,
        db_path: str | None = None,
        external_ontologies: list[str] | None = None,
        graph: Any = None,
        schema_pack: Any = None,
    ):
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

        # Initialize compiled / optimized Graph Compute Engine (Rust/epistemic-graph)
        self.graph_compute = (
            graph
            if graph is not None
            else GraphComputeEngine(backend_type="epistemic_graph")
        )
        self.graph = self.graph_compute

        strategy_str = os.getenv("GRAPH_ROUTING_STRATEGY", "hybrid").lower()
        try:
            self.routing_strategy = RoutingStrategy(strategy_str)
        except ValueError:
            logger.warning(
                "Unknown GRAPH_ROUTING_STRATEGY=%r, defaulting to hybrid", strategy_str
            )
            self.routing_strategy = RoutingStrategy.HYBRID

        super().__init__()

        # Start workers if there are pending tasks in the database natively
        if self.backend:
            try:
                # Lightweight check to avoid locking if queue is empty
                pending = self.query_cypher(
                    "MATCH (t:Task {status: 'pending'}) RETURN count(t) as c"
                )
                if pending and pending[0]["c"] > 0:
                    self.start_task_workers()
            except Exception:
                logger.debug(
                    "Failed to start task workers on initialization", exc_info=True
                )

        # Auto-register as active if none exists to support singleton pattern
        if IntelligenceGraphEngine._ACTIVE_ENGINE is None:
            IntelligenceGraphEngine._ACTIVE_ENGINE = self

        from ..retrieval.hybrid_retriever import HybridRetriever  # type: ignore
        from .inference_engine import InferenceEngine  # type: ignore

        # Resolve the active Schema Pack (explicit > env > config > core) and build
        # the retriever pack-aware so pack-driven retrieval signals (recency,
        # source-trust, autocut, relational-intent) are reachable (CONCEPT:KG-2.35).
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

        # CONCEPT:ORCH-1.4 — Auto-register service registry
        self._services_registered = False

    def register_services(self) -> int:
        """Register all services with the KG for orchestrator discovery.

        CONCEPT:ORCH-1.4 — Unified Service Discovery

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
                "[CONCEPT:ORCH-1.4] Registered %d services with KG engine", count
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
    def set_active(cls, engine: IntelligenceGraphEngine | None):
        """Explicitly set the active engine instance."""
        cls._ACTIVE_ENGINE = engine

    def _normalize_label(self, label: str) -> str:
        """Find canonical case for a label from the schema."""
        if not label:
            return label
        try:
            from ...models.schema_definition import SCHEMA

            for node_def in SCHEMA.nodes:
                if node_def.name.lower() == label.lower():
                    return node_def.name
        except ImportError:
            pass
        return label

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
    _SCHEMA_BACKED = {"LadybugBackend", "PostgreSQLBackend", "TieredGraphBackend"}

    def _schema_valid_keys(self, label: str | None) -> set[str] | None:
        """Declared columns for ``label`` on a schema-backed backend, else None."""
        if (
            not self.backend
            or self.backend.__class__.__name__ not in self._SCHEMA_BACKED
            or not label
        ):
            return None
        from agent_utilities.models.schema_definition import SCHEMA

        for node in SCHEMA.nodes:
            if node.name == label:
                return set(node.columns.keys())
        return None

    def _get_set_clause(
        self, data: dict[str, Any], alias: str = "n", label: str | None = None
    ) -> str:
        """Generate a SET clause for a Cypher query from a dictionary.

        On schema-backed backends, properties are filtered to declared columns.
        """
        if label:
            label = self._normalize_label(label)

        # Relationship tables have no properties in our current schema definition.
        if (
            alias == "r"
            and self.backend
            and (self.backend.__class__.__name__ in ("LadybugBackend",))
        ):
            return ""

        valid_keys = self._schema_valid_keys(label)

        sets = []
        for k in data.keys():
            if k == "id":
                continue
            if valid_keys is not None and k not in valid_keys:
                continue
            sets.append(f"{alias}.`{k}` = ${k}")
        return " SET " + ", ".join(sets) if sets else ""

    def _upsert_node(self, label: str, node_id: str, data: dict[str, Any]):
        """Perform an idempotent upsert of a node using MATCH/SET then CREATE."""
        if not self.backend:
            return

        label = self._normalize_label(label)

        # 1. Try to update existing
        set_clause = self._get_set_clause(data, label=label)
        update_query = f"MATCH (n:{label}) WHERE n.id = $id {set_clause} RETURN n.id"
        res = self.backend.execute(update_query, data)

        if not res:
            # 2. If not found, create. On schema-backed backends, filter props to
            # declared columns and fold any extras into the ``metadata`` column so
            # no data is lost on the durable tier (the L1 compute graph still gets
            # the full property set via ``graph_compute.add_node``).
            valid_keys = self._schema_valid_keys(label)

            create_data: dict[str, Any] = {}
            extras: dict[str, Any] = {}
            for k, v in data.items():
                if k == "id" or valid_keys is None or k in valid_keys:
                    create_data[k] = v
                elif k != "metadata":
                    extras[k] = v

            if extras and valid_keys is not None and "metadata" in valid_keys:
                import json

                meta: dict[str, Any] = {}
                existing = create_data.get("metadata")
                if isinstance(existing, str) and existing:
                    try:
                        meta = json.loads(existing)
                    except Exception:
                        meta = {"_": existing}
                elif isinstance(existing, dict):
                    meta = dict(existing)
                meta.update(extras)
                create_data["metadata"] = json.dumps(meta, default=str)

            cols = ", ".join([f"`{k}`: ${k}" for k in create_data.keys()])
            create_query = f"CREATE (n:{label} {{{cols}}})"
            self.backend.execute(create_query, create_data)

    def link_nodes(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
    ):
        """Create a relationship between two nodes in the graph.

        Write ordering: backend-first, then graph_compute.
        This prevents sync drift — the persistent layer is always the
        canonical state. The compute scratchpad is updated afterward
        as a real-time cache.
        """
        if rel_type:
            # Flag edge types outside an EXCLUSIVE pack before normalising case
            # (observe-only; no-op under the default core pack) (CONCEPT:KG-2.35).
            self._audit_candidate_type("edge", str(rel_type))
            rel_type = rel_type.upper()
        props = properties or {}
        # Inject lightweight provenance/confidence tags for structural memory
        if "confidence" not in props:
            props["confidence"] = 1.0
        if "source" not in props:
            props["source"] = "system"
        # CONCEPT:KG-2.11 — Bi-Temporal Memory. Stamp event_time / storage_time /
        # valid_from / valid_to so edges support as-of queries and event-time contradiction
        # precedence (extends the prior Graphiti-inspired valid_from). A caller-supplied
        # event_time (e.g. a narrative date resolved by the learner) is preserved.
        from agent_utilities.knowledge_graph.core.bitemporal import stamp_bitemporal

        stamp_bitemporal(props, event_time=props.get("event_time"))

        if self.backend and not ephemeral:
            # Tier 1: Backend is source of truth — write here FIRST
            set_clause = self._get_set_clause(props, alias="r")

            s_query = "MATCH (n) WHERE n.id = $id RETURN label(n) as lbl"
            t_query = "MATCH (n) WHERE n.id = $id RETURN label(n) as lbl"

            s_label_res = self.backend.execute(s_query, {"id": source_id})
            t_label_res = self.backend.execute(t_query, {"id": target_id})
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
                f"MATCH (s{s_label} {{id: $sid}}), (t{t_label} {{id: $tid}}) "
                f"MERGE (s)-[r:{rel_type}]->(t){set_clause}"
            )
            params = {"sid": source_id, "tid": target_id}
            params.update(props)
            self.backend.execute(query, params)

        # Tier 2: Update graph_compute cache after backend succeeds
        self.graph_compute.add_edge(source_id, target_id, {"type": rel_type, **props})

    def resolve_and_link(
        self,
        source_name: str,
        target_name: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
    ) -> bool:
        """Lightweight cross-entity relationship resolution.

        Attempts to resolve source and target names to existing node IDs using
        string matching before linking them. If backend is present, this is pushed
        down to Cypher to avoid O(N) memory scans on large enterprise graphs.
        """

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
            res = self.backend.execute(q, params)
            return len(res) > 0

        return False

    def _on_schema_pack_change(self, pack: Any) -> None:
        """Rewire the engine when the active Schema Pack changes (CONCEPT:KG-2.35).

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
    ):
        """Add a generic node to the graph.

        This is a convenience method for code that doesn't have a typed
        Pydantic model (e.g. council verdicts, ad-hoc decision nodes).

        Write ordering: backend-first, then graph_compute.
        """
        node_type = self._normalize_label(node_type)
        props = properties or {}
        props["type"] = node_type
        # Flag types outside an EXCLUSIVE pack, observe-only (CONCEPT:KG-2.35).
        self._audit_candidate_type("node", node_type)

        if self.backend and not ephemeral:
            # Tier 1: Backend is source of truth — write here FIRST
            data = {"id": node_id, **props}
            self._upsert_node(node_type, node_id, data)

        # Tier 2: Update graph_compute cache after backend succeeds
        self.graph_compute.add_node(node_id, props)

    def add_edge(
        self,
        source: str,
        target: str,
        rel_type: str = "",
        ephemeral: bool = False,
        **properties: Any,
    ) -> None:
        """Add a generic edge between two nodes (backend-first, then graph_compute).

        Convenience for ad-hoc relationships (e.g. provenance links like
        ``RunTrace -[:HAS_CONTEXT]-> ContextBlob``, CONCEPT:ORCH-1.39) where there is no typed
        model. Best-effort: a missing backend/compute method is tolerated.
        """
        if self.backend and not ephemeral:
            _be = getattr(self.backend, "add_edge", None)
            if callable(_be):
                with contextlib.suppress(Exception):
                    _be(source, target, rel_type, **properties)
        _ge = getattr(self.graph_compute, "add_edge", None)
        if callable(_ge):
            with contextlib.suppress(Exception):
                _ge(source, target, {"rel_type": rel_type, **properties})

    def get_blast_radius(self, node_id: str, depth: int) -> list[dict[str, Any]]:
        """Retrieve the blast radius (dependencies) from a starting node.

        Uses the high-performance GraphComputeEngine with compiled Rust and
        epistemic-graph backend for fast traversals.
        """
        if self.backend:
            # Cypher-powered query: handles millions of nodes directly in the database
            query = f"""
            MATCH (s {{id: $node_id}})-[*1..{depth}]->(t)
            WITH s, t, shortestPath((s)-[*1..{depth}]->(t)) as p
            RETURN distinct t.id as id, labels(t)[0] as type, length(p) as depth
            """
            try:
                results = self.backend.execute(query, {"node_id": node_id})
                return [
                    {
                        "id": r["id"],
                        "type": r.get("type", "Node"),
                        "depth": r["depth"],
                    }
                    for r in results
                ]
            except Exception as e:
                logger.warning(
                    f"Cypher blast radius query failed: {e}. Falling back to compute engine."
                )

        return self.graph_compute.get_blast_radius(node_id, depth)

    # --- Tier 2: Compute Scratchpad (Rust-native on-demand loading) ---

    def load_subgraph(
        self, query: str, params: dict[str, Any] | None = None
    ) -> GraphComputeEngine:
        """Dynamically load a specialized subgraph from the persistent backend.

        This is the formal gateway from Tier 1 (persistent storage) to Tier 2
        (compute scratchpad). It prevents OOM bottlenecks by loading ONLY the
        relevant nodes/edges needed for a specific graph algorithm.

        The Cypher query must RETURN nodes 'n' and relationships 'r'.

        When no backend exists (memory-only mode), returns the full local graph.
        """
        if not self.backend:
            raise RuntimeError("Backend required for load_subgraph")

        subgraph = GraphComputeEngine(backend_type="rust")
        results = self.backend.execute(query, params or {})
        for row in results:
            n = row.get("n")
            if n and isinstance(n, dict) and "id" in n:
                props = {k: v for k, v in n.items() if k != "id"}
                subgraph.add_node(n["id"], props)

            # Simple relationship extraction
            r = row.get("r")
            if r and isinstance(r, dict) and "source" in r and "target" in r:
                subgraph.add_edge(
                    r["source"], r["target"], {"type": r.get("type", "UNKNOWN")}
                )

        return subgraph

    def checkout_subgraph(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        durable: Any | None = None,
    ) -> Any:
        """Check out a bounded subgraph as a write-back-capable working copy.

        Like :meth:`load_subgraph`, but returns a
        :class:`~agent_utilities.knowledge_graph.core.subgraph_checkout.CheckedOutSubgraph`
        that tracks mutations and can flush **only the deltas** back to the durable
        tier — instead of the detached, load-only scratchpad ``load_subgraph``
        returns. This closes the checkout → mutate → write-back loop without a
        full-graph enumeration. (CONCEPT:KG-2.7 P2 — bounded checkout + delta
        write-back.)

        The Cypher query must ``RETURN`` nodes ``n`` (and optionally relationships
        ``r``). ``durable`` defaults to this engine's backend (the durable tier).
        """
        if not self.backend:
            raise RuntimeError("Backend required for checkout_subgraph")

        from .subgraph_checkout import CheckedOutSubgraph

        inner = GraphComputeEngine(backend_type="rust")
        baseline: dict[str, str] = {}
        results = self.backend.execute(query, params or {})
        for row in results:
            n = row.get("n")
            if n and isinstance(n, dict) and "id" in n:
                props = {k: v for k, v in n.items() if k != "id"}
                inner.add_node(n["id"], props)
                baseline[n["id"]] = CheckedOutSubgraph._version_of(props)
            r = row.get("r")
            if r and isinstance(r, dict) and "source" in r and "target" in r:
                inner.add_edge(
                    r["source"], r["target"], {"type": r.get("type", "UNKNOWN")}
                )

        return CheckedOutSubgraph(
            inner,
            durable=durable if durable is not None else self.backend,
            baseline=baseline,
        )

    def load_for_centrality(
        self, node_types: list[str] | None = None
    ) -> GraphComputeEngine:
        """Load a focused subgraph for centrality/PageRank computation.

        Args:
            node_types: Optional filter by node types. If None, loads all nodes.
        """
        if not self.backend or not node_types:
            return self.load_subgraph("MATCH (n)-[r]->(m) RETURN n, r, m")
        return self.load_subgraph(
            "MATCH (n)-[r]->(m) WHERE n.type IN $types OR m.type IN $types RETURN n, r, m",
            {"types": node_types},
        )

    def load_for_impact_analysis(self, target_id: str) -> GraphComputeEngine:
        """Load neighbors within 3 hops of target for impact analysis."""
        if not self.backend:
            raise RuntimeError("Backend required for load_for_impact_analysis")
        return self.load_subgraph(
            "MATCH path = (n)-[*1..3]-(t {id: $target}) "
            "UNWIND nodes(path) AS n UNWIND relationships(path) AS r "
            "RETURN DISTINCT n, r",
            {"target": target_id},
        )

    # --- Background Analysis Methods ---

    def execute_deep_analysis(self, query: str, max_depth: int = 2) -> dict[str, Any]:
        """Perform a native background deep analysis of a concept.

        Architecture (Hybrid L1 + Free-Text LLM):
            - **L1 (Structured)**: The ``discover_innovations`` engine provides
              structured domain signals, scores, biomimicry mappings, and
              innovation claims natively — no LLM needed.
            - **L2 (Free-Text Synthesis)**: The LLM generates a natural language
              synthesis summary. This plays to any model's strength (text gen)
              and eliminates JSON schema validation failures entirely.
            - **KG Writeback**: Domain recommendations from L1 are written as
              ``ANALOGOUS_TO`` edges. The LLM summary is stored as a semantic
              ``Memory`` node for future retrieval.
        """

        from pydantic_ai import Agent

        from agent_utilities.core.config import (
            DEFAULT_KG_MODEL_ID,
            DEFAULT_LLM_PROVIDER,
        )
        from agent_utilities.core.model_factory import create_model

        # ── L1: Structured Discovery (no LLM, instant) ──────────────
        l1_results = self.discover_innovations(query, top_k=10)
        enriched = l1_results.get("results", [])
        domain_recs = l1_results.get("domain_recommendations", [])
        if not enriched:
            return {"status": "skipped", "reason": "No initial concepts found"}

        # Build compact context for LLM from L1 signals
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

        # ── L2: Free-Text LLM Synthesis ─────────────────────────────
        llm_summary = ""
        try:
            import nest_asyncio

            nest_asyncio.apply()

            model = create_model(
                provider=DEFAULT_LLM_PROVIDER, model_id=DEFAULT_KG_MODEL_ID
            )
            agent = Agent(
                model,
                system_prompt=(
                    "You are an expert software architect analyzing research papers "
                    "and codebases. Produce clear, actionable synthesis reports in "
                    "structured markdown. Focus on practical implementation guidance."
                ),
            )

            result = agent.run_sync(prompt)
            llm_summary = str(result.output)
            logger.info(f"L2 synthesis complete: {len(llm_summary)} chars generated")
        except Exception as e:
            logger.warning(f"L2 LLM synthesis failed (non-fatal): {e}")
            llm_summary = (
                f"[LLM synthesis unavailable — L1 signals preserved]\n\n"
                f"Query: {query}\n"
                f"Matches: {len(enriched)}\n"
                f"Top domains: {', '.join(d['domain'] for d in domain_recs[:5])}"
            )

        # ── KG Writeback: Domain edges + Memory node ─────────────────
        source_id = (
            query if "-" in query else (enriched[0].get("id") if enriched else query)
        )

        new_concepts = []
        # Write ANALOGOUS_TO edges from L1 domain recommendations
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

    def delete_node(self, node_id: str, ephemeral: bool = False) -> None:
        """Remove a node and its associated relationships from the graph.

        Write ordering: backend-first, then graph_compute.
        """
        if self.backend and not ephemeral:
            try:
                self.backend.execute(
                    "MATCH (n {id: $id}) DETACH DELETE n",
                    {"id": node_id},
                )
            except Exception as e:
                logger.warning(f"Backend delete_node failed: {e}")

        # Tier 2: Update graph_compute cache after backend succeeds
        try:
            self.graph_compute.remove_node(node_id)
        except Exception as e:
            logger.debug(f"graph_compute remove_node failed or node not found: {e}")

    def remove_node(self, node_id: str, ephemeral: bool = False) -> None:
        """Remove a node — delegates to delete_node."""
        self.delete_node(node_id, ephemeral)

    def delete_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str | None = None,
        ephemeral: bool = False,
    ) -> None:
        """Remove a relationship between two nodes in the graph.

        Write ordering: backend-first, then graph_compute.
        """
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
                self.backend.execute(query, {"sid": source_id, "tid": target_id})
            except Exception as e:
                logger.warning(f"Backend delete_edge failed: {e}")

        # Tier 2: Update graph_compute cache after backend succeeds
        try:
            self.graph_compute.remove_edge(source_id, target_id)
        except Exception as e:
            logger.debug(f"graph_compute remove_edge failed or edge not found: {e}")

    # ── Startup Hydration & Sync ───────────────────────────────────────

    def hydrate_compute_engine(self, limit: int = 50000) -> int:
        """Rebuild the Rust compute engine state from the persistent backend.

        CONCEPT:KG-2.7 — On startup (or after a Rust service restart), the
        in-memory graph is empty. This method queries the backend for all
        nodes and edges, then replays them into the compute engine via
        ``batch_update()`` to restore full state parity.

        Args:
            limit: Maximum number of nodes to hydrate (safety cap).

        Returns:
            Total number of operations replayed.
        """
        if not self.backend:
            logger.warning("No backend available for hydration")
            return 0

        ops: list[dict[str, Any]] = []

        # 1. Hydrate nodes
        try:
            node_results = self.backend.execute(
                f"MATCH (n) RETURN n.id as id, labels(n)[0] as lbl LIMIT {limit}",
                {},
            )
            for row in node_results:
                node_id = row.get("id")
                if node_id:
                    ops.append(
                        {
                            "op": "add_node",
                            "node_id": node_id,
                            "properties_json": json.dumps(
                                {"type": row.get("lbl", "Node")}
                            ),
                        }
                    )
        except Exception as e:
            logger.warning(f"Node hydration query failed: {e}")

        # 2. Hydrate edges
        try:
            edge_results = self.backend.execute(
                f"MATCH (s)-[r]->(t) RETURN s.id as sid, t.id as tid, type(r) as rel LIMIT {limit}",
                {},
            )
            for row in edge_results:
                sid = row.get("sid")
                tid = row.get("tid")
                if sid and tid:
                    ops.append(
                        {
                            "op": "add_edge",
                            "source_id": sid,
                            "target_id": tid,
                            "properties_json": json.dumps(
                                {"type": row.get("rel", "RELATED_TO")}
                            ),
                        }
                    )
        except Exception as e:
            logger.warning(f"Edge hydration query failed: {e}")

        if not ops:
            logger.info("No data to hydrate")
            return 0

        # 3. Replay via batch_update for efficiency
        try:
            self.graph_compute.batch_update(ops)
            logger.info(f"Hydrated compute engine with {len(ops)} operations")
        except Exception as e:
            logger.error(f"Batch hydration failed: {e}")
            return 0

        return len(ops)

    def sync_embeddings(
        self,
        direction: str = "backend_to_rust",
        limit: int = 10000,
    ) -> int:
        """Synchronize vector embeddings between LadybugDB and Rust SemanticStore.

        CONCEPT:KG-2.7 — Bidirectional sync to ensure both the persistent
        VECTOR index (LadybugDB) and the in-memory SemanticStore (Rust) have
        the same embeddings.

        Args:
            direction: ``backend_to_rust`` or ``rust_to_backend``.
            limit: Maximum embeddings to sync per call.

        Returns:
            Number of embeddings synced.
        """
        if not self.backend:
            logger.warning("No backend available for embedding sync")
            return 0

        count = 0

        if direction == "backend_to_rust":
            # Pull embeddings from LadybugDB and push to Rust
            try:
                results = self.backend.execute(
                    f"MATCH (n) WHERE n.embedding IS NOT NULL "
                    f"RETURN n.id as id, n.embedding as emb LIMIT {limit}",
                    {},
                )
                for row in results:
                    node_id = row.get("id")
                    embedding = row.get("emb")
                    if node_id and embedding and isinstance(embedding, list):
                        try:
                            self.graph_compute.add_node(
                                node_id,
                                {"embedding": [float(x) for x in embedding]},
                            )
                            count += 1
                        except Exception as e:
                            logger.debug(f"Failed to push embedding for {node_id}: {e}")
            except Exception as e:
                logger.warning(f"Backend embedding query failed: {e}")

        elif direction == "rust_to_backend":
            # Pull from Rust ledger and push to LadybugDB
            # The Rust side stores embeddings in SemanticStore, accessible via
            # the ledger — any AddEmbedding operations are logged there.
            try:
                ledger = self.graph_compute.get_ledger()
                for entry_str in ledger:
                    try:
                        entry = json.loads(entry_str)
                        if entry.get("op") == "add_embedding":
                            node_id = entry.get("node_id")
                            embedding = entry.get("embedding")
                            if node_id and embedding:
                                self.backend.execute(
                                    "MATCH (n {id: $id}) SET n.embedding = $emb",
                                    {"id": node_id, "emb": embedding},
                                )
                                count += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
            except Exception as e:
                logger.warning(f"Rust→backend embedding sync failed: {e}")
        else:
            raise ValueError(
                f"Invalid direction '{direction}'. "
                "Use 'backend_to_rust' or 'rust_to_backend'."
            )

        logger.info(f"Synced {count} embeddings ({direction})")
        return count
