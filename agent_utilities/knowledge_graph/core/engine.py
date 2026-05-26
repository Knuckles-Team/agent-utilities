#!/usr/bin/python
from __future__ import annotations

"""Unified Intelligence Graph Engine — Tiered Architecture.

This module provides the high-level interface for querying the unified knowledge graph,
supporting structural Cypher queries, topological impact analysis, and hybrid search.

Architecture (Two-Tier Graph Engine):
    - **Tier 1 (Source of Truth)**: A persistent Cypher-capable backend
      (LadybugDB/Neo4j/PostgreSQL) handles all CRUD, schema enforcement,
      vector indexing, and filtered queries.
    - **Tier 2 (Compute Scratchpad)**: NetworkX is loaded on-demand via
      ``load_subgraph()`` for graph algorithms (PageRank, VF2, spectral
      clustering, causal reasoning) that databases cannot perform natively.

    When no persistent backend is available (``GRAPH_BACKEND=memory``),
    the engine falls back to using ``self.graph`` (NetworkX) as both
    storage and compute — suitable for testing and small graphs only.

The engine is composed of focused mixins for maintainability:
- ``engine_query.py``: Query, search, and retrieval methods.
- ``engine_memory.py``: Memory CRUD operations.
- ``engine_ingestion.py``: Episode, MCP, A2A, and skill ingestion.
- ``engine_ahe.py``: AHE self-improvement cycle methods.
- ``engine_registry.py``: Identity, prompt, resource, and codemap management.
"""

import json
import logging
import math
from enum import Enum
from typing import Any

import networkx as nx

from ..backends import create_backend, get_active_backend
from ..backends.base import GraphBackend

# Import mixins
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
from .engine_registry import FocusedSubgraph, RegistryMixin
from .engine_tasks import TaskManagerMixin

logger = logging.getLogger(__name__)

__all__ = [
    "IntelligenceGraphEngine",
    "RegistryGraphEngine",
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


class IntelligenceGraphEngine(
    QueryMixin,
    MemoryMixin,
    IngestionMixin,
    MCPDiscoveryMixin,
    AHEMixin,
    RegistryMixin,
    FederationMixin,
    EnterpriseEngineMixin,
    FinanceEngineMixin,
    MachineLearningEngineMixin,
    InfrastructureEngineMixin,
    TaskManagerMixin,
):
    """Engine for querying the unified intelligence graph (Agents, Tools, Code, Memory).

    Composed of focused mixins for maintainability. All 49+ existing importers
    continue to work since IntelligenceGraphEngine is still the single public class.

    Tiered Architecture:
        - Writes go to the persistent backend (Tier 1) when available.
        - ``self.graph`` (NetworkX) is the fallback when no backend exists,
          AND the compute scratchpad for graph algorithms via ``load_subgraph()``.
        - Dual-writes are avoided to prevent OOM at enterprise scale (100K+ nodes).
    """

    _ACTIVE_ENGINE: IntelligenceGraphEngine | None = None

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        backend: GraphBackend | None = None,
        db_path: str | None = None,
        external_ontologies: list[str] | None = None,
    ):
        self.graph = graph
        self.backend = None

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
                self.backend = None

        # Initialize compiled / optimized Graph Compute Engine (Rust/rustworkx/NetworkX)
        from agent_utilities.core.config import config

        from .graph_compute import GraphComputeEngine

        compute_backend = getattr(config, "graph_compute_backend", "epistemic_graph")
        self.graph_compute = GraphComputeEngine(backend_type=compute_backend)

        super().__init__()

        # Start workers if there are pending tasks in the database natively
        if self.backend:
            try:
                # Lightweight check to avoid locking if queue is empty
                pending = self.query_cypher(
                    "MATCH (t:Task {status: 'pending'}) RETURN count(t) as c"
                )
                if pending and pending[0]["c"] > 0:
                    import os

                    # Avoid auto-starting during unit tests to prevent thread deadlocks
                    if os.getenv("AGENT_UTILITIES_TESTING", "false").lower() != "true":
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

        self.hybrid_retriever = HybridRetriever(self)
        self.inference_engine = InferenceEngine(self)

        # Register external ontologies provided during initialization
        if external_ontologies:
            for ext in external_ontologies:
                # Basic parsing if formatted as URI|Endpoint
                parts = ext.split("|", 1)
                uri = parts[0].strip()
                endpoint = parts[1].strip() if len(parts) > 1 else None
                self.register_external_ontology(uri, endpoint)

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
            from ...graph.service_registry import ServiceRegistry

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
        return self.backend is None

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

    def _get_set_clause(
        self, data: dict[str, Any], alias: str = "n", label: str | None = None
    ) -> str:
        """Generate a SET clause for a Cypher query from a dictionary.

        Filters properties against the schema if the backend is Ladybug.
        """
        if label:
            label = self._normalize_label(label)
        valid_keys = None
        is_ladybug = (
            self.backend and self.backend.__class__.__name__ == "LadybugBackend"
        )

        # LadybugDB rel tables have no properties in our current schema definition.
        if is_ladybug and alias == "r":
            return ""

        if is_ladybug and label:
            from agent_utilities.models.schema_definition import SCHEMA

            for node in SCHEMA.nodes:
                if node.name == label:
                    valid_keys = set(node.columns.keys())
                    break

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
            # 2. If not found, create
            valid_keys = None
            is_ladybug = self.backend.__class__.__name__ == "LadybugBackend"
            if is_ladybug and label:
                from agent_utilities.models.schema_definition import SCHEMA

                for node in SCHEMA.nodes:
                    if node.name == label:
                        valid_keys = set(node.columns.keys())
                        break

            create_data = {}
            for k, v in data.items():
                if k == "id":
                    create_data[k] = v
                elif valid_keys is not None and k not in valid_keys:
                    continue
                else:
                    create_data[k] = v

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

        Tiered write path:
        - If backend exists and not ephemeral: writes to backend ONLY (source of truth).
        - If ephemeral or no backend: writes to NX (compute scratchpad / memory-only mode).
        """
        if rel_type:
            rel_type = rel_type.upper()
        props = properties or {}
        # Inject lightweight provenance/confidence tags for structural memory
        if "confidence" not in props:
            props["confidence"] = 1.0
        if "source" not in props:
            props["source"] = "system"
        # Graphiti-inspired temporal metadata: valid_from enables temporal queries
        if "valid_from" not in props:
            from datetime import UTC, datetime

            props["valid_from"] = datetime.now(UTC).isoformat()

        # Always update graph_compute cache in real-time
        self.graph_compute.add_edge(source_id, target_id, {"type": rel_type, **props})

        if self.backend and not ephemeral:
            # Tier 1: Backend is source of truth
            set_clause = self._get_set_clause(props, alias="r")

            is_postgres = self.backend.__class__.__name__ == "PostgreSQLBackend"
            s_query = (
                "MATCH (n) WHERE n.id = $id RETURN label(n) as lbl"
                if is_postgres
                else "MATCH (n) WHERE n.id = $id RETURN labels(n)[0] as lbl"
            )
            t_query = (
                "MATCH (n) WHERE n.id = $id RETURN label(n) as lbl"
                if is_postgres
                else "MATCH (n) WHERE n.id = $id RETURN labels(n)[0] as lbl"
            )

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
        elif source_id in self.graph and target_id in self.graph:
            # Tier 2 fallback: NX only (memory-only mode or ephemeral)
            self.graph.add_edge(source_id, target_id, type=rel_type, **props)

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
        source_id = None
        target_id = None

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

        # Fallback to O(N) NetworkX memory scan (memory-only mode)
        for node_id, data in self.graph.nodes(data=True):
            name = str(data.get("name", "")).lower()
            if not name:
                continue
            if source_name.lower() in name or name in source_name.lower():
                source_id = node_id
            if target_name.lower() in name or name in target_name.lower():
                target_id = node_id

            if source_id and target_id:
                break

        if source_id and target_id:
            self.link_nodes(
                source_id, target_id, rel_type, properties, ephemeral=ephemeral
            )
            return True
        return False

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

        Tiered write path:
        - If backend exists and not ephemeral: writes to backend ONLY (source of truth).
        - If ephemeral or no backend: writes to NX (compute scratchpad / memory-only mode).
        """
        node_type = self._normalize_label(node_type)
        props = properties or {}
        props["type"] = node_type

        # Always update graph_compute cache in real-time
        self.graph_compute.add_node(node_id, props)

        if self.backend and not ephemeral:
            # Tier 1: Backend is source of truth
            data = {"id": node_id, **props}
            self._upsert_node(node_type, node_id, data)
        else:
            # Tier 2 fallback: NX only (memory-only mode or ephemeral)
            self.graph.add_node(node_id, **props)

    def get_blast_radius(self, node_id: str, depth: int) -> list[dict[str, Any]]:
        """Retrieve the blast radius (dependencies) from a starting node.

        Uses the high-performance GraphComputeEngine with compiled Rust and
        rustworkx backends for fast traversals.
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

        # Fallback/Memory-only: ensure compute engine is hydrated from local graph
        if (
            not self.graph_compute.has_node(node_id)
            and self.graph.number_of_nodes() > 0
        ):
            for nid, data in self.graph.nodes(data=True):
                self.graph_compute.add_node(nid, data)
            for u, v, data in self.graph.edges(data=True):
                self.graph_compute.add_edge(u, v, data)

        return self.graph_compute.get_blast_radius(node_id, depth)

    # --- Tier 2: Compute Scratchpad (NetworkX on-demand loading) ---

    def load_subgraph(
        self, query: str, params: dict[str, Any] | None = None
    ) -> nx.MultiDiGraph:
        """Dynamically load a specialized subgraph from the persistent backend into NetworkX.

        This is the formal gateway from Tier 1 (persistent storage) to Tier 2
        (compute scratchpad). It prevents OOM bottlenecks by loading ONLY the
        relevant nodes/edges needed for a specific graph algorithm.

        The Cypher query must RETURN nodes 'n' and relationships 'r'.

        When no backend exists (memory-only mode), returns the full local graph.
        """
        if not self.backend:
            return self.graph  # Memory-only mode: NX IS the store

        subgraph = nx.MultiDiGraph()
        results = self.backend.execute(query, params or {})
        for row in results:
            n = row.get("n")
            if n and isinstance(n, dict) and "id" in n:
                subgraph.add_node(n["id"], **n)

            # Simple relationship extraction
            r = row.get("r")
            if r and isinstance(r, dict) and "source" in r and "target" in r:
                subgraph.add_edge(
                    r["source"], r["target"], type=r.get("type", "UNKNOWN")
                )

        return subgraph

    def load_for_centrality(
        self, node_types: list[str] | None = None
    ) -> nx.MultiDiGraph:
        """Load a focused subgraph for centrality/PageRank computation.

        Args:
            node_types: Optional filter by node types. If None, loads all nodes.
        """
        if not self.backend or not node_types:
            return (
                self.graph
                if not self.backend
                else self.load_subgraph("MATCH (n)-[r]->(m) RETURN n, r, m")
            )
        return self.load_subgraph(
            "MATCH (n)-[r]->(m) WHERE n.type IN $types OR m.type IN $types RETURN n, r, m",
            {"types": node_types},
        )

    def load_for_impact_analysis(self, target_id: str) -> nx.MultiDiGraph:
        """Load neighbors within 3 hops of target for impact analysis."""
        if not self.backend:
            return self.graph
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
            llm_summary = str(result.data)
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

    def delete_node(self, node_id: str, ephemeral: bool = False) -> None:
        """Remove a node and its associated relationships from the graph.

        Tiered write path:
        - If backend exists and not ephemeral: writes to backend ONLY (source of truth).
        - If ephemeral or no backend: writes to NX (compute scratchpad / memory-only mode).
        """
        # Always update graph_compute cache in real-time
        try:
            self.graph_compute.remove_node(node_id)
        except Exception as e:
            logger.debug(f"graph_compute remove_node failed or node not found: {e}")

        if self.backend and not ephemeral:
            try:
                self.backend.execute(
                    "MATCH (n {id: $id}) DETACH DELETE n",
                    {"id": node_id},
                )
            except Exception as e:
                logger.warning(f"Backend delete_node failed: {e}")
        elif node_id in self.graph:
            self.graph.remove_node(node_id)

    def remove_node(self, node_id: str, ephemeral: bool = False) -> None:
        """Alias for delete_node to support backward compatibility."""
        self.delete_node(node_id, ephemeral)

    def delete_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str | None = None,
        ephemeral: bool = False,
    ) -> None:
        """Remove a relationship between two nodes in the graph.

        Tiered write path:
        - If backend exists and not ephemeral: writes to backend ONLY (source of truth).
        - If ephemeral or no backend: writes to NX (compute scratchpad / memory-only mode).
        """
        # Always update graph_compute cache in real-time
        try:
            self.graph_compute.remove_edge(source_id, target_id)
        except Exception as e:
            logger.debug(f"graph_compute remove_edge failed or edge not found: {e}")

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
        elif self.graph.has_edge(source_id, target_id):
            # Find and remove matching edges in MultiDiGraph
            edges_to_remove = []
            for u, v, key, data in self.graph.edges(keys=True, data=True):
                if u == source_id and v == target_id:
                    if not rel_type or data.get("type", "").upper() == rel_type.upper():
                        edges_to_remove.append((u, v, key))
            for u, v, key in edges_to_remove:
                self.graph.remove_edge(u, v, key)


# Alias for backward compatibility
RegistryGraphEngine = IntelligenceGraphEngine
