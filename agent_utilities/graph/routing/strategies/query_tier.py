#!/usr/bin/python
from __future__ import annotations

"""Cost-Based Query Router.

CONCEPT:KG-2.7 — Intelligent Query Routing

Routes graph queries to the optimal execution tier based on query
classification and cost estimation:

    - **L1 (Rust GraphComputeEngine)**: Topological operations — PageRank,
      shortest path, blast radius, centrality, community detection.
      Fastest tier: sub-ms to <15ms.

    - **L2 (Working Set Cache)**: Subgraph operations on recently-loaded
      working sets. Avoids redundant L3 round-trips for hot subgraphs.

    - **L3 (Persistent Backend)**: Filtered MATCH/MERGE queries, schema
      operations, full-text search, SPARQL queries. Higher latency but
      complete dataset access.

    - **L4 (Vector Index)**: Semantic similarity search via embeddings.
      Used for natural-language retrieval queries.

Usage::

    router = QueryRouter(
        graph_engine=graph_compute,
        persistent_backend=ladybug_backend,
    )

    # Automatic routing
    results = router.route("MATCH (n:Agent) RETURN n LIMIT 10")
    results = router.route_topological("pagerank", damping=0.85)
    results = router.route_semantic(embedding_vector, n=5)

Environment Variables:
    QUERY_ROUTER_STRATEGY: Routing strategy ("auto", "l1-only", "l3-only").
        Default: "auto".
    QUERY_ROUTER_L1_THRESHOLD: Max node count for L1-eligible queries.
        Default: 50000.
"""

import logging
import re
import time
from enum import Enum
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)


class QueryTier(Enum):
    """Execution tier for query routing."""

    L1_RUST = "l1_rust"
    L2_CACHE = "l2_cache"
    L3_PERSISTENT = "l3_persistent"
    L4_VECTOR = "l4_vector"


class QueryType(Enum):
    """Classification of query intent."""

    TOPOLOGICAL = "topological"
    FILTERED_MATCH = "filtered_match"
    SEMANTIC = "semantic"
    SPARQL = "sparql"
    MUTATION = "mutation"
    SCHEMA = "schema"
    AGGREGATION = "aggregation"


# Patterns for classifying queries
_TOPOLOGICAL_KEYWORDS = {
    "pagerank",
    "centrality",
    "shortest_path",
    "blast_radius",
    "topological_sort",
    "connected_components",
    "community_detection",
    "degree",
    "neighbors",
    "predecessors",
    "successors",
    "subgraph",
    "bfs",
    "dfs",
    "cycle",
}

_MUTATION_KEYWORDS = {"CREATE", "MERGE", "DELETE", "SET", "REMOVE", "INSERT"}
_SCHEMA_KEYWORDS = {"INDEX", "CONSTRAINT", "SCHEMA", "DDL"}


class QueryRouter:
    """Cost-based query router for tiered graph architecture.

    Classifies incoming queries and routes them to the optimal
    execution tier. Tracks per-tier latency for adaptive routing.
    """

    def __init__(
        self,
        graph_engine: Any = None,
        persistent_backend: Any = None,
        working_set_manager: Any = None,
        *,
        strategy: str | None = None,
        l1_threshold: int | None = None,
    ) -> None:
        """Initialize the query router.

        Args:
            graph_engine: GraphComputeEngine (L1 Rust).
            persistent_backend: GraphBackend (L3 persistent).
            working_set_manager: WorkingSetManager (L2 cache).
            strategy: Routing strategy ("auto", "l1-only", "l3-only").
            l1_threshold: Max nodes for L1-eligible queries.
        """
        self._graph = graph_engine
        self._backend = persistent_backend
        self._wsm = working_set_manager

        self._strategy = (
            strategy or setting("QUERY_ROUTER_STRATEGY") or "auto"
        ).lower()
        self._l1_threshold = l1_threshold or int(
            setting("QUERY_ROUTER_L1_THRESHOLD", "50000")
        )

        # Latency tracking (exponential moving average)
        self._tier_latency: dict[QueryTier, float] = {
            QueryTier.L1_RUST: 1.0,  # ~1ms expected
            QueryTier.L2_CACHE: 5.0,  # ~5ms expected
            QueryTier.L3_PERSISTENT: 50.0,  # ~50ms expected
            QueryTier.L4_VECTOR: 20.0,  # ~20ms expected
        }
        self._query_count = 0
        self._tier_counts: dict[QueryTier, int] = {t: 0 for t in QueryTier}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        expected_hops: int = 1,
        requires_freshness: bool = False,
    ) -> list[dict[str, Any]]:
        """Automatically route a query to the optimal tier.

        Args:
            query: Cypher, SPARQL, or topological query string.
            params: Optional query parameters.
            expected_hops: Estimated multi-hop depth for routing decisions.
            requires_freshness: If True, bypass cache and hit L3 SPARQL.

        Returns:
            Query results from the selected tier.
        """
        query_type = self._classify(query)
        tier = self._select_tier(query_type, expected_hops, requires_freshness)

        return self._execute(tier, query_type, query, params)

    def route_topological(
        self,
        operation: str,
        **kwargs: Any,
    ) -> Any:
        """Route a topological operation directly to L1 (Rust).

        Args:
            operation: Algorithm name (pagerank, shortest_path, etc.).
            **kwargs: Algorithm-specific parameters.

        Returns:
            Algorithm result.
        """
        if not self._graph:
            raise RuntimeError("No GraphComputeEngine available for L1 operations")

        start = time.monotonic()
        try:
            method = getattr(self._graph, operation, None)
            if method is None:
                raise AttributeError(
                    f"GraphComputeEngine has no operation: {operation}"
                )
            result = method(**kwargs)
            self._record_latency(QueryTier.L1_RUST, time.monotonic() - start)
            return result
        except Exception as e:
            logger.error("L1 topological operation '%s' failed: %s", operation, e)
            raise

    def route_semantic(
        self,
        embedding: list[float],
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Route a semantic search to L4 (vector index).

        Args:
            embedding: Query embedding vector.
            n_results: Number of results to return.

        Returns:
            List of matching nodes with similarity scores.
        """
        if not self._backend:
            raise RuntimeError("No persistent backend available for semantic search")

        start = time.monotonic()
        try:
            results = self._backend.semantic_search(embedding, n_results)
            self._record_latency(QueryTier.L4_VECTOR, time.monotonic() - start)
            return results
        except Exception as e:
            logger.error("L4 semantic search failed: %s", e)
            return []

    def route_sparql(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Route a SPARQL query to a SPARQL-capable backend.

        Checks L3 backend for SPARQL support first. Falls back to
        L1 GraphComputeEngine if available.

        Args:
            query: SPARQL 1.1 query string.

        Returns:
            SPARQL result bindings.
        """
        if self._backend and hasattr(self._backend, "supports_sparql"):
            if self._backend.supports_sparql:
                start = time.monotonic()
                results = self._backend.execute_sparql(query, **kwargs)
                self._record_latency(QueryTier.L3_PERSISTENT, time.monotonic() - start)
                return results

        raise RuntimeError(
            "No SPARQL-capable backend available. Use create_backend('jena_fuseki')."
        )

    def get_stats(self) -> dict[str, Any]:
        """Return routing statistics."""
        return {
            "strategy": self._strategy,
            "total_queries": self._query_count,
            "tier_counts": {t.value: c for t, c in self._tier_counts.items()},
            "tier_latency_ms": {
                t.value: round(v, 2) for t, v in self._tier_latency.items()
            },
            "l1_threshold": self._l1_threshold,
        }

    # ------------------------------------------------------------------
    # Query Classification
    # ------------------------------------------------------------------

    def _classify(self, query: str) -> QueryType:
        """Classify a query by intent.

        Uses keyword analysis and pattern matching to determine
        the optimal execution tier.
        """
        q_upper = query.strip().upper()
        q_lower = query.strip().lower()

        # SPARQL detection
        if q_upper.startswith(("SELECT", "ASK", "CONSTRUCT", "PREFIX")) and (
            "WHERE" in q_upper and "{" in query
        ):
            return QueryType.SPARQL

        # Mutation detection
        if any(q_upper.startswith(kw) for kw in _MUTATION_KEYWORDS):
            return QueryType.MUTATION

        # Schema operations
        if any(kw in q_upper for kw in _SCHEMA_KEYWORDS):
            return QueryType.SCHEMA

        # Topological operations (method-style calls)
        if any(kw in q_lower for kw in _TOPOLOGICAL_KEYWORDS):
            return QueryType.TOPOLOGICAL

        # Aggregation queries
        if re.search(r"\b(COUNT|SUM|AVG|MIN|MAX|COLLECT)\s*\(", q_upper):
            return QueryType.AGGREGATION

        # Default: filtered match
        return QueryType.FILTERED_MATCH

    def _select_tier(
        self,
        query_type: QueryType,
        expected_hops: int = 1,
        requires_freshness: bool = False,
    ) -> QueryTier:
        """Select the optimal execution tier for a query type.

        Respects the configured strategy override and cost-heuristics
        (freshness, hops).
        """
        if self._strategy == "l1-only":
            return QueryTier.L1_RUST
        if self._strategy == "l3-only":
            return QueryTier.L3_PERSISTENT

        # Cost-Based Heuristics Overrides
        if requires_freshness:
            # Force hit persistent store (L3) to bypass cache
            if query_type == QueryType.TOPOLOGICAL:
                return QueryTier.L1_RUST  # L1 might be sync'd via Kafka in real-time
            return QueryTier.L3_PERSISTENT

        if expected_hops >= 2 and query_type in (
            QueryType.FILTERED_MATCH,
            QueryType.SPARQL,
        ):
            # Complex multi-hop traversals should hit Working Set (L2) or Rust (L1)
            # if we have it, rather than doing huge JOINs in SPARQL.
            if self._wsm and self._wsm.has_relevant_data():
                return QueryTier.L2_CACHE
            elif self._graph:
                return QueryTier.L1_RUST

        # Auto routing
        tier_map = {
            QueryType.TOPOLOGICAL: QueryTier.L1_RUST,
            QueryType.SEMANTIC: QueryTier.L4_VECTOR,
            QueryType.SPARQL: QueryTier.L3_PERSISTENT,
            QueryType.MUTATION: QueryTier.L3_PERSISTENT,
            QueryType.SCHEMA: QueryTier.L3_PERSISTENT,
            QueryType.AGGREGATION: QueryTier.L3_PERSISTENT,
            QueryType.FILTERED_MATCH: QueryTier.L3_PERSISTENT,
        }

        tier = tier_map.get(query_type, QueryTier.L3_PERSISTENT)

        # Try L2 cache for filtered matches if working set is available
        if query_type == QueryType.FILTERED_MATCH and self._wsm:
            if self._wsm.has_relevant_data():
                tier = QueryTier.L2_CACHE

        return tier

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute(
        self,
        tier: QueryTier,
        query_type: QueryType,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a query on the selected tier."""
        self._query_count += 1
        self._tier_counts[tier] = self._tier_counts.get(tier, 0) + 1

        start = time.monotonic()
        try:
            if tier == QueryTier.L1_RUST:
                result = self._execute_l1(query, query_type, params)
            elif tier == QueryTier.L2_CACHE:
                result = self._execute_l2(query, params)
            elif tier == QueryTier.L3_PERSISTENT:
                result = self._execute_l3(query, params)
            elif tier == QueryTier.L4_VECTOR:
                result = self._execute_l4(query, params)
            else:
                result = [{"error": f"Unknown tier: {tier}"}]

            elapsed = (time.monotonic() - start) * 1000
            self._record_latency(tier, elapsed)

            logger.debug(
                "QueryRouter: type=%s tier=%s latency=%.1fms results=%d",
                query_type.value,
                tier.value,
                elapsed,
                len(result),
            )
            return result

        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            logger.error(
                "QueryRouter: failed on tier=%s query=%s: %s (%.1fms)",
                tier.value,
                query[:80],
                e,
                elapsed,
            )
            # Fallback: try L3 if L1/L2 failed
            if tier in (QueryTier.L1_RUST, QueryTier.L2_CACHE) and self._backend:
                logger.info("Falling back to L3 persistent backend")
                return self._execute_l3(query, params)
            return [{"error": str(e)}]

    def _execute_l1(
        self,
        query: str,
        query_type: QueryType,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute on L1 (Rust GraphComputeEngine)."""
        if not self._graph:
            raise RuntimeError("No L1 engine available")

        # For topological queries, try to extract the operation
        q_lower = query.strip().lower()
        for keyword in _TOPOLOGICAL_KEYWORDS:
            if keyword in q_lower:
                method = getattr(self._graph, keyword, None)
                if method:
                    return [{"result": method(**(params or {}))}]

        # Fall through to L3
        return self._execute_l3(query, params)

    def _execute_l2(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute on L2 (Working Set Cache)."""
        if not self._wsm:
            raise RuntimeError("No L2 working set available")

        # The working set manager holds a cached subgraph
        cached_graph = self._wsm.get_working_set()
        if cached_graph and hasattr(cached_graph, "execute"):
            return cached_graph.execute(query, params)

        # Fall through to L3
        return self._execute_l3(query, params)

    def _execute_l3(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute on L3 (Persistent Backend)."""
        if not self._backend:
            raise RuntimeError("No L3 persistent backend available")
        return self._backend.execute(query, params)

    def _execute_l4(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute on L4 (Vector Index)."""
        if not self._backend:
            raise RuntimeError("No L4 vector backend available")

        # Extract embedding from params
        embedding = (params or {}).get("embedding", [])
        n = (params or {}).get("n_results", 5)
        return self._backend.semantic_search(embedding, n)

    # ------------------------------------------------------------------
    # Latency Tracking
    # ------------------------------------------------------------------

    def _record_latency(self, tier: QueryTier, latency_ms: float) -> None:
        """Update exponential moving average latency for a tier."""
        alpha = 0.1  # Smoothing factor
        current = self._tier_latency.get(tier, latency_ms)
        self._tier_latency[tier] = alpha * latency_ms + (1 - alpha) * current
