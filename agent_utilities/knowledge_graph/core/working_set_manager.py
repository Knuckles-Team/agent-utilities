#!/usr/bin/python
from __future__ import annotations

"""Working Set Manager — LRU Subgraph Cache.

CONCEPT:KG-2.21 — Working Set Eviction & Memory Management

Manages an LRU-evicting cache of subgraphs loaded into the Rust
``GraphComputeEngine`` (L1). Prevents memory explosion by enforcing
hard caps on the number of nodes and edges in the working set.

Architecture:
    The persistent backend (L3) holds the complete graph. The L1
    Rust engine holds only the "working set" — the subgraph that
    is currently relevant to active queries/tasks.

    When a query needs topological analysis on nodes not in L1,
    the WSM loads the relevant subgraph from L3, potentially
    evicting older, cold subgraphs to stay within limits.

Usage::

    from agent_utilities.knowledge_graph.core.working_set_manager import (
        WorkingSetManager,
    )

    wsm = WorkingSetManager(
        graph_engine=graph_compute,
        persistent_backend=backend,
        max_nodes=50_000,
        max_edges=200_000,
    )

    # Load a focus subgraph
    wsm.load_focus("Agent", depth=2)

    # Check if working set has data
    if wsm.has_relevant_data():
        graph = wsm.get_working_set()

    # Evict cold data if needed
    wsm.evict_if_needed()

Environment Variables:
    WORKING_SET_MAX_NODES: Maximum nodes in L1. Default: 50000.
    WORKING_SET_MAX_EDGES: Maximum edges in L1. Default: 200000.
    WORKING_SET_EVICTION_RATIO: Fraction to evict when limit hit. Default: 0.25.
    WORKING_SET_TTL_SECONDS: Time-to-live for cached subgraphs. Default: 3600.
"""

import logging
import time
from collections import OrderedDict
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)


class WorkingSetEntry:
    """A cached subgraph entry with access tracking."""

    __slots__ = (
        "key",
        "node_ids",
        "edge_count",
        "loaded_at",
        "last_accessed",
        "access_count",
    )

    def __init__(self, key: str, node_ids: set[str], edge_count: int) -> None:
        self.key = key
        self.node_ids = node_ids
        self.edge_count = edge_count
        self.loaded_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0

    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1

    @property
    def age_seconds(self) -> float:
        """Seconds since this entry was loaded."""
        return time.time() - self.loaded_at

    @property
    def idle_seconds(self) -> float:
        """Seconds since last access."""
        return time.time() - self.last_accessed


class WorkingSetManager:
    """LRU-evicting subgraph cache for the L1 Rust compute engine.

    Maintains an ordered dictionary of loaded subgraphs, evicting
    the least-recently-used entries when memory limits are exceeded.
    """

    def __init__(
        self,
        graph_engine: Any = None,
        persistent_backend: Any = None,
        *,
        max_nodes: int | None = None,
        max_edges: int | None = None,
        eviction_ratio: float | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        """Initialize the working set manager.

        Args:
            graph_engine: GraphComputeEngine (L1 Rust).
            persistent_backend: GraphBackend (L3 persistent).
            max_nodes: Maximum nodes in L1 before eviction.
            max_edges: Maximum edges in L1 before eviction.
            eviction_ratio: Fraction to evict when limit hit (0.0-1.0).
            ttl_seconds: Time-to-live for cached subgraphs.
        """
        self._graph = graph_engine
        self._backend = persistent_backend

        self._max_nodes = max_nodes or int(setting("WORKING_SET_MAX_NODES", "50000"))
        self._max_edges = max_edges or int(setting("WORKING_SET_MAX_EDGES", "200000"))
        self._eviction_ratio = eviction_ratio or float(
            setting("WORKING_SET_EVICTION_RATIO", "0.25")
        )
        self._ttl = ttl_seconds or int(setting("WORKING_SET_TTL_SECONDS", "3600"))

        # LRU cache of loaded subgraphs
        self._entries: OrderedDict[str, WorkingSetEntry] = OrderedDict()

        # Current working set size
        self._current_nodes = 0
        self._current_edges = 0

        # Stats
        self._loads = 0
        self._evictions = 0
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_relevant_data(self) -> bool:
        """Check if the working set has any loaded subgraphs."""
        return len(self._entries) > 0 and self._current_nodes > 0

    def get_working_set(self) -> Any:
        """Get the underlying graph engine with current working set data."""
        return self._graph

    def load_focus(
        self,
        focus_type: str,
        *,
        depth: int = 2,
        limit: int = 10_000,
        filters: dict[str, Any] | None = None,
    ) -> int:
        """Load a focus subgraph from L3 into L1.

        Args:
            focus_type: Node type to focus on (e.g. "Agent", "Skill").
            depth: Traversal depth from focus nodes.
            limit: Maximum nodes to load.
            filters: Optional property filters.

        Returns:
            Number of nodes loaded.
        """
        cache_key = f"focus:{focus_type}:d{depth}"

        # Cache hit
        if cache_key in self._entries:
            self._entries[cache_key].touch()
            self._entries.move_to_end(cache_key)
            self._hits += 1
            return len(self._entries[cache_key].node_ids)

        self._misses += 1

        if not self._backend:
            logger.warning("No persistent backend available for focus loading")
            return 0

        # Evict if needed before loading
        self.evict_if_needed()

        # Load from L3
        try:
            # Query focus nodes
            filter_clause = ""
            if filters:
                conditions = [f"n.{k} = '{v}'" for k, v in filters.items()]
                filter_clause = "WHERE " + " AND ".join(conditions)

            query = (
                f"MATCH (n {{type: '{focus_type}'}}) {filter_clause} "
                f"RETURN n LIMIT {limit}"
            )
            results = self._backend.execute(query)

            loaded_ids: set[str] = set()
            for result in results:
                node = result.get("n", result)
                node_id = node.get("id", "")
                if node_id and self._graph:
                    props = {k: v for k, v in node.items() if k != "id"}
                    self._graph.add_node(node_id, props)
                    loaded_ids.add(node_id)

            # Load edges between loaded nodes (up to depth)
            if loaded_ids and depth > 0:
                edge_count = self._load_edges_for_nodes(loaded_ids, depth)
            else:
                edge_count = 0

            # Register in LRU cache
            entry = WorkingSetEntry(
                key=cache_key,
                node_ids=loaded_ids,
                edge_count=edge_count,
            )
            self._entries[cache_key] = entry
            self._current_nodes += len(loaded_ids)
            self._current_edges += edge_count
            self._loads += 1

            logger.info(
                "WorkingSet loaded focus=%s: %d nodes, %d edges "
                "(total: %d/%d nodes, %d/%d edges)",
                focus_type,
                len(loaded_ids),
                edge_count,
                self._current_nodes,
                self._max_nodes,
                self._current_edges,
                self._max_edges,
            )

            return len(loaded_ids)

        except Exception as e:
            logger.error("Failed to load focus subgraph: %s", e)
            return 0

    def load_neighborhood(
        self,
        node_id: str,
        *,
        depth: int = 1,
    ) -> int:
        """Load the neighborhood of a specific node into L1.

        Args:
            node_id: Center node ID.
            depth: Traversal depth.

        Returns:
            Number of nodes loaded.
        """
        cache_key = f"neighborhood:{node_id}:d{depth}"

        if cache_key in self._entries:
            self._entries[cache_key].touch()
            self._entries.move_to_end(cache_key)
            self._hits += 1
            return len(self._entries[cache_key].node_ids)

        self._misses += 1

        if not self._backend:
            return 0

        self.evict_if_needed()

        try:
            # Load via blast radius query
            query = (
                f"MATCH (n {{id: '{node_id}'}})-[*1..{depth}]-(m) "
                f"RETURN DISTINCT m LIMIT 5000"
            )
            results = self._backend.execute(query)

            loaded_ids: set[str] = {node_id}
            for result in results:
                node = result.get("m", result)
                nid = node.get("id", "")
                if nid and self._graph:
                    props = {k: v for k, v in node.items() if k != "id"}
                    self._graph.add_node(nid, props)
                    loaded_ids.add(nid)

            edge_count = self._load_edges_for_nodes(loaded_ids, 1)

            entry = WorkingSetEntry(
                key=cache_key,
                node_ids=loaded_ids,
                edge_count=edge_count,
            )
            self._entries[cache_key] = entry
            self._current_nodes += len(loaded_ids)
            self._current_edges += edge_count
            self._loads += 1

            return len(loaded_ids)

        except Exception as e:
            logger.error("Failed to load neighborhood for %s: %s", node_id, e)
            return 0

    def evict_if_needed(self) -> int:
        """Evict cold subgraphs if limits are exceeded.

        Uses LRU policy: evicts the least-recently-accessed entries
        first. Also evicts entries that have exceeded TTL.

        Returns:
            Number of entries evicted.
        """
        evicted = 0

        # TTL-based eviction
        expired_keys = [
            key for key, entry in self._entries.items() if entry.age_seconds > self._ttl
        ]
        for key in expired_keys:
            self._evict_entry(key)
            evicted += 1

        # LRU eviction if still over limits
        while (
            self._current_nodes > self._max_nodes
            or self._current_edges > self._max_edges
        ) and self._entries:
            # Evict oldest (first item in OrderedDict)
            oldest_key = next(iter(self._entries))
            self._evict_entry(oldest_key)
            evicted += 1

        if evicted:
            self._evictions += evicted
            logger.info(
                "WorkingSet evicted %d entries (now: %d/%d nodes, %d/%d edges)",
                evicted,
                self._current_nodes,
                self._max_nodes,
                self._current_edges,
                self._max_edges,
            )

        return evicted

    def clear(self) -> None:
        """Clear the entire working set."""
        self._entries.clear()
        self._current_nodes = 0
        self._current_edges = 0
        logger.info("WorkingSet cleared")

    def get_stats(self) -> dict[str, Any]:
        """Return working set statistics."""
        return {
            "entries": len(self._entries),
            "current_nodes": self._current_nodes,
            "max_nodes": self._max_nodes,
            "current_edges": self._current_edges,
            "max_edges": self._max_edges,
            "utilization_nodes": round(self._current_nodes / self._max_nodes * 100, 1)
            if self._max_nodes
            else 0,
            "utilization_edges": round(self._current_edges / self._max_edges * 100, 1)
            if self._max_edges
            else 0,
            "loads": self._loads,
            "evictions": self._evictions,
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "hit_rate": round(self._hits / max(self._hits + self._misses, 1) * 100, 1),
            "ttl_seconds": self._ttl,
            "entry_details": [
                {
                    "key": e.key,
                    "nodes": len(e.node_ids),
                    "edges": e.edge_count,
                    "age_s": round(e.age_seconds, 1),
                    "idle_s": round(e.idle_seconds, 1),
                    "accesses": e.access_count,
                }
                for e in self._entries.values()
            ],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_entry(self, key: str) -> None:
        """Remove a single cache entry and update counters."""
        if key not in self._entries:
            return

        entry = self._entries.pop(key)
        self._current_nodes -= len(entry.node_ids)
        self._current_edges -= entry.edge_count

        # Ensure counters don't go negative
        self._current_nodes = max(0, self._current_nodes)
        self._current_edges = max(0, self._current_edges)

        # Note: We don't remove nodes from the Rust engine because
        # other entries may reference them. The engine's own memory
        # management handles this via the LRU in the compute layer.

    def _load_edges_for_nodes(self, node_ids: set[str], depth: int) -> int:
        """Load edges between a set of nodes from L3.

        Returns:
            Number of edges loaded.
        """
        if not self._backend or not node_ids or not self._graph:
            return 0

        edge_count = 0
        try:
            # Build an IN-list for the query
            id_list = ", ".join(f"'{nid}'" for nid in list(node_ids)[:1000])
            query = (
                f"MATCH (a)-[r]->(b) "
                f"WHERE a.id IN [{id_list}] AND b.id IN [{id_list}] "
                f"RETURN a.id AS source, b.id AS target, type(r) AS type "
                f"LIMIT 50000"
            )
            results = self._backend.execute(query)

            for result in results:
                source = result.get("source", "")
                target = result.get("target", "")
                edge_type = result.get("type", "related")
                if source and target:
                    self._graph.add_edge(source, target, {"type": edge_type})
                    edge_count += 1

        except Exception as e:
            logger.debug("Edge loading failed: %s", e)

        return edge_count
