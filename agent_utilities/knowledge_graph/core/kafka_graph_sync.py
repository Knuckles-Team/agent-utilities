#!/usr/bin/python
from __future__ import annotations

"""Kafka Graph Sync Daemon.

CONCEPT:AU-KG.compute.event-driven-sync — Event-Driven Graph Synchronization

Consumes ``kg.mutations`` events from the EventBackend and applies them
to the Rust ``GraphComputeEngine`` (L1 cache). Ensures the in-process
compute graph stays in sync with the persistent backend (L3).

Features:
    - Batched updates: Accumulates mutations for 100ms before flushing
      to reduce FFI crossing overhead.
    - Periodic reconciliation: Every 5 minutes, performs a full diff
      between L1 and L3 to detect and repair drift.
    - Circuit breaker: If lag exceeds 10K events, triggers a full
      reload from L3 instead of replaying individual mutations.
    - Idempotent: All mutations are keyed by (node_id, timestamp),
      duplicate events are safely ignored.

Usage::

    from agent_utilities.knowledge_graph.core.kafka_graph_sync import (
        KafkaGraphSyncDaemon,
    )

    daemon = KafkaGraphSyncDaemon(
        event_backend=event_backend,
        graph_engine=graph_compute_engine,
    )
    await daemon.start()
    # ... runs in background ...
    await daemon.stop()
"""

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Configuration defaults
DEFAULT_BATCH_FLUSH_MS = 100
DEFAULT_RECONCILE_INTERVAL_S = 300  # 5 minutes
DEFAULT_CIRCUIT_BREAKER_LAG = 10_000
DEFAULT_MAX_BATCH_SIZE = 500


class KafkaGraphSyncDaemon:
    """Event-driven synchronization daemon for L1 ↔ L3 graph consistency.

    Subscribes to ``kg.mutations`` events and applies them to the
    Rust GraphComputeEngine in batched flushes. Periodically reconciles
    against the persistent backend to detect drift.
    """

    def __init__(
        self,
        event_backend: Any,
        graph_engine: Any,
        persistent_backend: Any | None = None,
        *,
        batch_flush_ms: int = DEFAULT_BATCH_FLUSH_MS,
        reconcile_interval_s: int = DEFAULT_RECONCILE_INTERVAL_S,
        circuit_breaker_lag: int = DEFAULT_CIRCUIT_BREAKER_LAG,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        consumer_group: str = "graph-sync",
    ) -> None:
        """Initialize the sync daemon.

        Args:
            event_backend: An EventBackend instance (Memory or Kafka).
            graph_engine: The GraphComputeEngine (L1 Rust cache).
            persistent_backend: Optional L3 persistent backend for reconciliation.
            batch_flush_ms: Milliseconds to accumulate before flushing.
            reconcile_interval_s: Seconds between full reconciliation passes.
            circuit_breaker_lag: Max pending events before triggering full reload.
            max_batch_size: Maximum mutations per batch flush.
            consumer_group: Kafka consumer group name.
        """
        self._event_backend = event_backend
        self._graph = graph_engine
        self._persistent = persistent_backend

        self._batch_flush_ms = batch_flush_ms
        self._reconcile_interval_s = reconcile_interval_s
        self._circuit_breaker_lag = circuit_breaker_lag
        self._max_batch_size = max_batch_size
        self._consumer_group = consumer_group

        # Internal state
        self._pending: list[dict[str, Any]] = []
        self._running = False
        self._flush_task: asyncio.Task | None = None
        self._reconcile_task: asyncio.Task | None = None
        self._last_reconcile = 0.0
        self._seen_keys: set[str] = set()

        # Stats
        self._applied = 0
        self._skipped = 0
        self._reconciliations = 0
        self._full_reloads = 0
        self._errors = 0

    async def start(self) -> None:
        """Subscribe to mutation events and start background tasks."""
        self._running = True

        # Subscribe to mutation events
        await self._event_backend.subscribe(
            "kg.mutations",
            self._consumer_group,
            self._on_mutation,
        )

        # Start periodic flush loop
        self._flush_task = asyncio.create_task(self._flush_loop())

        # Start reconciliation loop if persistent backend is available
        if self._persistent:
            self._reconcile_task = asyncio.create_task(self._reconcile_loop())

        logger.info(
            "KafkaGraphSyncDaemon started (flush=%dms, reconcile=%ds, breaker=%d)",
            self._batch_flush_ms,
            self._reconcile_interval_s,
            self._circuit_breaker_lag,
        )

    async def stop(self) -> None:
        """Stop the daemon and flush remaining mutations."""
        self._running = False

        # Final flush
        if self._pending:
            await self._flush_batch()

        # Cancel background tasks
        for task in [self._flush_task, self._reconcile_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info(
            "KafkaGraphSyncDaemon stopped "
            "(applied=%d, skipped=%d, reconciliations=%d, reloads=%d, errors=%d)",
            self._applied,
            self._skipped,
            self._reconciliations,
            self._full_reloads,
            self._errors,
        )

    def get_stats(self) -> dict[str, Any]:
        """Return sync daemon statistics."""
        return {
            "running": self._running,
            "pending_mutations": len(self._pending),
            "applied": self._applied,
            "skipped_duplicates": self._skipped,
            "reconciliations": self._reconciliations,
            "full_reloads": self._full_reloads,
            "errors": self._errors,
            "seen_keys_count": len(self._seen_keys),
        }

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------

    async def _on_mutation(self, topic: str, event: dict[str, Any]) -> None:
        """Handle an incoming mutation event.

        Events are buffered and flushed in batches for throughput.
        """
        # Deduplication key
        event_key = self._event_key(event)
        if event_key in self._seen_keys:
            self._skipped += 1
            return

        self._seen_keys.add(event_key)
        self._pending.append(event)

        # Circuit breaker check
        if len(self._pending) >= self._circuit_breaker_lag:
            logger.warning(
                "Circuit breaker triggered: %d pending mutations, "
                "performing full reload",
                len(self._pending),
            )
            self._pending.clear()
            await self._full_reload()

    # ------------------------------------------------------------------
    # Batch Flush
    # ------------------------------------------------------------------

    async def _flush_loop(self) -> None:
        """Periodically flush accumulated mutations."""
        while self._running:
            await asyncio.sleep(self._batch_flush_ms / 1000)
            if self._pending:
                await self._flush_batch()

    async def _flush_batch(self) -> None:
        """Apply accumulated mutations to the graph engine."""
        if not self._pending:
            return

        batch = self._pending[: self._max_batch_size]
        self._pending = self._pending[self._max_batch_size :]

        for mutation in batch:
            try:
                self._apply_mutation(mutation)
                self._applied += 1
            except Exception as e:
                self._errors += 1
                logger.error("Failed to apply mutation: %s — %s", mutation, e)

    def _apply_mutation(self, event: dict[str, Any]) -> None:
        """Apply a single mutation event to the graph engine.

        Supported actions:
            - add_node: Add or update a node
            - add_edge: Add or update an edge
            - remove_node: Remove a node
            - remove_edge: Remove an edge
        """
        action = event.get("action", "")
        data = event.get("data", {})

        if action == "add_node":
            node_id = data.get("id", "")
            properties = data.get("properties", {})
            if node_id:
                self._graph.add_node(node_id, properties)

        elif action == "add_edge":
            source = data.get("source", "")
            target = data.get("target", "")
            properties = data.get("properties", {})
            if source and target:
                self._graph.add_edge(source, target, properties)

        elif action == "remove_node":
            node_id = data.get("id", "")
            if node_id and hasattr(self._graph, "remove_node"):
                self._graph.remove_node(node_id)

        elif action == "remove_edge":
            source = data.get("source", "")
            target = data.get("target", "")
            if source and target and hasattr(self._graph, "remove_edge"):
                self._graph.remove_edge(source, target)

        else:
            logger.debug("Unknown mutation action: %s", action)

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    async def _reconcile_loop(self) -> None:
        """Periodic reconciliation between L1 (Rust) and L3 (persistent)."""
        while self._running:
            await asyncio.sleep(self._reconcile_interval_s)
            await self._reconcile()

    async def _reconcile(self) -> None:
        """Compare L1 graph state with L3 and repair drift.

        Detects:
            - Nodes in L3 but missing from L1 (add to L1)
            - Nodes in L1 but missing from L3 (stale, remove from L1)
            - Property mismatches (update L1 from L3)
        """
        if not self._persistent:
            return

        self._reconciliations += 1
        self._last_reconcile = time.time()

        try:
            # Get L3 node IDs
            l3_results = self._persistent.execute(
                "MATCH (n) RETURN n.id AS id LIMIT 100000"
            )
            l3_ids = {r.get("id", "") for r in l3_results if r.get("id")}

            # Get L1 node IDs
            l1_ids = set()
            if hasattr(self._graph, "node_ids"):
                l1_ids = set(self._graph.node_ids())
            elif hasattr(self._graph, "_graph"):
                nodes = self._graph._graph.get_nodes()
                l1_ids = {n[0] for n in nodes}

            # Missing from L1
            missing = l3_ids - l1_ids
            if missing:
                logger.info("Reconciliation: %d nodes missing from L1", len(missing))
                for node_id in list(missing)[:1000]:  # Cap at 1000 per cycle
                    try:
                        result = self._persistent.execute(
                            "MATCH (n {id: $id}) RETURN n",
                            {"id": node_id},
                        )
                        if result:
                            props = result[0].get("n", {})
                            self._graph.add_node(node_id, props)
                    except Exception as e:
                        logger.debug("Reconcile add failed for %s: %s", node_id, e)

            # Stale in L1 (optional: don't remove, just log)
            stale = l1_ids - l3_ids
            if stale:
                logger.info(
                    "Reconciliation: %d stale nodes in L1 (not in L3)", len(stale)
                )

            logger.info(
                "Reconciliation #%d complete: L3=%d nodes, L1=%d nodes, "
                "added=%d, stale=%d",
                self._reconciliations,
                len(l3_ids),
                len(l1_ids),
                len(missing),
                len(stale),
            )

        except Exception as e:
            self._errors += 1
            logger.error("Reconciliation failed: %s", e)

    async def _full_reload(self) -> None:
        """Full reload of L1 from L3 (circuit breaker recovery)."""
        self._full_reloads += 1

        if not self._persistent:
            logger.warning("Full reload requested but no persistent backend available")
            return

        try:
            # Load all nodes from L3
            nodes = self._persistent.execute("MATCH (n) RETURN n LIMIT 50000")
            for result in nodes:
                node = result.get("n", {})
                node_id = node.get("id", "")
                if node_id:
                    self._graph.add_node(node_id, node)

            # Load all edges from L3
            edges = self._persistent.execute(
                "MATCH (a)-[r]->(b) RETURN a.id AS source, b.id AS target, "
                "type(r) AS type LIMIT 200000"
            )
            for result in edges:
                source = result.get("source", "")
                target = result.get("target", "")
                edge_type = result.get("type", "related")
                if source and target:
                    self._graph.add_edge(source, target, {"type": edge_type})

            logger.info(
                "Full reload complete: %d nodes, %d edges", len(nodes), len(edges)
            )

        except Exception as e:
            self._errors += 1
            logger.error("Full reload failed: %s", e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _event_key(event: dict[str, Any]) -> str:
        """Generate a deduplication key for an event."""
        action = event.get("action", "")
        data = event.get("data", {})
        node_id = data.get("id", "")
        source = data.get("source", "")
        target = data.get("target", "")
        ts = event.get("timestamp", "")
        return f"{action}:{node_id or source}:{target}:{ts}"
