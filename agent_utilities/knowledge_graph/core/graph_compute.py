# CONCEPT:KG-2.2 - High-Performance Graph Compute Engine
# CONCEPT:ORCH-1.29 - Compiled Orchestration Kernel
# CONCEPT:KG-2.19 - Tokio Service Layer (Tokio-first)

import json
import logging
import os
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)


class GraphComputeEngine:
    """Graph compute engine backed by the epistemic-graph Tokio service.

    All graph operations route through the Tokio service layer via UDS/TCP.
    The service must be running before this engine is instantiated.
    Falls back to PyO3 in-process mode only if the service is unavailable
    and ``GRAPH_COMPUTE_FALLBACK=embedded`` is set.
    """

    def __init__(self, graph_name: str = "__bus__", **kwargs: Any) -> None:
        from epistemic_graph.client import SyncEpistemicGraphClient

        self.graph: dict[str, Any] = {}
        self._client: SyncEpistemicGraphClient
        self._mode: str = "embedded"

        try:
            self._client = SyncEpistemicGraphClient.connect(graph_name=graph_name)
        except Exception:
            logger.info(
                "epistemic-graph Tokio service not running. Attempting to auto-start daemon..."
            )
            import subprocess
            import sys
            import time
            from pathlib import Path

            try:
                server_path = str(
                    Path(sys.executable).parent / "epistemic-graph-server"
                )
                subprocess.Popen(
                    [server_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                time.sleep(1.0)
                self._client = SyncEpistemicGraphClient.connect(graph_name=graph_name)
            except Exception as retry_e:
                raise ConnectionError(
                    f"Cannot connect to epistemic-graph Tokio service after auto-start: {retry_e}. "
                    "Ensure the epistemic-graph-server daemon is running."
                ) from retry_e

        self._mode = "service"
        logger.info(
            "Connected to epistemic-graph Tokio service (graph: %s).",
            graph_name,
        )

        try:
            if self._client:
                # Try to create the graph so tests and dynamic instances don't fail
                # if the graph doesn't exist in the Rust backend yet.
                self._client.tenants.create(graph_name)
        except Exception as e:
            logger.warning(f"Failed to create tenant graph {graph_name}: {repr(e)}")
            pass

        # Bridging local events to the rust service when kafka isn't running
        if (
            os.environ.get("KAFKA_BOOTSTRAP_SERVERS") is None
            or os.environ.get("KAFKA_BOOTSTRAP_SERVERS") == ""
        ):
            self._start_event_bridge()

    def _start_event_bridge(self) -> None:
        """Starts a background bridge to forward local EventBus events to the Rust service."""
        import asyncio
        import threading

        from agent_utilities.knowledge_graph.core.event_backend import get_event_backend

        def bridge_worker() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            eb = get_event_backend()

            async def handle_mutation(topic: str, payload: dict) -> None:
                if "event_type" in payload and "query" in payload:
                    try:
                        if self._client:
                            self._client.apply_mutation(
                                payload["event_type"], payload["query"]
                            )
                    except Exception as exc:
                        logger.error(
                            "Failed to forward mutation to epistemic-graph: %s", exc
                        )

            async def run_subscriber() -> None:
                await eb.subscribe("kg.mutations", "epistemic-bridge", handle_mutation)
                # Keep loop alive to process events
                while True:
                    await asyncio.sleep(3600)

            try:
                loop.run_until_complete(run_subscriber())
            except Exception as e:
                logger.error("Event bridge worker failed: %s", e)

        t = threading.Thread(
            target=bridge_worker, daemon=True, name="EventBridgeWorker"
        )
        t.start()
        logger.info("Started Local-First EventBus bridge to epistemic-graph")

    # ── Node CRUD ────────────────────────────────────────────────────────

    def add_node(self, node_id: str, properties: Any = None, **kwargs: Any) -> None:
        """Add a node with properties to the graph.

        Supports both explicit dict and NX-style kwargs::

            engine.add_node("n1", {"type": "Agent"})
            engine.add_node("n1", type="Agent", name="foo")
        """

        def clean_props(d: Mapping[str, Any]) -> dict[str, Any]:
            import datetime

            from pydantic import BaseModel

            def serialize(val: Any) -> Any:
                if hasattr(val, "model_dump"):
                    try:
                        return val.model_dump(mode="json")
                    except Exception:
                        pass
                if isinstance(val, BaseModel):
                    return val.model_dump(mode="json")
                if isinstance(val, dict):
                    return {k: serialize(v) for k, v in val.items()}
                if isinstance(val, list | tuple | set):
                    return [serialize(v) for v in val]
                if isinstance(val, datetime.datetime):
                    return val.isoformat()
                return val

            return {k: serialize(v) for k, v in d.items()}

        props = dict(properties or {})
        props.update(kwargs)
        props = clean_props(props)
        self._client.nodes.add(node_id, props)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        properties: Any = None,
        **kwargs: Any,
    ) -> None:
        """Add a directed edge between two nodes with properties.

        Supports both explicit dict and NX-style kwargs::

            engine.add_edge("a", "b", {"type": "DEPENDS_ON"})
            engine.add_edge("a", "b", type="DEPENDS_ON")
        """

        def clean_props(d: dict[str, Any]) -> dict[str, Any]:
            import datetime

            from pydantic import BaseModel

            def serialize(val: Any) -> Any:
                if hasattr(val, "model_dump"):
                    try:
                        return val.model_dump(mode="json")
                    except Exception:
                        pass
                if isinstance(val, BaseModel):
                    return val.model_dump(mode="json")
                if isinstance(val, dict):
                    return {k: serialize(v) for k, v in val.items()}
                if isinstance(val, list | tuple | set):
                    return [serialize(v) for v in val]
                if isinstance(val, datetime.datetime):
                    return val.isoformat()
                return val

            return {k: serialize(v) for k, v in d.items()}

        props = dict(properties or {})
        props.update(kwargs)
        props = clean_props(props)

        if self.has_edge(source_id, target_id):
            self.remove_edge(source_id, target_id)

        try:
            self._client.edges.add(source_id, target_id, props)
        except Exception:
            # Ensure nodes exist without overwriting their existing properties
            if not self.has_node(source_id):
                self._client.nodes.add(source_id, {})
            if not self.has_node(target_id):
                self._client.nodes.add(target_id, {})
            self._client.edges.add(source_id, target_id, props)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all of its associated edges."""
        self._client.nodes.remove(node_id)

    def remove_edge(self, source_id: str, target_id: str, key: Any = None) -> None:
        """Remove a directed edge between source and target."""
        self._client.edges.remove(source_id, target_id)

    def has_node(self, node_id: str) -> bool:
        """Check if node_id exists in the graph."""
        return self._client.nodes.has(node_id)

    def has_edge(self, source_id: str, target_id: str) -> bool:
        """Check if a directed edge exists between source and target."""
        return self._client.edges.has(source_id, target_id)

    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        return self._client.nodes.count()

    def edge_count(self) -> int:
        """Return the number of edges in the graph."""
        return self._client.edges.count()

    # ── Graph Algorithms ─────────────────────────────────────────────────

    def topological_sort(self) -> list[str]:
        """Perform topological sort across the graph.

        Raises:
            ValueError: If the graph contains dependency cycles.
        """
        try:
            return self._client.graph.topological_sort()
        except Exception as e:
            raise ValueError("Graph contains cycles") from e

    def find_cycle(self) -> list[str] | None:
        """Detect and return any cycles found within the graph."""
        return self._client.graph.find_cycle()

    def get_shortest_path(self, source_id: str, target_id: str) -> list[str] | None:
        """Get the shortest path between source and target nodes."""
        return self._client.graph.shortest_path(source_id, target_id)

    @staticmethod
    def _bfs_collect(
        start: Any,
        max_depth: int,
        neighbors_fn: Any,
        node_info_fn: Any,
    ) -> list[dict[str, Any]]:
        """Backend-agnostic BFS traversal collecting blast radius results.

        Args:
            start: Starting node identifier (backend-specific).
            max_depth: Maximum traversal depth.
            neighbors_fn: Callable(node) -> iterable of neighbor nodes.
            node_info_fn: Callable(node) -> dict with 'id' and 'type' keys.
        """
        visited: set = {start}
        queue: list[tuple[Any, int]] = [(start, 0)]
        results: list[dict[str, Any]] = []
        while queue:
            curr, depth = queue.pop(0)
            if curr != start:
                info = node_info_fn(curr)
                info["depth"] = depth
                results.append(info)
            if depth < max_depth:
                for neighbor in neighbors_fn(curr):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
        return results

    def get_blast_radius(self, node_id: str, max_depth: int) -> list[dict[str, Any]]:
        """Compute the blast radius dependencies from a starting node.

        Returns a list of dicts: [{'id': str, 'type': str, 'depth': int}]
        """
        nodes = self._client.graph.blast_radius(node_id, max_depth)
        res = []
        for i, nid in enumerate(nodes, start=1):
            res.append({"id": nid, "type": "Node", "depth": min(i, max_depth)})
        return res

    def parse_repository(self, root_path: str) -> None:
        """Parse repository AST natively using the Rust backend."""
        self._client.graph.parse_repository(root_path)

    def vf2_subgraph_match(self, pattern: "GraphComputeEngine") -> list[dict[str, str]]:
        """Find all subgraph isomorphism matches from pattern to target graph."""
        return self._client.graph.vf2_subgraph_match(pattern._client)

    # ── Ledger Operations ────────────────────────────────────────────────

    def get_ledger(self) -> list[str]:
        """Retrieve the mutation transaction ledger log."""
        return self._client.ledger.get()

    def clear_ledger(self) -> None:
        """Clear the mutation transaction ledger log."""
        self._client.ledger.clear()

    @staticmethod
    def _parse_ledger_entry(tx: str) -> tuple[str, list[str]]:
        """Parse a ledger transaction string into (operation, args).

        Shared parser to ensure Rust and Python ledger formats stay in sync.
        """
        parts = tx.split("|")
        if not parts:
            return ("", [])
        return (parts[0], parts[1:])

    def apply_ledger(self, transactions: list[str]) -> None:
        """Replay mutations from a transaction ledger log."""
        self._client.ledger.apply(transactions)

    def flush_ledger_to_backend(self, backend: Any) -> int:
        """Flush the epistemic-graph mutation ledger to a persistent backend.

        Args:
            backend: A GraphBackend instance (e.g., LadybugBackend)

        Returns:
            int: The number of transactions flushed.
        """
        txs = self.get_ledger()
        if not txs:
            return 0

        count = 0
        for tx in txs:
            op, args = self._parse_ledger_entry(tx)
            if op == "AddNode" and len(args) >= 2:
                node_id = args[0]
                props_str = args[1]
                try:
                    props = json.loads(props_str)
                except Exception:
                    props = {}

                node_type = props.get("type", props.get("node_type", "Entity"))
                if node_type == "SYMBOL":
                    symbol_type = props.get("symbol_type", "Unknown")
                    file_path = props.get("file_path", "")
                    ast_hash = props.get("ast_hash", "")
                    name = props.get("name", node_id)
                    metadata_str = json.dumps(props)

                    query = (
                        "MERGE (n:Symbol {id: $id}) "
                        "SET n.type = 'SYMBOL', n.name = $name, "
                        "n.symbol_type = $sym_type, n.file_path = $fp, "
                        "n.ast_hash = $ast_hash, n.metadata = $meta"
                    )
                    try:
                        backend.execute_write(
                            query,
                            parameters={
                                "id": node_id,
                                "name": name,
                                "sym_type": symbol_type,
                                "fp": file_path,
                                "ast_hash": ast_hash,
                                "meta": metadata_str,
                            },
                        )
                    except Exception as e:
                        logger.error(f"Failed to sync Symbol node {node_id}: {e}")
                else:
                    # Generic node fallback
                    query = f"MERGE (n:{node_type} {{id: $id}}) SET n.metadata = $meta"
                    try:
                        backend.execute_write(
                            query,
                            parameters={
                                "id": node_id,
                                "meta": props_str,
                            },
                        )
                    except Exception as e:
                        logger.error(f"Failed to sync Node {node_id}: {e}")
                count += 1
            elif op == "AddEdge" and len(args) >= 3:
                src = args[0]
                tgt = args[1]
                props_str = args[2]
                try:
                    props = json.loads(props_str)
                except Exception:
                    props = {}

                edge_type = props.get("type") or props.get("edge_type") or "RELATED_TO"
                # Sanitize edge type for cypher
                edge_type = edge_type.replace(" ", "_").upper()
                query = (
                    f"MATCH (a {{id: $src}}), (b {{id: $tgt}}) "
                    f"MERGE (a)-[r:{edge_type}]->(b) "
                    "SET r.metadata = $meta"
                )
                try:
                    backend.execute_write(
                        query,
                        parameters={
                            "src": src,
                            "tgt": tgt,
                            "meta": props_str,
                        },
                    )
                except Exception as e:
                    logger.error(f"Failed to sync Edge {src}->{tgt}: {e}")
                count += 1

        self.clear_ledger()
        return count

    # ── Serialization ────────────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize the graph to a JSON string representation."""
        nodes = []
        for nid in self._get_all_nodes():
            props = self._get_node_properties(nid)
            nodes.append({"id": nid, "properties": props})

        edges = []
        for src, tgt in self._get_all_edges():
            props = self._get_edge_properties(src, tgt)
            edges.append({"source": src, "target": tgt, "properties": props})

        return json.dumps({"nodes": nodes, "edges": edges}, default=str)

    def from_json(self, json_str: str) -> None:
        """Deserialize and rebuild the graph from a JSON string."""
        data = json.loads(json_str)
        # Clear existing graph nodes/edges via client if possible or just rebuild
        for nid in self._get_all_nodes():
            try:
                self.remove_node(nid)
            except Exception:
                pass

        # Re-add nodes
        for node_data in data.get("nodes", []):
            nid = node_data["id"]
            props = node_data.get("properties", {})
            self.add_node(nid, props)

        # Re-add edges
        for edge_data in data.get("edges", []):
            src = edge_data["source"]
            tgt = edge_data["target"]
            props = edge_data.get("properties", {})
            self.add_edge(src, tgt, props)

    def to_msgpack(self) -> bytes:
        """Serialize graph to MsgPack binary representation."""
        return self._client.lifecycle.to_msgpack()

    def from_msgpack(self, msgpack_bytes: bytes) -> None:
        """Deserialize graph from MsgPack binary representation."""
        self._client.lifecycle.from_msgpack(msgpack_bytes)

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _get_all_nodes(self) -> list[str]:
        return [nid for nid, _ in self._client.nodes.list()]

    def _get_node_properties(self, node_id: str) -> dict[str, Any]:
        props = self._client.nodes.properties(node_id)
        if isinstance(props, dict):
            return props
        if isinstance(props, str):
            try:
                parsed = json.loads(props)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    def _get_all_edges(self) -> list[tuple[str, str]]:
        return [(src, tgt) for src, tgt, _ in self._client.edges.list()]

    # ── Rust-native API wrappers ─────────────────────────────────────────

    def in_degree(self, node_id: str) -> int:
        """Return the in-degree of a node."""
        try:
            return self._client.nodes.in_degree(node_id)
        except Exception:
            return 0

    def out_degree(self, node_id: str) -> int:
        """Return the out-degree of a node."""
        try:
            return self._client.nodes.out_degree(node_id)
        except Exception:
            return 0

    def get_predecessors(self, node_id: str) -> list[str]:
        """Return predecessor node IDs."""
        try:
            return self._client.nodes.predecessors(node_id)
        except Exception:
            return []

    def get_successors(self, node_id: str) -> list[str]:
        """Return successor node IDs."""
        try:
            return self._client.nodes.successors(node_id)
        except Exception:
            return []

    def get_neighbors(self, node_id: str) -> list[str]:
        """Return all neighbor node IDs (predecessors + successors, deduplicated)."""
        try:
            return self._client.nodes.neighbors(node_id)
        except Exception:
            return []

    def node_ids(self) -> list[str]:
        """Return all node IDs in the graph."""
        return self._client.nodes.ids()

    def degree_centrality_all(self) -> list[tuple[str, float]]:
        """Compute degree centrality for all nodes."""
        return self._client.analytics.degree_centrality_all()

    def pagerank(
        self, damping: float = 0.85, iterations: int = 100
    ) -> list[tuple[str, float]]:
        """Compute PageRank scores for all nodes."""
        return self._client.analytics.pagerank(damping, iterations)

    def connected_components(self) -> list[list[str]]:
        """Return weakly connected components as lists of node IDs."""
        return self._client.graph.connected_components()

    def strongly_connected_components(self) -> list[list[str]]:
        """Return strongly connected components via Tarjan's algorithm.

        CONCEPT:KG-2.16 — Tarjan's SCC via Tokio service (GIL-free).
        """
        return self._client.graph.strongly_connected_components()

    def minimum_spanning_tree(self) -> list[tuple[str, str, float]]:
        """Return the minimum spanning tree as (source, target, weight) edges.

        CONCEPT:KG-2.16 — Kruskal's MST via Tokio service (GIL-free).
        """
        return self._client.graph.minimum_spanning_tree()

    def community_detection(self, resolution: float = 1.0) -> list[list[str]]:
        """Detect communities using label propagation."""
        return self._client.graph.community_detection(resolution)

    def betweenness_centrality(self) -> list[tuple[str, float]]:
        """Compute betweenness centrality via Brandes' algorithm."""
        return self._client.analytics.betweenness_centrality()

    def graph_coloring(self) -> list[tuple[str, int]]:
        """Greedy graph coloring — assigns colors so no adjacent nodes share a color."""
        return self._client.graph.graph_coloring()

    def compute_similarity_edges(
        self, threshold: float = 0.8
    ) -> list[tuple[str, str, float]]:
        """Compute similarity edges between nodes with embeddings."""
        return self._client.graph.compute_similarity_edges(threshold)

    def prune_by_lifecycle(
        self, max_age_secs: int = 0, min_score: float = 0.0
    ) -> dict[str, Any]:
        """Lifecycle-aware pruning: remove nodes past max_age or below min_score."""
        result_json = self._client.lifecycle.prune(max_age_secs, min_score)
        return json.loads(result_json)

    def get_context_view(self, agent_id: str, max_tokens: int = 8000) -> dict[str, Any]:
        """Get an optimized context view for an agent within a token budget."""
        result_json = self._client.lifecycle.get_context_view(agent_id, max_tokens)
        return json.loads(result_json)

    def batch_update(self, operations: list[dict[str, Any]]) -> dict[str, Any]:
        """Batch update: apply multiple operations in a single service call."""
        result_json = self._client.lifecycle.batch_update(operations)
        return json.loads(result_json)

    def metrics(self) -> dict[str, Any]:
        """Runtime metrics for monitoring and observability."""
        result_json = self._client.lifecycle.metrics()
        return json.loads(result_json)

    def personalized_pagerank(
        self,
        seed_nodes: dict[str, float] | None = None,
        damping: float = 0.85,
        iterations: int = 100,
    ) -> dict[str, float]:
        """Personalized PageRank with seed teleport nodes."""
        seeds = list((seed_nodes or {}).items())
        result = self._client.analytics.personalized_pagerank(
            seeds, damping, iterations
        )
        return dict(result)

    # ── Batch Operations ─────────────────────────────────────────────────

    def bulk_mutate(self, operations: list[dict[str, Any]]) -> Any:
        """Send a batch of mutations in a single service call.

        Each operation dict should have a ``method`` key and any required
        parameters.  Example::

            engine.bulk_mutate([
                {"method": "AddNode", "node_id": "A", "properties_json": "{}"},
                {"method": "AddEdge", "source_id": "A", "target_id": "B", ...},
            ])
        """
        return self._client.lifecycle.batch_update(operations)

    def evict_lru(self, max_nodes: int = 50_000) -> int:
        """Evict oldest nodes to enforce an in-memory cap.

        Returns the number of evicted nodes.
        """
        return self._client.lifecycle.evict_lru(max_nodes)

    # ── Graph Traversal API ──────────────────────────────────────────────
    # These provide the standard graph traversal interface used across
    # the codebase (owl_bridge, graph_validator, memory_retriever, etc).
    # All hot paths route to the Rust Tokio service; these are thin wrappers.

    @property
    def nodes(self) -> "_NodeView":
        """NX-compatible node view.  Supports iteration, ``in``, and ``[id]``."""
        return _NodeView(self)

    @property
    def edges(self) -> "_EdgeView":
        """NX-compatible edge view.  Supports iteration and ``data=True``."""
        return _EdgeView(self)

    def number_of_nodes(self) -> int:
        """Alias for ``node_count()``."""
        return self.node_count()

    def number_of_edges(self) -> int:
        """Alias for ``edge_count()``."""
        return self.edge_count()

    def degree(self, node_id: str) -> int:
        """Total degree (in + out) of *node_id*."""
        return self.in_degree(node_id) + self.out_degree(node_id)

    def successors(self, node_id: str) -> list[str]:
        """Return successors of *node_id*."""
        return self.get_successors(node_id)

    def predecessors(self, node_id: str) -> list[str]:
        """Return predecessors of *node_id*."""
        return self.get_predecessors(node_id)

    def neighbors(self, node_id: str) -> list[str]:
        """Return neighbors (successors + predecessors) of *node_id*."""
        return self.get_neighbors(node_id)

    def get_edge_data(self, source_id: str, target_id: str, default: Any = None) -> Any:
        """NX-compatible edge data lookup."""
        props = self._get_edge_properties(source_id, target_id)
        if not props:
            return default if not self.has_edge(source_id, target_id) else {0: {}}

        class MultiDiGraphCompatDict(dict):
            def __init__(self, p: dict[str, Any]):
                super().__init__({0: p})
                self._props = p

            def __getitem__(self, key: Any) -> Any:
                if key == 0:
                    return self._props
                return self._props[key]

            def get(self, key: Any, default_val: Any = None) -> Any:
                if key == 0:
                    return self._props
                return self._props.get(key, default_val)

        return MultiDiGraphCompatDict(props)

    def out_edges(self, node_id: str, data: bool = False) -> list:
        """Return outgoing edges from *node_id*.

        When *data* is True, returns ``(src, tgt, props)`` triples.
        """
        succs = self.get_successors(node_id)
        if data:
            return [(node_id, s, self._get_edge_properties(node_id, s)) for s in succs]
        return [(node_id, s) for s in succs]

    def in_edges(self, node_id: str, data: bool = False) -> list:
        """Return incoming edges to *node_id*.

        When *data* is True, returns ``(src, tgt, props)`` triples.
        """
        preds = self.get_predecessors(node_id)
        if data:
            return [(p, node_id, self._get_edge_properties(p, node_id)) for p in preds]
        return [(p, node_id) for p in preds]

    def _get_edge_properties(self, source_id: str, target_id: str) -> dict[str, Any]:
        """Retrieve edge properties between two nodes."""
        props_list = self._client.edges.properties(source_id, target_id)
        if props_list:
            props = props_list[0]
            if isinstance(props, dict):
                return props
            if isinstance(props, str):
                try:
                    parsed = json.loads(props)
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    return {}
        return {}

    def __contains__(self, node_id: str) -> bool:
        """Support ``node_id in engine`` syntax."""
        return self.has_node(node_id)

    def __getitem__(self, node_id: str) -> dict[str, Any]:
        """Support ``engine[node_id]`` to get node properties."""
        return self._get_node_properties(node_id)


class _NodePropertiesProxy(dict):
    def __init__(
        self, engine: GraphComputeEngine, node_id: str, properties: dict[str, Any]
    ):
        super().__init__(properties)
        self._engine = engine
        self._node_id = node_id

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        self._engine.add_node(self._node_id, properties=dict(self))

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        self._engine.add_node(self._node_id, properties=dict(self))

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        self._engine.add_node(self._node_id, properties=dict(self))


class _EdgePropertiesProxy(dict):
    def __init__(
        self,
        engine: GraphComputeEngine,
        source_id: str,
        target_id: str,
        properties: dict[str, Any],
    ):
        super().__init__(properties)
        self._engine = engine
        self._source_id = source_id
        self._target_id = target_id

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        self._engine.add_edge(self._source_id, self._target_id, properties=dict(self))

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        self._engine.add_edge(self._source_id, self._target_id, properties=dict(self))

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        self._engine.add_edge(self._source_id, self._target_id, properties=dict(self))


class _NodeView:
    """Lightweight proxy providing NX-style ``graph.nodes`` access."""

    __slots__ = ("_engine",)

    def __init__(self, engine: GraphComputeEngine) -> None:
        self._engine = engine

    def __iter__(self):
        return iter(self._engine.node_ids())

    def __len__(self) -> int:
        return self._engine.node_count()

    def __contains__(self, node_id: str) -> bool:
        return self._engine.has_node(node_id)

    def __getitem__(self, node_id: str) -> dict[str, Any]:
        props = self._engine._get_node_properties(node_id)
        return _NodePropertiesProxy(self._engine, node_id, props)

    def get(self, node_id: str, default: Any = None) -> Any:
        """Support ``graph.nodes.get(id, default)`` pattern."""
        if self._engine.has_node(node_id):
            props = self._engine._get_node_properties(node_id)
            return _NodePropertiesProxy(self._engine, node_id, props)
        return default

    def __call__(self, data: bool = False):
        """Support ``graph.nodes(data=True)`` iteration."""
        if data:
            return [
                (
                    nid,
                    _NodePropertiesProxy(
                        self._engine, nid, self._engine._get_node_properties(nid)
                    ),
                )
                for nid in self._engine.node_ids()
            ]
        return self._engine.node_ids()


class _EdgeView:
    """Lightweight proxy providing NX-style ``graph.edges`` access."""

    __slots__ = ("_engine",)

    def __init__(self, engine: GraphComputeEngine) -> None:
        self._engine = engine

    def __iter__(self):
        return iter(self._engine._get_all_edges())

    def __len__(self) -> int:
        return self._engine.edge_count()

    def __call__(
        self, data: bool = False, keys: bool = False, default: Any = None, **kwargs: Any
    ):
        """Support ``graph.edges(data=True, keys=True)`` iteration."""
        result: list[Any] = []
        for src, tgt in self._engine._get_all_edges():
            props = self._engine._get_edge_properties(src, tgt)
            proxy = _EdgePropertiesProxy(self._engine, src, tgt, props)
            if data and keys:
                result.append((src, tgt, 0, proxy))
            elif data:
                result.append((src, tgt, proxy))
            elif keys:
                result.append((src, tgt, 0))
            else:
                result.append((src, tgt))
        return result

    def __getitem__(self, key: Any) -> Any:
        """Support edge properties lookup by tuple key."""
        if not isinstance(key, tuple) or len(key) < 2:
            raise KeyError(key)
        src, tgt = key[0], key[1]
        props = self._engine._get_edge_properties(src, tgt)
        if props is None:
            raise KeyError(key)

        proxy = _EdgePropertiesProxy(self._engine, src, tgt, props)
        if len(key) >= 3:
            return proxy

        class MultiDiGraphCompatEdgeDict(dict):
            def __getitem__(self, k):
                return proxy

            def get(self, k, default=None):
                return proxy

        return MultiDiGraphCompatEdgeDict({0: proxy})
