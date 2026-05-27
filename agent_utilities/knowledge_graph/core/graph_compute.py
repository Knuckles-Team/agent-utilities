# CONCEPT:KG-2.2 - High-Performance Graph Compute Engine
# CONCEPT:ORCH-1.29 - Compiled Orchestration Kernel

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class GraphComputeEngine:
    """Unified graph compute abstraction supporting Rust (epistemic-graph), rustworkx, and NetworkX backends.

    Bridges local-first performance with robust production fallback pathways.
    """

    def __init__(self, backend_type: str = "rust"):
        self.backend_type = backend_type.lower()
        self._rust_graph = None
        self._rx_graph = None
        self._nx_graph = None
        self._rx_node_map: dict[str, Any] = {}
        self._ledger: list[str] = []

        self._init_backend()

    def _init_backend(self) -> None:
        """Dynamically select and initialize the fastest available backend."""
        if self.backend_type in ("epistemic_graph", "rust"):
            try:
                import epistemic_graph

                self._rust_graph = epistemic_graph.EpistemicGraph()
                self.backend_type = "rust"
                logger.info(
                    "Initialized high-performance Rust EpistemicGraph compute backend."
                )
                return
            except ImportError:
                logger.warning(
                    "epistemic_graph not available. Falling back to rustworkx or NetworkX."
                )
                self.backend_type = "rustworkx"

        if self.backend_type in ("rustworkx", "rx"):
            try:
                import rustworkx as rx

                self._rx_graph = rx.PyDiGraph()
                self.backend_type = "rustworkx"
                logger.info("Initialized rustworkx high-performance compute backend.")
                return
            except ImportError:
                logger.warning("rustworkx not available. Falling back to NetworkX.")
                self.backend_type = "networkx"

        # Default NetworkX fallback
        import networkx as nx

        self._nx_graph = nx.MultiDiGraph()
        self.backend_type = "networkx"
        logger.info("Initialized standard NetworkX compute backend.")

    def add_node(self, node_id: str, properties: dict[str, Any]) -> None:
        """Add a node with properties to the active graph compute instance."""
        if self.backend_type == "rust" and self._rust_graph is not None:
            self._rust_graph.add_node(node_id, json.dumps(properties))
        elif self.backend_type == "rustworkx" and self._rx_graph is not None:
            if node_id not in self._rx_node_map:
                idx = self._rx_graph.add_node((node_id, properties))
                self._rx_node_map[node_id] = idx
            self._ledger.append(f"ADD_NODE|{node_id}|{json.dumps(properties)}")
        elif self._nx_graph is not None:
            self._nx_graph.add_node(node_id, **properties)
            self._ledger.append(f"ADD_NODE|{node_id}|{json.dumps(properties)}")

    def add_edge(
        self, source_id: str, target_id: str, properties: dict[str, Any]
    ) -> None:
        """Add a directed edge between two nodes with properties."""
        if self.backend_type == "rust" and self._rust_graph is not None:
            try:
                self._rust_graph.add_edge(source_id, target_id, json.dumps(properties))
            except Exception:
                # Ensure nodes exist
                self._rust_graph.add_node(source_id, "{}")
                self._rust_graph.add_node(target_id, "{}")
                self._rust_graph.add_edge(source_id, target_id, json.dumps(properties))
        elif self.backend_type == "rustworkx" and self._rx_graph is not None:
            if source_id not in self._rx_node_map:
                self.add_node(source_id, {})
            if target_id not in self._rx_node_map:
                self.add_node(target_id, {})
            u = self._rx_node_map[source_id]
            v = self._rx_node_map[target_id]
            self._rx_graph.add_edge(u, v, properties)
            self._ledger.append(
                f"ADD_EDGE|{source_id}|{target_id}|{json.dumps(properties)}"
            )
        elif self._nx_graph is not None:
            self._nx_graph.add_edge(source_id, target_id, **properties)
            self._ledger.append(
                f"ADD_EDGE|{source_id}|{target_id}|{json.dumps(properties)}"
            )

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all of its associated edges."""
        if self.backend_type == "rust" and self._rust_graph is not None:
            self._rust_graph.remove_node(node_id)
        elif self.backend_type == "rustworkx" and self._rx_graph is not None:
            if node_id in self._rx_node_map:
                idx = self._rx_node_map.pop(node_id)
                self._rx_graph.remove_node(idx)
            self._ledger.append(f"REMOVE_NODE|{node_id}")
        elif self._nx_graph is not None:
            if self._nx_graph.has_node(node_id):
                self._nx_graph.remove_node(node_id)
            self._ledger.append(f"REMOVE_NODE|{node_id}")

    def remove_edge(self, source_id: str, target_id: str) -> None:
        """Remove a directed edge between source and target."""
        if self.backend_type == "rust" and self._rust_graph is not None:
            self._rust_graph.remove_edge(source_id, target_id)
        elif self.backend_type == "rustworkx" and self._rx_graph is not None:
            if source_id in self._rx_node_map and target_id in self._rx_node_map:
                u = self._rx_node_map[source_id]
                v = self._rx_node_map[target_id]
                for edge_idx in self._rx_graph.edge_indices():
                    endpoints = self._rx_graph.get_edge_endpoints_by_index(edge_idx)
                    if endpoints == (u, v):
                        self._rx_graph.remove_edge_from_index(edge_idx)
                        break
            self._ledger.append(f"REMOVE_EDGE|{source_id}|{target_id}")
        elif self._nx_graph is not None:
            if self._nx_graph.has_edge(source_id, target_id):
                self._nx_graph.remove_edge(source_id, target_id)
            self._ledger.append(f"REMOVE_EDGE|{source_id}|{target_id}")

    def has_node(self, node_id: str) -> bool:
        """Check if node_id exists in the active graph compute backend."""
        if self.backend_type == "rust" and self._rust_graph is not None:
            return self._rust_graph.has_node(node_id)
        elif self.backend_type == "rustworkx":
            return node_id in self._rx_node_map
        elif self._nx_graph is not None:
            return self._nx_graph.has_node(node_id)
        return False

    def has_edge(self, source_id: str, target_id: str) -> bool:
        """Check if a directed edge exists between source and target."""
        if self.backend_type == "rust" and self._rust_graph is not None:
            return self._rust_graph.has_edge(source_id, target_id)
        elif self.backend_type == "rustworkx" and self._rx_graph is not None:
            if source_id in self._rx_node_map and target_id in self._rx_node_map:
                u = self._rx_node_map[source_id]
                v = self._rx_node_map[target_id]
                return self._rx_graph.has_edge(u, v)
            return False
        elif self._nx_graph is not None:
            return self._nx_graph.has_edge(source_id, target_id)
        return False

    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        if self.backend_type == "rust" and self._rust_graph is not None:
            return self._rust_graph.node_count()
        elif self.backend_type == "rustworkx" and self._rx_graph is not None:
            return len(self._rx_graph.nodes())
        elif self._nx_graph is not None:
            return self._nx_graph.number_of_nodes()
        return 0

    def edge_count(self) -> int:
        """Return the number of edges in the graph."""
        if self.backend_type == "rust" and self._rust_graph is not None:
            return self._rust_graph.edge_count()
        elif self.backend_type == "rustworkx" and self._rx_graph is not None:
            return len(self._rx_graph.edges())
        elif self._nx_graph is not None:
            return self._nx_graph.number_of_edges()
        return 0

    def topological_sort(self) -> list[str]:
        """Perform topological sort across the graph.

        Raises:
            ValueError: If the graph contains dependency cycles.
        """
        if self.backend_type == "rust" and self._rust_graph is not None:
            try:
                return self._rust_graph.topological_sort()
            except Exception as e:
                raise ValueError("Graph contains cycles") from e
        elif self.backend_type == "rustworkx" and self._rx_graph is not None:
            import rustworkx as rx

            try:
                indices = rx.topological_sort(self._rx_graph)
                return [str(self._rx_graph[idx][0]) for idx in indices]
            except rx.DAGHasCycle as e:
                raise ValueError("Graph contains cycles") from e
        elif self._nx_graph is not None:
            import networkx as nx

            try:
                return [str(node) for node in nx.topological_sort(self._nx_graph)]
            except nx.NetworkXUnfeasible as e:
                raise ValueError("Graph contains cycles") from e
        return []

    def find_cycle(self) -> list[str] | None:
        """Detect and return any cycles found within the graph."""
        if self.backend_type == "rust" and self._rust_graph is not None:
            return self._rust_graph.find_cycle()
        elif self.backend_type == "rustworkx" and self._rx_graph is not None:
            import rustworkx as rx

            cycle = rx.digraph_find_cycle(self._rx_graph)
            if cycle:
                nodes = []
                for u, _, _ in cycle:
                    nodes.append(str(self._rx_graph[u][0]))
                if cycle:
                    nodes.append(str(self._rx_graph[cycle[-1][1]][0]))
                return nodes
            return None
        elif self._nx_graph is not None:
            import networkx as nx

            try:
                cycle = nx.find_cycle(self._nx_graph, orientation="original")
                nodes = [edge[0] for edge in cycle]
                if cycle:
                    nodes.append(cycle[-1][1])
                return [str(n) for n in nodes]
            except nx.NetworkXNoCycle:
                return None
        return None

    def get_shortest_path(self, source_id: str, target_id: str) -> list[str] | None:
        """Get the shortest path between source and target nodes."""
        if self.backend_type == "rust" and self._rust_graph is not None:
            return self._rust_graph.get_shortest_path(source_id, target_id)
        elif self.backend_type == "rustworkx" and self._rx_graph is not None:
            import rustworkx as rx

            if source_id in self._rx_node_map and target_id in self._rx_node_map:
                u = self._rx_node_map[source_id]
                v = self._rx_node_map[target_id]
                try:
                    path_indices = rx.dijkstra_shortest_paths(self._rx_graph, u, v)
                    if v in path_indices:
                        return [str(self._rx_graph[idx][0]) for idx in path_indices[v]]
                except Exception:
                    pass
            return None
        elif self._nx_graph is not None:
            import networkx as nx

            try:
                return nx.shortest_path(self._nx_graph, source_id, target_id)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None
        return None

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
        if self.backend_type == "rust" and self._rust_graph is not None:
            nodes = self._rust_graph.get_blast_radius(node_id, max_depth)
            res = []
            for i, nid in enumerate(nodes, start=1):
                res.append({"id": nid, "type": "Node", "depth": min(i, max_depth)})
            return res
        elif self.backend_type == "rustworkx" and self._rx_graph is not None:
            if node_id not in self._rx_node_map:
                return []
            start_idx = self._rx_node_map[node_id]

            def rx_neighbors(idx: Any) -> Any:
                return self._rx_graph.neighbors(idx)

            def rx_info(idx: Any) -> dict[str, Any]:
                node_val = self._rx_graph[idx]
                return {
                    "id": node_val[0],
                    "type": node_val[1].get("type", "Node")
                    if isinstance(node_val[1], dict)
                    else "Node",
                }

            return self._bfs_collect(start_idx, max_depth, rx_neighbors, rx_info)
        elif self._nx_graph is not None:
            if not self._nx_graph.has_node(node_id):
                return []

            def nx_neighbors(nid: str) -> Any:
                return self._nx_graph.neighbors(nid)

            def nx_info(nid: str) -> dict[str, Any]:
                return {
                    "id": nid,
                    "type": self._nx_graph.nodes[nid].get("type", "Node"),
                }

            return self._bfs_collect(node_id, max_depth, nx_neighbors, nx_info)
        return []

    def parse_repository(self, root_path: str) -> None:
        """Parse repository AST natively using the active compute engine backend."""
        if self.backend_type == "rust" and self._rust_graph is not None:
            self._rust_graph.parse_repository(root_path)
        else:
            # Pure-Python fallback directory AST parser
            import os

            ignore_dirs = {
                ".git",
                "node_modules",
                "venv",
                ".venv",
                "__pycache__",
                "build",
                "dist",
                "target",
            }
            for root, dirs, files_list in os.walk(root_path):
                dirs[:] = [d for d in dirs if d not in ignore_dirs]
                for f in files_list:
                    if f.endswith((".py", ".js", ".ts")):
                        file_path = os.path.relpath(os.path.join(root, f), root_path)
                        self.add_node(file_path, {"type": "file", "path": file_path})
                        full_p = os.path.join(root, f)
                        try:
                            with open(full_p, encoding="utf-8") as file_obj:
                                for idx, line in enumerate(file_obj, start=1):
                                    trimmed = line.strip()
                                    if trimmed.startswith("class "):
                                        parts = trimmed.split()
                                        if len(parts) > 1:
                                            name = (
                                                parts[1]
                                                .split("(")[0]
                                                .split(":")[0]
                                                .strip()
                                            )
                                            if name:
                                                node_id = f"{file_path}::{name}"
                                                self.add_node(
                                                    node_id,
                                                    {
                                                        "type": "class",
                                                        "file": file_path,
                                                        "line": idx,
                                                    },
                                                )
                                                self.add_edge(
                                                    file_path,
                                                    node_id,
                                                    {"relationship": "contains"},
                                                )
                                    elif trimmed.startswith("def "):
                                        parts = trimmed.split()
                                        if len(parts) > 1:
                                            name = (
                                                parts[1]
                                                .split("(")[0]
                                                .split(":")[0]
                                                .strip()
                                            )
                                            if name:
                                                node_id = f"{file_path}::{name}"
                                                self.add_node(
                                                    node_id,
                                                    {
                                                        "type": "function",
                                                        "file": file_path,
                                                        "line": idx,
                                                    },
                                                )
                                                self.add_edge(
                                                    file_path,
                                                    node_id,
                                                    {"relationship": "contains"},
                                                )
                                    elif trimmed.startswith("function "):
                                        parts = trimmed.split()
                                        if len(parts) > 1:
                                            name = parts[1].split("(")[0].strip()
                                            if name:
                                                node_id = f"{file_path}::{name}"
                                                self.add_node(
                                                    node_id,
                                                    {
                                                        "type": "function",
                                                        "file": file_path,
                                                        "line": idx,
                                                    },
                                                )
                                                self.add_edge(
                                                    file_path,
                                                    node_id,
                                                    {"relationship": "contains"},
                                                )
                        except Exception:
                            pass

    def vf2_subgraph_match(self, pattern: "GraphComputeEngine") -> list[dict[str, str]]:
        """Find all subgraph isomorphism matches from pattern to target graph."""
        if self.backend_type == "rust" and self._rust_graph is not None:
            return self._rust_graph.vf2_subgraph_match(pattern._rust_graph)

        # Universal pure-Python backtracking isomorphic solver
        matches: list[dict[str, str]] = []
        pattern_nodes = list(pattern._get_all_nodes())
        if not pattern_nodes:
            return matches

        current_mapping: dict[str, str] = {}
        mapped_targets = set()

        def backtrack(idx):
            if idx == len(pattern_nodes):
                matches.append(current_mapping.copy())
                return
            u = pattern_nodes[idx]
            for v in self._get_all_nodes():
                if v in mapped_targets:
                    continue
                u_props = pattern._get_node_properties(u)
                v_props = self._get_node_properties(v)
                match = True
                for k, val in u_props.items():
                    if v_props.get(k) != val:
                        match = False
                        break
                if not match:
                    continue

                edges_compatible = True
                for src, tgt in pattern._get_all_edges():
                    if src == u:
                        mapped_tgt = current_mapping.get(tgt)
                        if mapped_tgt is not None and not self.has_edge(v, mapped_tgt):
                            edges_compatible = False
                            break
                    if tgt == u:
                        mapped_src = current_mapping.get(src)
                        if mapped_src is not None and not self.has_edge(mapped_src, v):
                            edges_compatible = False
                            break

                if edges_compatible:
                    current_mapping[u] = v
                    mapped_targets.add(v)
                    backtrack(idx + 1)
                    current_mapping.pop(u)
                    mapped_targets.remove(v)

        backtrack(0)
        return matches

    def get_ledger(self) -> list[str]:
        """Retrieve the mutation transaction ledger log."""
        if self.backend_type == "rust" and self._rust_graph is not None:
            return self._rust_graph.get_ledger()
        return self._ledger

    def clear_ledger(self) -> None:
        """Clear the mutation transaction ledger log."""
        if self.backend_type == "rust" and self._rust_graph is not None:
            self._rust_graph.clear_ledger()
        else:
            self._ledger.clear()

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
        if self.backend_type == "rust" and self._rust_graph is not None:
            self._rust_graph.apply_ledger(transactions)
        else:
            for tx in transactions:
                op, args = self._parse_ledger_entry(tx)
                if op == "ADD_NODE" and len(args) >= 2:
                    try:
                        self.add_node(args[0], json.loads(args[1]))
                    except Exception:
                        self.add_node(args[0], {})
                elif op == "ADD_EDGE" and len(args) >= 3:
                    try:
                        self.add_edge(args[0], args[1], json.loads(args[2]))
                    except Exception:
                        self.add_edge(args[0], args[1], {})
                elif op == "REMOVE_NODE" and len(args) >= 1:
                    self.remove_node(args[0])
                elif op == "REMOVE_EDGE" and len(args) >= 2:
                    self.remove_edge(args[0], args[1])

    def to_json(self) -> str:
        """Serialize graph to JSON string representation."""
        if self.backend_type == "rust" and self._rust_graph is not None:
            return self._rust_graph.to_json()

        # Serialization fallback
        data: dict[str, list[Any]] = {"nodes": [], "edges": []}
        for node_id in self._get_all_nodes():
            data["nodes"].append(
                (node_id, json.dumps(self._get_node_properties(node_id)))
            )
        for src, tgt in self._get_all_edges():
            data["edges"].append((src, tgt, json.dumps({})))
        return json.dumps(data)

    def from_json(self, json_str: str) -> None:
        """Deserialize graph from JSON string representation."""
        if self.backend_type == "rust" and self._rust_graph is not None:
            self._rust_graph.from_json(json_str)
        else:
            try:
                data = json.loads(json_str)
                for node_id, props_str in data.get("nodes", []):
                    try:
                        self.add_node(node_id, json.loads(props_str))
                    except Exception:
                        self.add_node(node_id, {})
                for src, tgt, props_str in data.get("edges", []):
                    try:
                        self.add_edge(src, tgt, json.loads(props_str))
                    except Exception:
                        self.add_edge(src, tgt, {})
            except Exception:
                pass

    def _get_all_nodes(self) -> list[str]:
        if self.backend_type == "rust" and self._rust_graph is not None:
            return [nid for nid, _ in self._rust_graph.get_nodes()]
        elif self.backend_type == "rustworkx" and self._rx_graph is not None:
            return list(self._rx_node_map.keys())
        elif self._nx_graph is not None:
            return list(self._nx_graph.nodes)
        return []

    def _get_node_properties(self, node_id: str) -> dict[str, Any]:
        if self.backend_type == "rust" and self._rust_graph is not None:
            props_str = self._rust_graph.get_node_properties(node_id)
            if props_str is not None:
                try:
                    return json.loads(props_str)
                except Exception:
                    return {}
            return {}
        elif self.backend_type == "rustworkx" and self._rx_graph is not None:
            idx = self._rx_node_map.get(node_id)
            if idx is not None:
                node_data = self._rx_graph[idx]
                return node_data[1] if len(node_data) > 1 else {}
            return {}
        elif self._nx_graph is not None:
            if self._nx_graph.has_node(node_id):
                return dict(self._nx_graph.nodes[node_id])
            return {}
        return {}

    def _get_all_edges(self) -> list[tuple[str, str]]:
        if self.backend_type == "rust" and self._rust_graph is not None:
            return [(src, tgt) for src, tgt, _ in self._rust_graph.get_edges()]
        elif self.backend_type == "rustworkx" and self._rx_graph is not None:
            res = []
            for u, v, _ in self._rx_graph.weighted_edge_list():
                u_id = self._rx_graph[u][0]
                v_id = self._rx_graph[v][0]
                res.append((u_id, v_id))
            return res
        elif self._nx_graph is not None:
            return [(u, v) for u, v, *_ in self._nx_graph.edges]
        return []
