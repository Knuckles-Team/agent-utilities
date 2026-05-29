"""Drop-in replacement for rustworkx graph types used in formal_reasoning_core.

Provides PyDiGraph and PyGraph with an API surface compatible with the
subset of rustworkx used in this codebase. Backed by plain Python dicts.
This eliminates the rustworkx dependency while maintaining call-site
compatibility throughout the formal reasoning module.
"""

from __future__ import annotations

from collections import deque
from typing import Any


class _BaseGraph:
    """Shared base for directed and undirected graphs."""

    def __init__(self) -> None:
        self._nodes: dict[int, Any] = {}
        self._next_id: int = 0

    def add_node(self, data: Any) -> int:
        idx = self._next_id
        self._nodes[idx] = data
        self._next_id += 1
        return idx

    def remove_node(self, idx: int) -> None:
        self._nodes.pop(idx, None)

    def num_nodes(self) -> int:
        return len(self._nodes)

    def node_indices(self) -> list[int]:
        return list(self._nodes.keys())

    def __getitem__(self, idx: int) -> Any:
        return self._nodes[idx]

    def __len__(self) -> int:
        return len(self._nodes)

    def copy(self) -> _BaseGraph:
        raise NotImplementedError


class PyDiGraph(_BaseGraph):
    """Minimal directed graph matching the rustworkx PyDiGraph API subset."""

    def __init__(self) -> None:
        super().__init__()
        # edges: dict[edge_idx] -> (src, tgt, data)
        self._edges: dict[int, tuple[int, int, Any]] = {}
        self._next_edge_id: int = 0
        # adjacency: src -> [(tgt, edge_idx)]
        self._out_adj: dict[int, list[tuple[int, int]]] = {}
        self._in_adj: dict[int, list[tuple[int, int]]] = {}

    def add_node(self, data: Any) -> int:
        idx = super().add_node(data)
        self._out_adj[idx] = []
        self._in_adj[idx] = []
        return idx

    def remove_node(self, idx: int) -> None:
        # Remove all incident edges
        edge_ids_to_remove = []
        for eid, (src, tgt, _) in self._edges.items():
            if src == idx or tgt == idx:
                edge_ids_to_remove.append(eid)
        for eid in edge_ids_to_remove:
            src, tgt, _ = self._edges.pop(eid)
            if src in self._out_adj:
                self._out_adj[src] = [(t, e) for t, e in self._out_adj[src] if e != eid]
            if tgt in self._in_adj:
                self._in_adj[tgt] = [(s, e) for s, e in self._in_adj[tgt] if e != eid]
        self._out_adj.pop(idx, None)
        self._in_adj.pop(idx, None)
        super().remove_node(idx)

    def add_edge(self, src: int, tgt: int, data: Any = None) -> int:
        eid = self._next_edge_id
        self._edges[eid] = (src, tgt, data)
        self._next_edge_id += 1
        self._out_adj.setdefault(src, []).append((tgt, eid))
        self._in_adj.setdefault(tgt, []).append((src, eid))
        return eid

    def successor_indices(self, idx: int) -> list[int]:
        return [tgt for tgt, _ in self._out_adj.get(idx, [])]

    def predecessor_indices(self, idx: int) -> list[int]:
        return [src for src, _ in self._in_adj.get(idx, [])]

    def get_edge_data(self, src: int, tgt: int) -> Any:
        for _, (s, t, d) in self._edges.items():
            if s == src and t == tgt:
                return d
        return None

    def get_edge_data_by_index(self, eid: int) -> Any:
        if eid in self._edges:
            return self._edges[eid][2]
        return None

    def incident_edges(self, idx: int) -> list[int]:
        result = []
        for eid, (src, tgt, _) in self._edges.items():
            if src == idx or tgt == idx:
                result.append(eid)
        return result

    def edge_indices(self) -> list[int]:
        return list(self._edges.keys())

    def weighted_edge_list(self) -> list[tuple[int, int, Any]]:
        return [(src, tgt, data) for src, tgt, data in self._edges.values()]

    def has_edge(self, src: int, tgt: int) -> bool:
        """Check if a directed edge exists between src and tgt."""
        return any(s == src and t == tgt for s, t, _ in self._edges.values())

    def remove_edge(self, src: int, tgt: int) -> None:
        """Remove a directed edge between src and tgt."""
        to_remove = None
        for eid, (s, t, _) in self._edges.items():
            if s == src and t == tgt:
                to_remove = eid
                break
        if to_remove is not None:
            self._edges.pop(to_remove)
            self._out_adj[src] = [
                (t, e) for t, e in self._out_adj.get(src, []) if e != to_remove
            ]
            self._in_adj[tgt] = [
                (s, e) for s, e in self._in_adj.get(tgt, []) if e != to_remove
            ]

    def successors(self, idx: int) -> list[Any]:
        """Return data of successor nodes."""
        return [
            self._nodes[tgt]
            for tgt in self.successor_indices(idx)
            if tgt in self._nodes
        ]

    def copy(self) -> PyDiGraph:
        g = PyDiGraph()
        g._nodes = dict(self._nodes)
        g._next_id = self._next_id
        g._edges = dict(self._edges)
        g._next_edge_id = self._next_edge_id
        g._out_adj = {k: list(v) for k, v in self._out_adj.items()}
        g._in_adj = {k: list(v) for k, v in self._in_adj.items()}
        return g


class PyGraph(_BaseGraph):
    """Minimal undirected graph matching the rustworkx PyGraph API subset."""

    def __init__(self) -> None:
        super().__init__()
        self._edges: dict[int, tuple[int, int, Any]] = {}
        self._next_edge_id: int = 0
        self._adj: dict[int, list[tuple[int, int]]] = {}

    def add_node(self, data: Any) -> int:
        idx = super().add_node(data)
        self._adj[idx] = []
        return idx

    def remove_node(self, idx: int) -> None:
        edge_ids_to_remove = []
        for eid, (src, tgt, _) in self._edges.items():
            if src == idx or tgt == idx:
                edge_ids_to_remove.append(eid)
        for eid in edge_ids_to_remove:
            src, tgt, _ = self._edges.pop(eid)
            if src in self._adj:
                self._adj[src] = [(t, e) for t, e in self._adj[src] if e != eid]
            if tgt in self._adj:
                self._adj[tgt] = [(s, e) for s, e in self._adj[tgt] if e != eid]
        self._adj.pop(idx, None)
        super().remove_node(idx)

    def add_edge(self, src: int, tgt: int, data: Any = None) -> int:
        eid = self._next_edge_id
        self._edges[eid] = (src, tgt, data)
        self._next_edge_id += 1
        self._adj.setdefault(src, []).append((tgt, eid))
        self._adj.setdefault(tgt, []).append((src, eid))
        return eid

    def degree(self, idx: int) -> int:
        return len(self._adj.get(idx, []))

    def neighbors(self, idx: int) -> list[int]:
        return list({tgt for tgt, _ in self._adj.get(idx, [])})

    def edge_indices(self) -> list[int]:
        return list(self._edges.keys())

    def weighted_edge_list(self) -> list[tuple[int, int, Any]]:
        return [(src, tgt, data) for src, tgt, data in self._edges.values()]

    def get_edge_data_by_index(self, eid: int) -> Any:
        if eid in self._edges:
            return self._edges[eid][2]
        return None

    def get_edge_endpoints_by_index(self, eid: int) -> tuple[int, int] | None:
        if eid in self._edges:
            return (self._edges[eid][0], self._edges[eid][1])
        return None

    def copy(self) -> PyGraph:
        g = PyGraph()
        g._nodes = dict(self._nodes)
        g._next_id = self._next_id
        g._edges = dict(self._edges)
        g._next_edge_id = self._next_edge_id
        g._adj = {k: list(v) for k, v in self._adj.items()}
        return g


# ── Module-level functions matching rustworkx API ─────────────────────────


def topological_sort(graph: PyDiGraph) -> list[int]:
    """Kahn's algorithm for topological sorting."""
    in_degree = {n: 0 for n in graph.node_indices()}
    for _, (_, tgt, _) in graph._edges.items():
        if tgt in in_degree:
            in_degree[tgt] += 1

    queue = deque([n for n, d in in_degree.items() if d == 0])
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for tgt, _ in graph._out_adj.get(node, []):
            in_degree[tgt] -= 1
            if in_degree[tgt] == 0:
                queue.append(tgt)

    if len(result) != graph.num_nodes():
        raise ValueError("Graph contains a cycle")
    return result


def topological_generations(graph: PyDiGraph) -> list[list[int]]:
    """Group nodes by topological level (parallel waves).

    Returns a list of lists, where each inner list contains node indices
    that can be executed in parallel (all their dependencies are in earlier
    generations).
    """
    in_degree: dict[int, int] = {n: 0 for n in graph.node_indices()}
    for _, (_, tgt, _) in graph._edges.items():
        if tgt in in_degree:
            in_degree[tgt] += 1

    current_gen = [n for n, d in in_degree.items() if d == 0]
    generations: list[list[int]] = []

    while current_gen:
        generations.append(current_gen)
        next_gen_set: dict[int, int] = {}
        for node in current_gen:
            for tgt, _ in graph._out_adj.get(node, []):
                in_degree[tgt] -= 1
                if in_degree[tgt] == 0:
                    next_gen_set[tgt] = 1
        current_gen = list(next_gen_set.keys())

    total = sum(len(g) for g in generations)
    if total != graph.num_nodes():
        raise ValueError("Graph contains a cycle")
    return generations


def is_connected(graph: PyGraph) -> bool:
    """Check if undirected graph is connected via BFS."""
    nodes = graph.node_indices()
    if len(nodes) <= 1:
        return True
    visited: set[int] = set()
    queue = deque([nodes[0]])
    visited.add(nodes[0])
    while queue:
        curr = queue.popleft()
        for neighbor, _ in graph._adj.get(curr, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return len(visited) == len(nodes)


def descendants(graph: PyDiGraph, node: int) -> set[int]:
    """Find all descendants of a node via BFS."""
    visited: set[int] = set()
    queue = deque(graph.successor_indices(node))
    while queue:
        curr = queue.popleft()
        if curr not in visited:
            visited.add(curr)
            queue.extend(graph.successor_indices(curr))
    return visited


def graph_greedy_color(graph: PyGraph) -> dict[int, int]:
    """Greedy graph coloring — smallest available color for each node."""
    colors: dict[int, int] = {}
    for node in graph.node_indices():
        neighbor_colors = {colors[n] for n in graph.neighbors(node) if n in colors}
        color = 0
        while color in neighbor_colors:
            color += 1
        colors[node] = color
    return colors


def dfs_search(graph: PyGraph, starts: list[int]) -> list[Any]:
    """DFS traversal returning visit events."""

    class DFSEvent:
        def __init__(self, node: int):
            self.node = node

    visited: set[int] = set()
    events: list[DFSEvent] = []
    for start in starts:
        stack = [start]
        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                events.append(DFSEvent(curr))
                for neighbor in graph.neighbors(curr):
                    if neighbor not in visited:
                        stack.append(neighbor)
    return events


def is_subgraph_isomorphic(
    graph: PyDiGraph,
    pattern: PyDiGraph,
    induced: bool = False,
    node_matcher: Any = None,
) -> bool:
    """Check if pattern is a subgraph of graph (simplified VF2).

    Uses brute-force backtracking for small graphs. For production-scale
    graphs, this should delegate to the Rust backend's vf2_subgraph_match.
    """
    mappings = vf2_mapping(
        graph, pattern, node_matcher=node_matcher, subgraph=True, induced=induced
    )
    return len(list(mappings)) > 0


def vf2_mapping(
    graph: PyDiGraph,
    pattern: PyDiGraph,
    node_matcher: Any = None,
    subgraph: bool = True,
    induced: bool = False,
) -> list[dict[int, int]]:
    """VF2 subgraph isomorphism mapping (simplified backtracking).

    Returns list of dicts mapping graph node indices to pattern node indices.
    For small graphs only — large graphs should use the Rust backend.
    """
    pattern_nodes = pattern.node_indices()
    graph_nodes = graph.node_indices()

    if len(pattern_nodes) > len(graph_nodes):
        return []

    def _match(
        mapping: dict[int, int], p_idx: int, remaining: list[int]
    ) -> list[dict[int, int]]:
        if p_idx >= len(pattern_nodes):
            return [dict(mapping)]

        p_node = pattern_nodes[p_idx]
        results: list[dict[int, int]] = []
        used = set(mapping.values())

        for g_node in graph_nodes:
            if g_node in used:
                continue

            # Node compatibility check
            if node_matcher is not None:
                if not node_matcher(graph[g_node], pattern[p_node]):
                    continue

            # Edge compatibility check
            compatible = True
            for prev_p_idx in range(p_idx):
                prev_p = pattern_nodes[prev_p_idx]
                prev_g = mapping[prev_p]

                # Check forward edges
                if pattern.has_edge(prev_p, p_node):
                    if not graph.has_edge(prev_g, g_node):
                        compatible = False
                        break
                if pattern.has_edge(p_node, prev_p):
                    if not graph.has_edge(g_node, prev_g):
                        compatible = False
                        break

            if compatible:
                mapping[p_node] = g_node
                results.extend(_match(mapping, p_idx + 1, remaining))
                del mapping[p_node]

        return results

    return _match({}, 0, list(graph_nodes))
