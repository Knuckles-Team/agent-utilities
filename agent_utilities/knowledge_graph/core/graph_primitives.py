"""Drop-in replacement for rustworkx graph types used in formal_reasoning_core.

Provides PyDiGraph and PyGraph with an API surface compatible with the
subset of rustworkx used in this codebase. Backed by plain Python dicts.
This eliminates the rustworkx dependency while maintaining call-site
compatibility throughout the formal reasoning module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
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
        g = self.__class__()
        g._nodes = dict(self._nodes)
        g._next_id = self._next_id
        return g


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
        for eid in self._incident_edge_ids(idx):
            src, tgt, _ = self._edges.pop(eid)
            self._unlink_directed_edge(src, tgt, eid)
        self._out_adj.pop(idx, None)
        self._in_adj.pop(idx, None)
        super().remove_node(idx)

    def _incident_edge_ids(self, idx: int) -> list[int]:
        """Edge ids where idx is either endpoint."""
        return [
            eid
            for eid, (src, tgt, _) in self._edges.items()
            if src == idx or tgt == idx
        ]

    def _unlink_directed_edge(self, src: int, tgt: int, eid: int) -> None:
        """Remove eid from the out/in adjacency lists of its endpoints."""
        if src in self._out_adj:
            self._out_adj[src] = [(t, e) for t, e in self._out_adj[src] if e != eid]
        if tgt in self._in_adj:
            self._in_adj[tgt] = [(s, e) for s, e in self._in_adj[tgt] if e != eid]

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
        for eid in self._incident_edge_ids(idx):
            src, tgt, _ = self._edges.pop(eid)
            self._unlink_undirected_edge(src, tgt, eid)
        self._adj.pop(idx, None)
        super().remove_node(idx)

    def _incident_edge_ids(self, idx: int) -> list[int]:
        """Edge ids where idx is either endpoint."""
        return [
            eid
            for eid, (src, tgt, _) in self._edges.items()
            if src == idx or tgt == idx
        ]

    def _unlink_undirected_edge(self, src: int, tgt: int, eid: int) -> None:
        """Remove eid from the shared adjacency lists of its endpoints."""
        if src in self._adj:
            self._adj[src] = [(t, e) for t, e in self._adj[src] if e != eid]
        if tgt in self._adj:
            self._adj[tgt] = [(s, e) for s, e in self._adj[tgt] if e != eid]

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


def _compute_in_degrees(graph: PyDiGraph) -> dict[int, int]:
    """Count incoming edges per node (0 for nodes with none)."""
    in_degree = {n: 0 for n in graph.node_indices()}
    for _, (_, tgt, _) in graph._edges.items():
        if tgt in in_degree:
            in_degree[tgt] += 1
    return in_degree


def topological_sort(graph: PyDiGraph) -> list[int]:
    """Kahn's algorithm for topological sorting."""
    in_degree = _compute_in_degrees(graph)

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


def _next_generation(
    graph: PyDiGraph, current_gen: list[int], in_degree: dict[int, int]
) -> list[int]:
    """Decrement in-degree for successors of current_gen; return newly-zero nodes.

    Mutates in_degree in place, matching the caller's expectation that the
    same dict is threaded through every generation.
    """
    next_gen_set: dict[int, int] = {}
    for node in current_gen:
        for tgt, _ in graph._out_adj.get(node, []):
            in_degree[tgt] -= 1
            if in_degree[tgt] == 0:
                next_gen_set[tgt] = 1
    return list(next_gen_set.keys())


def topological_generations(graph: PyDiGraph) -> list[list[int]]:
    """Group nodes by topological level (parallel waves).

    Returns a list of lists, where each inner list contains node indices
    that can be executed in parallel (all their dependencies are in earlier
    generations).
    """
    in_degree = _compute_in_degrees(graph)
    current_gen = [n for n, d in in_degree.items() if d == 0]
    generations: list[list[int]] = []

    while current_gen:
        generations.append(current_gen)
        current_gen = _next_generation(graph, current_gen, in_degree)

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


@dataclass
class _VF2Context:
    """Read-only state shared by every recursion step of the VF2 backtracker."""

    graph: PyDiGraph
    pattern: PyDiGraph
    pattern_nodes: list[int]
    graph_nodes: list[int]
    node_matcher: Any = None


def _vf2_node_compatible(ctx: _VF2Context, g_node: int, p_node: int) -> bool:
    """True if g_node may stand in for p_node under ctx.node_matcher."""
    if ctx.node_matcher is None:
        return True
    return bool(ctx.node_matcher(ctx.graph[g_node], ctx.pattern[p_node]))


def _vf2_edges_compatible(
    ctx: _VF2Context,
    mapping: dict[int, int],
    p_idx: int,
    p_node: int,
    g_node: int,
) -> bool:
    """True if every pattern edge between p_node and already-mapped pattern
    nodes has a corresponding edge between g_node and their graph images.
    """
    for prev_p_idx in range(p_idx):
        prev_p = ctx.pattern_nodes[prev_p_idx]
        prev_g = mapping[prev_p]

        if ctx.pattern.has_edge(prev_p, p_node) and not ctx.graph.has_edge(
            prev_g, g_node
        ):
            return False
        if ctx.pattern.has_edge(p_node, prev_p) and not ctx.graph.has_edge(
            g_node, prev_g
        ):
            return False
    return True


def _vf2_candidates(
    ctx: _VF2Context, mapping: dict[int, int], p_idx: int, remaining: list[int]
) -> list[dict[int, int]]:
    """Try every unused graph node as the image of pattern_nodes[p_idx],
    recursing into _vf2_match on each compatible choice.
    """
    p_node = ctx.pattern_nodes[p_idx]
    used = set(mapping.values())
    results: list[dict[int, int]] = []

    for g_node in ctx.graph_nodes:
        if g_node in used:
            continue
        if not _vf2_node_compatible(ctx, g_node, p_node):
            continue
        if not _vf2_edges_compatible(ctx, mapping, p_idx, p_node, g_node):
            continue

        mapping[p_node] = g_node
        results.extend(_vf2_match(ctx, mapping, p_idx + 1, remaining))
        del mapping[p_node]

    return results


def _vf2_match(
    ctx: _VF2Context, mapping: dict[int, int], p_idx: int, remaining: list[int]
) -> list[dict[int, int]]:
    """Recursive VF2 backtracking step: complete or extend mapping.

    ``remaining`` is threaded through unchanged (unused by the matching
    logic itself, same as in the original implementation) to preserve the
    original recursion signature exactly.
    """
    if p_idx >= len(ctx.pattern_nodes):
        return [dict(mapping)]
    return _vf2_candidates(ctx, mapping, p_idx, remaining)


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

    ctx = _VF2Context(graph, pattern, pattern_nodes, graph_nodes, node_matcher)
    return _vf2_match(ctx, {}, 0, list(graph_nodes))
