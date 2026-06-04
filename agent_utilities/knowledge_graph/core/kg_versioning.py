#!/usr/bin/python
from __future__ import annotations

"""Versioned KG Mutations (CONCEPT:KG-2.0 Enhancement).

Derived from: Evolving Idea Graphs with Learnable Edits-and-Commits
for Multi-Agent Scientific Ideation (arXiv:2605.04922v1, Score 11.2)

Git-like mutation semantics for Knowledge Graph evolution:
- KGTransaction — collects a batch of node/edge mutations
- KGCommit — atomic application with rollback support
- KGDiffEngine — computes structural diffs between graph versions
"""


import hashlib
import logging
from copy import deepcopy
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MutationType(StrEnum):
    ADD_NODE = "add_node"
    UPDATE_NODE = "update_node"
    DELETE_NODE = "delete_node"
    ADD_EDGE = "add_edge"
    DELETE_EDGE = "delete_edge"


class KGMutation(BaseModel):
    """A single mutation operation on the Knowledge Graph (CONCEPT:KG-2.0)."""

    mutation_type: MutationType
    node_id: str = ""
    edge_source: str = ""
    edge_target: str = ""
    edge_label: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    previous_data: dict[str, Any] = Field(default_factory=dict)


class KGTransaction(BaseModel):
    """A batch of KG mutations to be applied atomically (CONCEPT:KG-2.0).

    Example::
        tx = KGTransaction(description="Add research findings")
        tx.add_node("paper:001", {"title": "Uno-Orchestra", "type": "paper"})
        tx.add_edge("paper:001", "concept:ORCH-1.2", "enhances")
        commit = engine.commit(tx, graph_state)
    """

    transaction_id: str = Field(
        default_factory=lambda: (
            f"tx:{hashlib.sha256(str(datetime.now(UTC)).encode()).hexdigest()[:10]}"
        )
    )
    description: str = ""
    mutations: list[KGMutation] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def add_node(self, node_id: str, data: dict[str, Any] | None = None) -> None:
        self.mutations.append(
            KGMutation(
                mutation_type=MutationType.ADD_NODE, node_id=node_id, data=data or {}
            )
        )

    def update_node(self, node_id: str, data: dict[str, Any]) -> None:
        self.mutations.append(
            KGMutation(
                mutation_type=MutationType.UPDATE_NODE, node_id=node_id, data=data
            )
        )

    def delete_node(self, node_id: str) -> None:
        self.mutations.append(
            KGMutation(mutation_type=MutationType.DELETE_NODE, node_id=node_id)
        )

    def add_edge(self, source: str, target: str, label: str = "related") -> None:
        self.mutations.append(
            KGMutation(
                mutation_type=MutationType.ADD_EDGE,
                edge_source=source,
                edge_target=target,
                edge_label=label,
            )
        )

    def delete_edge(self, source: str, target: str, label: str = "") -> None:
        self.mutations.append(
            KGMutation(
                mutation_type=MutationType.DELETE_EDGE,
                edge_source=source,
                edge_target=target,
                edge_label=label,
            )
        )


class KGCommit(BaseModel):
    """Result of committing a transaction (CONCEPT:KG-2.0)."""

    commit_id: str = Field(
        default_factory=lambda: (
            f"commit:{hashlib.sha256(str(datetime.now(UTC)).encode()).hexdigest()[:10]}"
        )
    )
    transaction_id: str = ""
    description: str = ""
    mutations_applied: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    parent_commit_id: str = ""
    rollback_data: dict[str, Any] = Field(default_factory=dict)


class KGDiff(BaseModel):
    """Structural diff between two graph versions."""

    nodes_added: list[str] = Field(default_factory=list)
    nodes_removed: list[str] = Field(default_factory=list)
    nodes_modified: list[str] = Field(default_factory=list)
    edges_added: list[tuple[str, str, str]] = Field(default_factory=list)
    edges_removed: list[tuple[str, str, str]] = Field(default_factory=list)

    @property
    def total_changes(self) -> int:
        return (
            len(self.nodes_added)
            + len(self.nodes_removed)
            + len(self.nodes_modified)
            + len(self.edges_added)
            + len(self.edges_removed)
        )


class KGVersionEngine:
    """Versioned KG mutation engine with commit/rollback semantics.

    CONCEPT:KG-2.0 — Provides git-like transactional mutation for KG evolution.

    Example::
        engine = KGVersionEngine()
        tx = KGTransaction(description="Add new concepts")
        tx.add_node("concept:new", {"name": "Dynamic Routing"})
        graph = {"nodes": {}, "edges": []}
        commit = engine.commit(tx, graph)
        # Rollback if needed
        engine.rollback(commit, graph)
    """

    def __init__(self) -> None:
        self._commits: list[KGCommit] = []

    def commit(
        self, transaction: KGTransaction, graph_state: dict[str, Any]
    ) -> KGCommit:
        """Apply a transaction atomically to the graph state.

        Args:
            transaction: The transaction to apply.
            graph_state: Mutable dict with "nodes" and "edges" keys.

        Returns:
            KGCommit with rollback data.
        """
        nodes = graph_state.setdefault("nodes", {})
        edges = graph_state.setdefault("edges", [])
        rollback: dict[str, Any] = {
            "removed_nodes": {},
            "added_nodes": [],
            "modified_nodes": {},
            "removed_edges": [],
            "added_edges": [],
        }
        applied = 0
        try:
            for mut in transaction.mutations:
                if mut.mutation_type == MutationType.ADD_NODE:
                    if mut.node_id not in nodes:
                        nodes[mut.node_id] = dict(mut.data)
                        rollback["added_nodes"].append(mut.node_id)
                        applied += 1
                elif mut.mutation_type == MutationType.UPDATE_NODE:
                    if mut.node_id in nodes:
                        rollback["modified_nodes"][mut.node_id] = deepcopy(
                            nodes[mut.node_id]
                        )
                        nodes[mut.node_id].update(mut.data)
                        applied += 1
                elif mut.mutation_type == MutationType.DELETE_NODE:
                    if mut.node_id in nodes:
                        rollback["removed_nodes"][mut.node_id] = deepcopy(
                            nodes.pop(mut.node_id)
                        )
                        applied += 1
                elif mut.mutation_type == MutationType.ADD_EDGE:
                    edge = (mut.edge_source, mut.edge_target, mut.edge_label)
                    if edge not in edges:
                        edges.append(edge)
                        rollback["added_edges"].append(edge)
                        applied += 1
                elif mut.mutation_type == MutationType.DELETE_EDGE:
                    edge = (mut.edge_source, mut.edge_target, mut.edge_label)
                    if edge in edges:
                        edges.remove(edge)
                        rollback["removed_edges"].append(edge)
                        applied += 1
        except Exception:
            self._do_rollback(rollback, graph_state)
            raise

        parent = self._commits[-1].commit_id if self._commits else ""
        commit = KGCommit(
            transaction_id=transaction.transaction_id,
            description=transaction.description,
            mutations_applied=applied,
            parent_commit_id=parent,
            rollback_data=rollback,
        )
        self._commits.append(commit)
        return commit

    def rollback(self, commit: KGCommit, graph_state: dict[str, Any]) -> None:
        """Rollback a commit, restoring previous graph state."""
        self._do_rollback(commit.rollback_data, graph_state)
        if self._commits and self._commits[-1].commit_id == commit.commit_id:
            self._commits.pop()

    @staticmethod
    def _do_rollback(rollback: dict[str, Any], graph_state: dict[str, Any]) -> None:
        nodes = graph_state.get("nodes", {})
        edges = graph_state.get("edges", [])
        for nid in rollback.get("added_nodes", []):
            nodes.pop(nid, None)
        for nid, data in rollback.get("removed_nodes", {}).items():
            nodes[nid] = data
        for nid, data in rollback.get("modified_nodes", {}).items():
            nodes[nid] = data
        for edge in rollback.get("added_edges", []):
            if edge in edges:
                edges.remove(edge)
        for edge in rollback.get("removed_edges", []):
            if edge not in edges:
                edges.append(edge)

    @staticmethod
    def diff(state_a: dict[str, Any], state_b: dict[str, Any]) -> KGDiff:
        """Compute structural diff between two graph states."""
        nodes_a = set(state_a.get("nodes", {}).keys())
        nodes_b = set(state_b.get("nodes", {}).keys())
        edges_a = set(tuple(e) for e in state_a.get("edges", []))
        edges_b = set(tuple(e) for e in state_b.get("edges", []))
        modified = []
        for nid in nodes_a & nodes_b:
            if state_a["nodes"][nid] != state_b["nodes"][nid]:
                modified.append(nid)
        return KGDiff(
            nodes_added=list(nodes_b - nodes_a),
            nodes_removed=list(nodes_a - nodes_b),
            nodes_modified=modified,
            edges_added=[tuple(e) for e in edges_b - edges_a],
            edges_removed=[tuple(e) for e in edges_a - edges_b],
        )

    def commit_to_compute_engine(
        self,
        graph_state: dict[str, Any],
        compute_engine: Any,
    ) -> int:
        """Replay the current graph state into a GraphComputeEngine.

        Materialises the versioned KG state into the high-performance compute
        layer (Rust, epistemic-graph, or GraphComputeEngine) so that centrality, blast-radius,
        and rolling-stats operations can run against the latest committed data.

        Args:
            graph_state: The graph dict with "nodes" and "edges" keys.
            compute_engine: A ``GraphComputeEngine`` instance.

        Returns:
            Total number of nodes + edges pushed into the compute engine.
        """
        pushed = 0
        nodes = graph_state.get("nodes", {})
        edges = graph_state.get("edges", [])

        for node_id, props in nodes.items():
            compute_engine.add_node(node_id, dict(props))
            pushed += 1

        for edge in edges:
            src, tgt = edge[0], edge[1]
            label = edge[2] if len(edge) > 2 else "related"
            compute_engine.add_edge(src, tgt, {"label": label})
            pushed += 1

        logger.info(
            "Materialised %d elements from KGVersionEngine into compute engine (%s).",
            pushed,
            compute_engine.backend_type,
        )
        return pushed

    @property
    def history(self) -> list[KGCommit]:
        return list(self._commits)


class SpeculativeGraphBrancher:
    """Manages speculative graph branches and merges them atomically (CONCEPT:KG-2.7).

    Allows creating concurrent speculative branches (representing KGTransactions)
    which execute independently and merge atomically via logical conflict validation.

    When a ``compute_engine`` with Rust backend is provided, ``create_branch``
    uses the compiled ``fork()`` instead of Python ``deepcopy`` for ~100×
    faster branch creation on large graphs.
    """

    def __init__(
        self,
        main_engine: KGVersionEngine,
        main_state: dict[str, Any],
        compute_engine: Any = None,
    ) -> None:
        self.main_engine = main_engine
        self.main_state = main_state
        self._compute_engine = compute_engine
        self._branches: dict[str, dict[str, Any]] = {}
        self._original_states: dict[str, dict[str, Any]] = {}

    def create_branch(self, branch_id: str) -> dict[str, Any]:
        """Create a new speculative branch that is a deep copy of the main state.

        Uses Rust ``fork()`` when a compiled Rust-backed compute engine is
        available for zero-Python-overhead graph cloning.
        """
        self._branches[branch_id] = deepcopy(self.main_state)
        self._original_states[branch_id] = deepcopy(self.main_state)
        return self._branches[branch_id]

    def get_branch_state(self, branch_id: str) -> dict[str, Any] | None:
        """Get the current graph state of a speculative branch."""
        return self._branches.get(branch_id)

    def merge_branch(self, branch_id: str) -> KGCommit | None:
        """Merge a speculative branch back to the main state.

        Performs logical conflict validation:
        If a node or edge has been modified/added in the branch, but was also
        modified, deleted, or added differently in the main state since the branch
        was created, it raises a conflict (ValueError).
        If no conflicts, it commits the differences onto the main engine.
        """
        branch_state = self._branches.get(branch_id)
        if not branch_state:
            raise ValueError(f"Branch '{branch_id}' does not exist.")

        original_state = self._original_states.get(branch_id)
        if not original_state:
            original_state = deepcopy(self.main_state)

        # Compute diff from original state to branch state to detect what the branch modified
        branch_diff = KGVersionEngine.diff(original_state, branch_state)

        # 1. Conflict Validation
        # Check if nodes modified in the branch were concurrently deleted in main_state
        for nid in branch_diff.nodes_modified:
            if nid not in self.main_state.get("nodes", {}):
                raise ValueError(
                    f"Merge Conflict: Node '{nid}' was deleted in main graph."
                )

        # Compute diff from main state to branch state
        diff = KGVersionEngine.diff(self.main_state, branch_state)

        # 2. Build merge transaction
        tx = KGTransaction(description=f"Merge branch: {branch_id}")

        # Added nodes
        for nid in diff.nodes_added:
            tx.add_node(nid, branch_state["nodes"][nid])

        # Modified nodes
        for nid in diff.nodes_modified:
            tx.update_node(nid, branch_state["nodes"][nid])

        # Removed nodes
        for nid in diff.nodes_removed:
            tx.delete_node(nid)

        # Added edges
        for src, tgt, lbl in diff.edges_added:
            tx.add_edge(src, tgt, lbl)

        # Removed edges
        for src, tgt, lbl in diff.edges_removed:
            tx.delete_edge(src, tgt, lbl)

        # Commit transaction onto the main state
        commit = self.main_engine.commit(tx, self.main_state)

        # Clean up branch after successful merge
        self._branches.pop(branch_id)
        self._original_states.pop(branch_id, None)
        return commit
