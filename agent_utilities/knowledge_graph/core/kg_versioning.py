#!/usr/bin/python
"""Versioned KG Mutations (CONCEPT:KG-2.0 Enhancement).

Derived from: Evolving Idea Graphs with Learnable Edits-and-Commits
for Multi-Agent Scientific Ideation (arXiv:2605.04922v1, Score 11.2)

Git-like mutation semantics for Knowledge Graph evolution:
- KGTransaction — collects a batch of node/edge mutations
- KGCommit — atomic application with rollback support
- KGDiffEngine — computes structural diffs between graph versions
"""

from __future__ import annotations

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

    @property
    def history(self) -> list[KGCommit]:
        return list(self._commits)
