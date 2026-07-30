#!/usr/bin/python
from __future__ import annotations

"""Shared reasoning-state substrate (CONCEPT:AU-ORCH.planning.reasoning-graph-topologies).

Every topology in this package (CoT, self-consistent CoT, ToT, GoT, ReAct, RAP)
reads and writes the SAME :class:`ReasoningState` — the paper's central claim is
that these algorithms differ only in their node contracts and edge/routing
functions, not in the state they need. :class:`ThoughtNode` is a DAG node (a
node may carry more than one ``parent_id``, which is exactly GoT's
merge/aggregate provenance — no separate structure is needed for it); the
state additionally carries the frontier/visited-set ToT needs, the MCTS
visit/value stats RAP backpropagates, the candidate distribution
self-consistent CoT votes over, and the tool-call log ReAct needs.

Truthfulness / privacy contract (``AGENTS.md``): :meth:`ReasoningState.
append_rationale` only ever appends a caller-supplied SUMMARY string. Raw,
private per-step model reasoning is never captured by this module and must
never be persisted verbatim by a caller either — only append the summary a
topology step function chooses to keep.
"""

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class NodeKind(StrEnum):
    """The node-contract vocabulary shared by every topology in this package."""

    ROOT = "root"
    GENERATE = "generate"
    TRANSFORM = "transform"
    AGGREGATE = "aggregate"
    REFINE = "refine"
    VOTE = "vote"
    ACT = "act"
    OBSERVE = "observe"


@dataclass
class ThoughtNode:
    """One node in the shared thought DAG.

    ``parent_ids`` with more than one entry is GoT's merge/aggregate
    provenance record: which prior thoughts were fused into this one.
    ``visits``/``total_reward`` are RAP's MCTS backpropagation targets;
    ``value`` is ToT's per-node score used for frontier ranking/pruning.
    """

    node_id: str
    kind: NodeKind
    content: str
    parent_ids: list[str] = field(default_factory=list)
    depth: int = 0
    value: float | None = None
    visits: int = 0
    total_reward: float = 0.0
    is_terminal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def uct(self, parent_visits: int, c: float) -> float:
        """UCT score: exploitation (mean reward) + ``c``·exploration bonus.

        Unvisited nodes score ``+inf`` so they are always tried first. Shared
        by :mod:`.tot` (frontier ranking) and :mod:`.rap` (MCTS selection).
        """
        if self.visits == 0:
            return math.inf
        exploitation = self.total_reward / self.visits
        exploration = c * math.sqrt(math.log(max(parent_visits, 1)) / self.visits)
        return exploitation + exploration

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "content": self.content,
            "parent_ids": list(self.parent_ids),
            "depth": self.depth,
            "value": self.value,
            "visits": self.visits,
            "total_reward": self.total_reward,
            "is_terminal": self.is_terminal,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThoughtNode:
        return cls(
            node_id=data["node_id"],
            kind=NodeKind(data["kind"]),
            content=data["content"],
            parent_ids=list(data.get("parent_ids", [])),
            depth=data.get("depth", 0),
            value=data.get("value"),
            visits=data.get("visits", 0),
            total_reward=data.get("total_reward", 0.0),
            is_terminal=data.get("is_terminal", False),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class CandidateVote:
    """One final-answer candidate in a self-consistency vote (candidate
    distribution + cost + confidence — the required state semantics for
    self-consistent CoT)."""

    candidate_id: str
    answer: str
    votes: int = 0
    cost_tokens: int = 0
    confidence: float = 0.0


@dataclass
class ToolCallRecord:
    """One ReAct thought→action→observation step's tool result + retry count."""

    step: int
    action: str
    observation: str
    success: bool
    retry: int = 0


@dataclass
class ReasoningState:
    """The one shared state store every reasoning-graph topology reads/writes."""

    topology: str
    nodes: dict[str, ThoughtNode] = field(default_factory=dict)
    #: Open list for BFS/DFS frontier expansion (ToT).
    frontier: list[str] = field(default_factory=list)
    #: Nodes already expanded/selected (ToT/RAP).
    visited: set[str] = field(default_factory=set)
    #: Self-consistency candidate distribution, keyed by candidate_id.
    candidates: dict[str, CandidateVote] = field(default_factory=dict)
    #: ReAct tool-call log, in step order.
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    #: CoT append-only rationale SUMMARIES — never raw private reasoning text.
    rationale_summary: list[str] = field(default_factory=list)
    step: int = 0

    def add_node(self, node: ThoughtNode) -> None:
        self.nodes[node.node_id] = node

    def append_rationale(self, summary: str) -> None:
        """Append a rationale SUMMARY. Never pass raw private model reasoning here."""
        self.rationale_summary.append(summary)

    def children_of(self, node_id: str) -> list[str]:
        """All node ids whose ``parent_ids`` include ``node_id`` (insertion order)."""
        return [n.node_id for n in self.nodes.values() if node_id in n.parent_ids]

    def path_to_root(self, node_id: str) -> list[str]:
        """Walk the PRIMARY edge (first ``parent_id``) from ``node_id`` to the root."""
        path = [node_id]
        seen = {node_id}
        current = self.nodes.get(node_id)
        while current is not None and current.parent_ids:
            parent_id = current.parent_ids[0]
            if parent_id in seen:  # defensive: never loop on a malformed graph
                break
            path.append(parent_id)
            seen.add(parent_id)
            current = self.nodes.get(parent_id)
        return path

    def to_memento(self) -> dict[str, Any]:
        """Serializable checkpoint of the whole state — the topology-resource's
        checkpoint semantics: a run halted by budget exhaustion resumes from
        exactly this snapshot rather than losing partial progress."""
        return {
            "topology": self.topology,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "frontier": list(self.frontier),
            "visited": list(self.visited),
            "candidates": {cid: vars(c) | {} for cid, c in self.candidates.items()},
            "tool_calls": [vars(t) for t in self.tool_calls],
            "rationale_summary": list(self.rationale_summary),
            "step": self.step,
        }

    @classmethod
    def from_memento(cls, data: dict[str, Any]) -> ReasoningState:
        state = cls(topology=data["topology"])
        state.nodes = {
            nid: ThoughtNode.from_dict(nd) for nid, nd in data.get("nodes", {}).items()
        }
        state.frontier = list(data.get("frontier", []))
        state.visited = set(data.get("visited", []))
        state.candidates = {
            cid: CandidateVote(**cv) for cid, cv in data.get("candidates", {}).items()
        }
        state.tool_calls = [ToolCallRecord(**tc) for tc in data.get("tool_calls", [])]
        state.rationale_summary = list(data.get("rationale_summary", []))
        state.step = data.get("step", 0)
        return state
