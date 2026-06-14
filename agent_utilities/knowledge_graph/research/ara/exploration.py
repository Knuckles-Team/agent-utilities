#!/usr/bin/python
from __future__ import annotations

"""Exploration graph — the ARA ``/trace`` layer producer (CONCEPT:KG-2.80).

The paper's high-signal forensic layer is the exploration DAG: not just what worked, but
the questions asked, the decisions taken, and crucially the **dead-ends** and **pivots**.
The ``exploration_node`` type + ``pivoted_from`` / ``reached_dead_end`` edges exist
(A1); this is their producer. A Loop emits trace nodes each cycle:

- **dead_end**  ← failure clusters (AHE-3.18) — an approach that was tried and abandoned;
- **pivot**     ← ConceptMatcher rejections (KG-2.75) — a candidate direction reconsidered;
- question / decision / experiment / result ← the ordinary cycle deliberation.

Because the DAG is promoted into the one ontology, reasoning extrapolates pivot/dead-end
*patterns across artifacts* (e.g. the same dead-end recurring), which the paper cannot do
until a corpus accumulates. Inputs are plain values so the producer is unit-testable
without a live Loop; callers feed it failure clusters / matcher rejects directly.

Concept: exploration-graph
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from .artifact import ExplorationKind, ExplorationNode

logger = logging.getLogger(__name__)


class ResearchTrajectory(BaseModel):
    """An ordered exploration DAG under one root question (CONCEPT:KG-2.80)."""

    root_id: str
    nodes: list[ExplorationNode] = Field(default_factory=list)

    def add(
        self,
        node_id: str,
        kind: ExplorationKind,
        text: str,
        *,
        parent_id: str = "",
    ) -> ExplorationNode:
        """Append a node, defaulting its parent to the trajectory root."""
        node = ExplorationNode(
            id=node_id,
            kind=kind,
            text=text,
            parent_id=parent_id or self.root_id,
        )
        self.nodes.append(node)
        return node

    def dead_ends(self) -> list[ExplorationNode]:
        return [n for n in self.nodes if n.kind == "dead_end"]

    def pivots(self) -> list[ExplorationNode]:
        return [n for n in self.nodes if n.kind == "pivot"]


class ExplorationGraphBuilder:
    """Turn one Loop cycle's signals into a :class:`ResearchTrajectory`."""

    def __init__(self, article_id: str) -> None:
        self._aid = article_id
        self._n = 0

    def _nid(self, kind: str) -> str:
        self._n += 1
        return f"exploration_node:{self._aid}:{kind}:{self._n}"

    def build(
        self,
        question: str,
        *,
        decisions: list[str] | None = None,
        experiments: list[str] | None = None,
        results: list[str] | None = None,
        failure_clusters: list[Any] | None = None,
        matcher_rejects: list[Any] | None = None,
    ) -> ResearchTrajectory:
        """Assemble the trace DAG. ``failure_clusters`` become dead-ends and
        ``matcher_rejects`` become pivots — the two forensic markers."""
        root = ExplorationNode(id=self._nid("question"), kind="question", text=question)
        traj = ResearchTrajectory(root_id=root.id, nodes=[root])

        for text in decisions or []:
            traj.add(self._nid("decision"), "decision", text)
        for text in experiments or []:
            traj.add(self._nid("experiment"), "experiment", text)
        for text in results or []:
            traj.add(self._nid("result"), "result", text)
        for cluster in failure_clusters or []:
            traj.add(self._nid("dead_end"), "dead_end", _text_of(cluster))
        for reject in matcher_rejects or []:
            traj.add(self._nid("pivot"), "pivot", _text_of(reject))
        return traj

    @staticmethod
    def attach(artifact: Any, trajectory: ResearchTrajectory) -> int:
        """Attach a trajectory's nodes to an artifact's ``/trace`` layer."""
        artifact.exploration.extend(trajectory.nodes)
        return len(trajectory.nodes)


def _text_of(obj: Any) -> str:
    """Best-effort human text from a failure cluster / matcher reject / str."""
    if isinstance(obj, str):
        return obj
    for attr in ("summary", "title", "description", "name", "reason", "text"):
        val = getattr(obj, attr, None) or (
            obj.get(attr) if isinstance(obj, dict) else None
        )
        if val:
            return str(val)
    return str(obj)


__all__ = ["ResearchTrajectory", "ExplorationGraphBuilder"]
