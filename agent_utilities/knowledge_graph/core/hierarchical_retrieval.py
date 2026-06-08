#!/usr/bin/python
from __future__ import annotations

"""Hierarchical (global→local) community retrieval over the KG.

CONCEPT:KG-2.5 — Topological Analysis (hierarchical GraphRAG retrieval)

Distilled from Deep GraphRAG (`.specify/specs/research-evolution-20260606/` plan
b2-04): instead of flat per-node similarity, retrieve in stages —

1. **global**: rank communities by aggregate query relevance of their members;
2. **community**: drill into the top-k communities;
3. **local**: rank entities *within* those, with a **parent-context** boost so a
   node in a highly-relevant community ranks above an equally-similar node in an
   irrelevant one (Deep GraphRAG F3 context-aware ranking).

Pure Python, deterministic (lexical relevance by default; an embedder can be
injected). Communities come from the existing flat detector unless supplied, so
this composes with `TopologicalAnalysisEngine` rather than replacing it.

Concept: hierarchical-retrieval
"""

import re
from typing import Any

from pydantic import BaseModel, Field

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _overlap(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    dt = _tokens(text)
    return len(query_tokens & dt) / len(query_tokens)


class EntityHit(BaseModel):
    """A retrieved entity with its hierarchical provenance + score components."""

    id: str
    score: float = Field(ge=0.0, le=1.0)
    entity_score: float = Field(ge=0.0, le=1.0)
    community_index: int
    community_score: float = Field(ge=0.0, le=1.0)


class HierarchicalResult(BaseModel):
    """Result of a hierarchical retrieval."""

    query: str
    hits: list[EntityHit] = Field(default_factory=list)
    communities_ranked: int = 0
    communities_searched: int = 0


class HierarchicalCommunityRetriever:
    """Global→local community drill-down over a graph-compute engine.

    Args:
        graph: A graph with ``_get_node_properties(node_id)`` (e.g. GraphComputeEngine).
        embedder: Optional ``embedder.score(query, text) -> float`` for relevance
            (defaults to deterministic lexical overlap).
        parent_weight: Blend weight for the parent community's score in the final
            entity rank (the context-aware F3 boost).
    """

    def __init__(
        self, graph: Any, *, embedder: Any = None, parent_weight: float = 0.3
    ) -> None:
        self.graph = graph
        self.embedder = embedder
        self.parent_weight = max(0.0, min(1.0, parent_weight))

    def _node_text(self, node_id: str) -> str:
        try:
            props = self.graph._get_node_properties(node_id) or {}
        except Exception:
            return node_id
        parts = [
            str(props.get(k, "")) for k in ("name", "content", "summary", "description")
        ]
        body = " ".join(p for p in parts if p)
        return body or node_id

    def _relevance(self, query: str, q_tokens: set[str], text: str) -> float:
        if self.embedder is not None:
            return float(self.embedder.score(query, text))
        return _overlap(q_tokens, text)

    def retrieve(
        self,
        query: str,
        *,
        communities: list[set[str]] | None = None,
        top_communities: int = 2,
        top_entities: int = 10,
    ) -> HierarchicalResult:
        """Run the 3-stage global→local retrieval."""
        if communities is None:
            try:
                from .topological_partition import detect_communities

                communities = detect_communities(self.graph)
            except Exception:
                communities = []
        result = HierarchicalResult(query=query, communities_ranked=len(communities))
        if not communities:
            return result

        q_tokens = _tokens(query)

        # Stage 1 (global): score each community by mean member relevance.
        scored: list[tuple[int, set[str], float]] = []
        for idx, comm in enumerate(communities):
            members = list(comm)
            if not members:
                continue
            cscore = sum(
                self._relevance(query, q_tokens, self._node_text(m)) for m in members
            ) / len(members)
            scored.append((idx, comm, cscore))
        scored.sort(key=lambda t: t[2], reverse=True)

        # Stage 2 (community): take the top-k communities.
        top = scored[:top_communities]
        result.communities_searched = len(top)

        # Stage 3 (local): rank entities within, with parent-context boost.
        hits: list[EntityHit] = []
        for idx, comm, cscore in top:
            for nid in comm:
                ent = self._relevance(query, q_tokens, self._node_text(nid))
                final = (1.0 - self.parent_weight) * ent + self.parent_weight * cscore
                hits.append(
                    EntityHit(
                        id=nid,
                        score=round(min(1.0, final), 6),
                        entity_score=round(min(1.0, ent), 6),
                        community_index=idx,
                        community_score=round(min(1.0, cscore), 6),
                    )
                )
        hits.sort(key=lambda h: h.score, reverse=True)
        result.hits = hits[:top_entities]
        return result
