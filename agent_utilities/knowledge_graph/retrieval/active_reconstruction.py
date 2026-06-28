#!/usr/bin/python
from __future__ import annotations

"""Active memory reconstruction over the knowledge graph.

CONCEPT:KG-2.275 — assimilated from MRAgent: "Memory is Reconstructed, Not
Retrieved: Graph Memory for LLM Agents" (arXiv:2606.06036, Ji/Li/Hooi, NUS).

MRAgent's contribution is *active reconstruction*: instead of one-shot top-k
retrieval, an agent walks a **Cue-Tag-Content** graph in an evidence-conditioned
loop. At each step it

1. activates candidate associative **tags** (relation types) on the current cue
   frontier — ``Cue -> Tag``;
2. expands content **only** along the tags most relevant to the query —
   ``(Cue, Tag) -> Content`` — pruning the combinatorial neighbour blow-up that a
   fixed n-hop expansion would incur; and
3. lets newly-retrieved high-relevance content become the next cue frontier —
   ``Content -> Cue`` (reverse traversal),

so the walk progressively reconstructs a query-relevant subgraph and
self-terminates once fresh evidence stops arriving.

agent-utilities already had the *iterative query-reformulation* loop (KG-2.88,
ADORE) and *single-hop typed-edge* traversal (KG-2.34); what was missing is the
**evidence-conditioned multi-hop graph reconstruction** with tag-mediated frontier
pruning and reverse-cue derivation. This module adds exactly that, graph-native and
**dependency-injected** — callers supply ``neighbor_fn`` and ``score_fn`` — so the
whole policy is unit-testable with no LLM and no backend. Termination reuses the
shared :class:`~.adaptive_stopping.IterativeStopper` (KG-2.87), so this is one more
control structure over the existing retrieval primitives, not a new subsystem.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .adaptive_stopping import IterativeStopper

#: ``node_id -> [(tag, node_dict), ...]`` — the typed neighbours of a cue. ``tag``
#: is the relation type (the associative bridge); ``node_dict`` carries at least an
#: ``id`` plus any text-ish field.
Neighbor = tuple[str, dict[str, Any]]
NeighborFn = Callable[[str], list[Neighbor]]
#: ``(query, text) -> relevance in [0, 1]``. Defaults to :func:`lexical_relevance`.
ScoreFn = Callable[[str, str], float]

_TEXT_KEYS = ("name", "title", "text", "summary", "snippet", "label")
_WORD = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set[str]:
    return set(_WORD.findall((s or "").lower()))


def lexical_relevance(query: str, text: str) -> float:
    """Dependency-free query-coverage relevance in ``[0, 1]``.

    Fraction of the query's content tokens present in ``text`` (recall-oriented, so
    a node mentioning every query term scores ``1.0``). Mirrors MRAgent's text-only
    cue matching — no embeddings, no torch — and stays useful as the always-on
    default; callers with an embedder can inject a vector ``score_fn`` instead.
    """
    q = _tokens(query)
    if not q:
        return 0.0
    return len(q & _tokens(text)) / len(q)


def humanize_tag(tag: str) -> str:
    """Render a relation type (``DEPENDS_ON``) as words (``depends on``) for scoring."""
    return re.sub(r"[_:.-]+", " ", str(tag or "")).strip().lower()


def node_text(node: dict[str, Any]) -> str:
    """First present text-ish field of a node, falling back to its id."""
    for key in _TEXT_KEYS:
        val = node.get(key)
        if val:
            return str(val)
    return str(node.get("id", ""))


@dataclass
class EvidenceNode:
    """One reconstructed content node with how the walk reached it."""

    id: str
    text: str
    score: float
    via_tag: str
    hop: int
    label: str = ""


@dataclass
class ReconstructionStep:
    """Trace of a single hop of the reconstruction walk."""

    hop: int
    frontier: list[str]
    activated_tags: list[str]
    added_ids: list[str]
    pruned: int


@dataclass
class Reconstruction:
    """The reconstructed subgraph plus the trajectory that produced it."""

    query: str
    seeds: list[str]
    evidence: list[EvidenceNode] = field(default_factory=list)
    steps: list[ReconstructionStep] = field(default_factory=list)
    stop_reason: str = ""

    def top(self, k: int) -> list[EvidenceNode]:
        return self.evidence[: max(0, k)]


def reconstruct(
    query: str,
    seed_ids: list[str],
    *,
    neighbor_fn: NeighborFn,
    score_fn: ScoreFn = lexical_relevance,
    max_hops: int = 4,
    tag_top_k: int = 4,
    content_top_k: int = 8,
    relevance_floor: float = 0.05,
) -> Reconstruction:
    """Run the evidence-conditioned Cue-Tag-Content reconstruction walk.

    Args:
        query: The natural-language question driving relevance.
        seed_ids: Resolved cue node ids to start from.
        neighbor_fn: Typed-neighbour expander (see :data:`NeighborFn`).
        score_fn: Query/text relevance in ``[0, 1]`` (default lexical).
        max_hops: Hard cap on reconstruction hops.
        tag_top_k: Keep only this many most-relevant tags per hop (the
            ``Cue -> Tag`` prune that bounds combinatorial expansion).
        content_top_k: Carry at most this many fresh nodes into the next frontier
            (the ``Content -> Cue`` reverse-traversal width).
        relevance_floor: Drop neighbours scoring below this (branch pruning).

    Returns:
        A :class:`Reconstruction` with evidence sorted by best score and the
        per-hop trajectory. Self-terminates via :class:`IterativeStopper`.
    """
    recon = Reconstruction(query=query, seeds=list(seed_ids))
    frontier = [s for s in seed_ids if s]
    if not frontier:
        recon.stop_reason = "no_seed"
        return recon

    stopper = IterativeStopper(max_rounds=max(1, max_hops), min_new_evidence=1)
    seen: set[str] = set(frontier)
    best: dict[str, EvidenceNode] = {}

    for hop in range(1, max_hops + 1):
        # Gather typed neighbours of every cue on the frontier.
        candidates: list[tuple[str, dict[str, Any], float]] = []
        tag_relevance: dict[str, float] = {}
        for cue in frontier:
            for tag, node in neighbor_fn(cue) or []:
                if not node.get("id"):
                    continue
                content_score = score_fn(query, node_text(node))
                # A tag's relevance is the query-relevance of its label OR of the
                # best content it leads to (associative bridge salience).
                tag_score = max(score_fn(query, humanize_tag(tag)), content_score)
                if tag_score > tag_relevance.get(tag, -1.0):
                    tag_relevance[tag] = tag_score
                candidates.append((tag, node, content_score))

        if not candidates:
            recon.stop_reason = "frontier_exhausted"
            break

        # Cue -> Tag: keep only the most relevant tags this hop (prune the blow-up).
        selected_tags = {
            t
            for t, _ in sorted(
                tag_relevance.items(), key=lambda kv: kv[1], reverse=True
            )[: max(1, tag_top_k)]
        }

        # (Cue, Tag) -> Content: expand only the selected tags; prune weak content.
        added: list[tuple[str, float]] = []
        pruned = 0
        round_evidence: list[str] = []
        for tag, node, content_score in candidates:
            if tag not in selected_tags:
                continue
            if content_score < relevance_floor:
                pruned += 1
                continue
            nid = str(node["id"])
            round_evidence.append(nid)
            prev = best.get(nid)
            if prev is None or content_score > prev.score:
                best[nid] = EvidenceNode(
                    id=nid,
                    text=node_text(node),
                    score=content_score,
                    via_tag=tag,
                    hop=hop,
                    label=str(node.get("label") or ""),
                )
            if nid not in seen:
                seen.add(nid)
                added.append((nid, content_score))

        # Content -> Cue: the best fresh content becomes the next cue frontier.
        next_frontier = [
            nid
            for nid, _ in sorted(added, key=lambda t: t[1], reverse=True)[
                : max(1, content_top_k)
            ]
        ]
        recon.steps.append(
            ReconstructionStep(
                hop=hop,
                frontier=list(frontier),
                activated_tags=sorted(selected_tags),
                added_ids=[nid for nid, _ in added],
                pruned=pruned,
            )
        )

        # Self-termination: answer = current best evidence ids (TASR repeat rule)
        # plus coverage saturation when no fresh nodes arrive.
        answer = " ".join(
            e.id for e in sorted(best.values(), key=lambda e: e.score, reverse=True)
        )
        decision = stopper.update(answer=answer, evidence_ids=round_evidence)
        if decision.stop:
            recon.stop_reason = decision.reason
            frontier = next_frontier
            break
        if not next_frontier:
            recon.stop_reason = "frontier_exhausted"
            break
        frontier = next_frontier
    else:
        recon.stop_reason = "max_hops"

    recon.evidence = sorted(best.values(), key=lambda e: e.score, reverse=True)
    return recon


# --------------------------------------------------------------------------- #
# Engine-backed adapters (the live, default neighbour/seed providers).         #
# --------------------------------------------------------------------------- #


def engine_neighbor_fn(engine: Any, *, fanout: int = 64) -> NeighborFn:
    """A :data:`NeighborFn` reading typed neighbours from the live KG engine.

    Best-effort: degrades to ``[]`` on any backend error (via ``read_rows``), so a
    reconstruction over a cold/degraded backend simply yields no evidence.
    """
    from .context_plane import read_rows

    cypher = (
        "MATCH (s {id: $id})-[r]-(t) "
        "RETURN type(r) AS tag, t.id AS id, t.name AS name, t.title AS title, "
        "t.text AS text, t.summary AS summary, labels(t)[0] AS label "
        f"LIMIT {int(fanout)}"
    )

    def _fn(node_id: str) -> list[Neighbor]:
        out: list[Neighbor] = []
        for r in read_rows(engine, cypher, {"id": node_id}):
            nid = r.get("id")
            if not nid:
                continue
            node = {
                "id": nid,
                "name": r.get("name"),
                "title": r.get("title"),
                "text": r.get("text"),
                "summary": r.get("summary"),
                "label": r.get("label"),
            }
            out.append((str(r.get("tag") or "RELATED"), node))
        return out

    return _fn


def resolve_seeds(engine: Any, query: str, *, top_k: int = 3) -> list[dict[str, Any]]:
    """Resolve a query to seed cue nodes (keyword search, then name-contains)."""
    sk = getattr(engine, "_search_keyword", None)
    if callable(sk):
        try:
            hits = [h for h in (sk(query, top_k=top_k) or []) if h.get("id")]
            if hits:
                return hits
        except Exception:  # pragma: no cover - defensive
            pass
    from .context_plane import read_rows

    q = (query or "").strip()[:80]
    rows = read_rows(
        engine,
        "MATCH (n) WHERE toLower(toString(n.name)) CONTAINS toLower($q) "
        f"RETURN n.id AS id, n.name AS name LIMIT {max(1, int(top_k))}",
        {"q": q},
    )
    return [r for r in rows if r.get("id")]


__all__ = [
    "Neighbor",
    "NeighborFn",
    "ScoreFn",
    "lexical_relevance",
    "humanize_tag",
    "node_text",
    "EvidenceNode",
    "ReconstructionStep",
    "Reconstruction",
    "reconstruct",
    "engine_neighbor_fn",
    "resolve_seeds",
]
