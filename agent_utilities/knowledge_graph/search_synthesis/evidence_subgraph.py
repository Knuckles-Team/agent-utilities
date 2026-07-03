#!/usr/bin/python
"""Build an evidence-graph workspace from the live epistemic graph.

CONCEPT:KG-2.70 — evidence-subgraph construction

The live adapter for FORT-Searcher's graph-initialization + graph-construction
stages (arXiv:2606.12087, §3.1.1–3.1.2). Where FORT re-mines Wikidata cycles per
run, this checks out a bounded neighborhood of a chosen answer entity from the
*persistent, provenance-rich* epistemic graph and converts incident facts into an
:class:`EvidenceGraph` workspace — reusing existing provenance (``source`` /
``document_id``) so the downstream co-coverage detector is an exact source-sharing
test rather than a heuristic.

Atomic-fact extraction is deterministic. LLM-driven *derived-fact construction*
(FORT Table 2) and *exact-value fuzzing* (Table 5) are the pluggable ``enrich``
seam — supply a callable to enrich the extracted facts; omit it (the default) for
a fully deterministic, CPU-testable extraction.

Concept: evidence-subgraph
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any, Protocol

from .models import EvidenceFact, EvidenceGraph, SearchTask
from .question_formulation import refine
from .shortcut_risks import PriorProbe

# Optional enrichment seam: takes the deterministically-extracted facts and the
# answer node, returns an augmented/fuzzed fact list (FORT derived facts + value
# fuzzing). Default is identity.
EnrichFn = Callable[[list[EvidenceFact], dict[str, Any]], list[EvidenceFact]]


class GraphReader(Protocol):
    """Minimal read surface — satisfied by ``KnowledgeGraph`` (``query``)."""

    def query(self, cypher: str, params: Any = None) -> list[dict[str, Any]]: ...


_NEIGHBOR_QUERY = "MATCH (a {id: $id})-[r]-(b) RETURN a, r, b"


def _names(node: dict[str, Any]) -> tuple[str, ...]:
    out: list[str] = []
    name = node.get("name") or node.get("label") or node.get("title")
    if name:
        out.append(str(name))
    aliases = node.get("aliases")
    if isinstance(aliases, str):
        out.extend(a.strip() for a in aliases.split(",") if a.strip())
    elif isinstance(aliases, list | tuple):
        out.extend(str(a) for a in aliases if a)
    return tuple(dict.fromkeys(out))  # de-dup, keep order


def _humanize(rel: str) -> str:
    return str(rel or "RELATED_TO").replace("_", " ").strip().lower()


def _source_of(edge: dict[str, Any], node: dict[str, Any]) -> str:
    return str(
        edge.get("source")
        or edge.get("document_id")
        or node.get("source")
        or node.get("document_id")
        or f"doc:{node.get('id')}"
    )


def build_evidence_subgraph(
    kg: GraphReader,
    answer_id: str,
    *,
    hops: int = 2,
    fanout: int = 8,
    min_trust: float = 0.0,
    root_popularity: float = 0.0,
    enrich: EnrichFn | None = None,
) -> EvidenceGraph:
    """Check out a bounded neighborhood of ``answer_id`` as an evidence workspace.

    BFS to depth ``hops`` (≤ ``fanout`` neighbors per node), turning each incident
    edge into a clue fact about the answer-bearing path. 1-hop clues are marked
    *required* (the identifying constraints); deeper clues are supporting/redundant
    and may be pruned during refinement. Facts whose source carries a
    ``source_trust`` below ``min_trust`` are dropped (recency/trust ranking reuses
    the same node properties the hybrid retriever reads).
    """
    answer_node: dict[str, Any] = {}
    facts: list[EvidenceFact] = []
    seen = {answer_id}
    frontier = [answer_id]
    endpoint_degree: Counter[str] = Counter()
    raw: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    for depth in range(max(1, hops)):
        nxt: list[str] = []
        for nid in frontier:
            rows = kg.query(_NEIGHBOR_QUERY, {"id": nid}) or []
            for row in rows[:fanout]:
                a = row.get("a") or {}
                b = row.get("b") or {}
                r = row.get("r") or {}
                bid = b.get("id")
                if not bid:
                    continue
                if nid == answer_id and not answer_node:
                    answer_node = a
                trust = float(b.get("source_trust", 1.0) or 0.0)
                if trust < min_trust:
                    continue
                endpoint_degree[bid] += 1
                raw.append((depth, a, r, b))
                if bid not in seen and depth + 1 < hops:
                    seen.add(bid)
                    nxt.append(bid)
        frontier = nxt

    for i, (depth, _a, r, b) in enumerate(raw):
        bid = str(b.get("id"))
        bnames = _names(b)
        neighbor = bnames[0] if bnames else bid
        # Degree within the fetched subgraph approximates standalone selectivity:
        # a neighbor shared by many edges is generic (large candidate pool); a
        # singly-linked neighbor is highly selective.
        deg = endpoint_degree[bid]
        standalone_pool = max(1, deg)
        facts.append(
            EvidenceFact(
                id=f"f{i}",
                clue=f"is {_humanize(r.get('type', ''))} {neighbor}",
                source_document_id=_source_of(r, b),
                standalone_pool=standalone_pool,
                derived=False,
                required=(depth == 0),
                referenced_names=bnames,
                referring_expr="the related entity",
            )
        )

    if enrich is not None:
        facts = enrich(facts, answer_node)

    answer_aliases = _names(answer_node) or (str(answer_id),)
    return EvidenceGraph(
        answer_id=answer_id,
        answer_aliases=answer_aliases,
        facts=facts,
        root_popularity=root_popularity,
    )


def synthesize(
    kg: GraphReader,
    answer_id: str,
    *,
    hops: int = 2,
    fanout: int = 8,
    min_trust: float = 0.0,
    root_popularity: float = 0.0,
    enrich: EnrichFn | None = None,
    max_per_source: int = 1,
    probe: PriorProbe | None = None,
) -> SearchTask:
    """End-to-end: build the evidence workspace, then formulate + adversarially refine.

    Returns a :class:`SearchTask`; inspect ``task.risk_report.clear`` to decide
    whether to keep it (a non-clear task — e.g. prior-bound — should be re-seeded).
    """
    eg = build_evidence_subgraph(
        kg,
        answer_id,
        hops=hops,
        fanout=fanout,
        min_trust=min_trust,
        root_popularity=root_popularity,
        enrich=enrich,
    )
    return refine(eg, max_per_source=max_per_source, probe=probe)
