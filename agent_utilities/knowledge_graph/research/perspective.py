#!/usr/bin/python
from __future__ import annotations

"""Multi-perspective inquiry — STORM made native to the research fan-out.

CONCEPT:AU-KG.research.perspectival-inquiry — Perspectival inquiry engine
CONCEPT:AU-KG.research.contradiction-agreement-blind-spot — Contradiction / agreement / blind-spot as first-class KG structures
CONCEPT:AU-KG.research.peer-review-self-critique — Peer-review / self-critique closing the loop

Stanford's STORM (NAACL 2024) showed that researching a topic from *several distinct
expert lenses* — each asking different questions — then mapping where they disagree,
yields markedly more organized and broader coverage than a single prompt. We make that
the **default behaviour of the research fan-out**, not a separate tool: instead of one
semantic probe of the topic name (:func:`search.acquire_for_topic`), the loop fans the
*same* probe across questions asked from multiple perspectives, then derives a
contradiction map and a self-critique — grounded in the KG, no LLM required.

Four phases, all deterministic over the graph (an optional ``llm_fn`` only enriches
question phrasing):

1. **Perspectives** — distinct expert lenses (ontology-flavoured by the topic's KG
   neighbourhood, falling back to a canonical set);
2. **Fan-out** — each lens asks questions; each question is one bounded ``acquire`` over
   the KG, yielding that lens's source set;
3. **Contradiction map** — sources ≥2 lenses share are *agreements* ("likely true");
   lenses with disjoint source sets are *divergences*; KG-neighbour types no lens
   touched are the *blind spot*;
4. **Peer review** — per-source confidence from corroboration, the dominant/missing
   lens (bias check), and a *frontier question* that is submitted back as the next
   research loop (the self-critique STORM lacked).

The result materializes as typed KG nodes (Perspective / Agreement / Contradiction /
BlindSpot / PeerReview) so the inquiry itself is graph-queryable.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Canonical expert lenses (STORM's five), used when the topic's KG neighbourhood is too
# sparse to derive domain-specific ones. Each is a (lens, what-it-uniquely-sees) pair.
CANONICAL_LENSES: list[tuple[str, str]] = [
    ("practitioner", "the practical realities of working with this daily"),
    ("academic", "what the peer-reviewed evidence actually says"),
    ("skeptic", "the strongest counter-argument and ignored evidence"),
    ("economist", "who profits and the incentives shaping the narrative"),
    ("historian", "the historical parallels and how they played out"),
]

# Bound the added cost over the single-lens path: questions per lens.
_QUESTIONS_PER_LENS = 2


@dataclass
class Perspective:
    """One expert lens and the KG sources its questions surfaced."""

    id: str
    lens: str
    rationale: str
    source_node_ids: list[str] = field(default_factory=list)


@dataclass
class ContradictionMap:
    """Where the lenses agree, diverge, and what none of them addressed."""

    agreements: list[str] = field(default_factory=list)
    divergences: list[tuple[str, str]] = field(default_factory=list)
    blind_spot: list[str] = field(default_factory=list)


@dataclass
class PeerReview:
    """The self-critique pass (the known STORM gap)."""

    confidence: dict[str, int] = field(default_factory=dict)
    weakest_link: str | None = None
    dominant_lens: str | None = None
    missing_perspective: str | None = None
    frontier_question: str | None = None


@dataclass
class PerspectiveInquiry:
    """A full multi-perspective inquiry over one topic."""

    topic_id: str
    topic_name: str
    perspectives: list[Perspective] = field(default_factory=list)
    contradiction_map: ContradictionMap = field(default_factory=ContradictionMap)
    peer_review: PeerReview = field(default_factory=PeerReview)

    def all_source_ids(self) -> list[str]:
        """Union of every source any perspective found (preserves discovery order)."""
        out: list[str] = []
        seen: set[str] = set()
        for p in self.perspectives:
            for sid in p.source_node_ids:
                if sid not in seen:
                    seen.add(sid)
                    out.append(sid)
        return out

    def to_entities(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Render the inquiry as typed KG entities + relationships (CONCEPT:AU-KG.research.contradiction-agreement-blind-spot)."""
        root = f"inquiry:{self.topic_id}"
        entities: list[dict[str, Any]] = [
            {
                "id": root,
                "type": "research_inquiry",
                "name": f"Perspectival inquiry: {self.topic_name}",
                "domain": "research",
                "addressesTopic": self.topic_id,
            }
        ]
        rels: list[dict[str, Any]] = []
        for p in self.perspectives:
            entities.append(
                {
                    "id": p.id,
                    "type": "perspective",
                    "name": p.lens,
                    "rationale": p.rationale,
                    "sourceCount": len(p.source_node_ids),
                    "domain": "research",
                }
            )
            rels.append(
                {
                    "source": p.id,
                    "target": root,
                    "type": "part_of",
                    "domain": "research",
                }
            )
            for sid in p.source_node_ids:
                rels.append(
                    {
                        "source": p.id,
                        "target": sid,
                        "type": "asks_from",
                        "domain": "research",
                    }
                )
        cm = self.contradiction_map
        if cm.agreements:
            agree_id = f"{root}:agreement"
            entities.append(
                {
                    "id": agree_id,
                    "type": "agreement",
                    "name": f"{len(cm.agreements)} corroborated sources",
                    "domain": "research",
                }
            )
            rels.append(
                {
                    "source": agree_id,
                    "target": root,
                    "type": "part_of",
                    "domain": "research",
                }
            )
            for sid in cm.agreements:
                rels.append(
                    {
                        "source": agree_id,
                        "target": sid,
                        "type": "agrees_with",
                        "domain": "research",
                    }
                )
        if cm.blind_spot:
            blind_id = f"{root}:blind_spot"
            entities.append(
                {
                    "id": blind_id,
                    "type": "blind_spot",
                    "name": "Uncovered: " + ", ".join(cm.blind_spot[:6]),
                    "domain": "research",
                }
            )
            rels.append(
                {
                    "source": blind_id,
                    "target": root,
                    "type": "part_of",
                    "domain": "research",
                }
            )
        for a, b in cm.divergences:
            entities.append(
                {
                    "id": f"{root}:divergence:{a}:{b}",
                    "type": "contradiction",
                    "name": f"{a} vs {b}: non-overlapping evidence",
                    "domain": "research",
                }
            )
        pr = self.peer_review
        review_id = f"{root}:peer_review"
        entities.append(
            {
                "id": review_id,
                "type": "peer_review",
                "name": "Self-critique",
                "dominantLens": pr.dominant_lens or "",
                "missingPerspective": pr.missing_perspective or "",
                "frontierQuestion": pr.frontier_question or "",
                "domain": "research",
            }
        )
        rels.append(
            {
                "source": review_id,
                "target": root,
                "type": "reviews",
                "domain": "research",
            }
        )
        return entities, rels


class PerspectiveEngine:
    """Run multi-perspective inquiry over a topic, grounded in the KG.

    Dependency-injectable (``engine`` for KG grounding/materialization, optional
    ``llm_fn`` to enrich question phrasing); degrades to a safe empty inquiry and never
    raises into the research loop.
    """

    def __init__(
        self, engine: Any = None, *, llm_fn: Callable[[str], str] | None = None
    ) -> None:
        self.engine = engine
        self.llm_fn = llm_fn

    # -- phase 1: perspectives ------------------------------------------------ #
    def derive_perspectives(self, topic_id: str, topic_name: str) -> list[Perspective]:
        """Distinct lenses, ontology-flavoured by the topic's KG neighbourhood.

        The canonical five lenses are always available; when the topic's neighbours in
        the graph reveal domains (a ``service``/``capability``/``regulation``/…), their
        rationale is annotated so the lens is grounded in what the graph actually holds.
        """
        neighbours = self._neighbour_types(topic_id)
        hint = (
            f" (graph neighbours: {', '.join(sorted(neighbours)[:5])})"
            if neighbours
            else ""
        )
        out: list[Perspective] = []
        for lens, sees in CANONICAL_LENSES:
            out.append(
                Perspective(
                    id=f"inquiry:{topic_id}:lens:{lens}",
                    lens=lens,
                    rationale=f"Sees {sees}{hint}",
                )
            )
        return out

    def _neighbour_types(self, topic_id: str) -> set[str]:
        engine = self.engine
        if engine is None or not topic_id:
            return set()
        try:
            rows = engine.query_cypher(
                "MATCH (t {id: $id})--(n) RETURN DISTINCT labels(n) AS labels LIMIT 25",
                {"id": topic_id},
            )
        except Exception:  # noqa: BLE001 — neighbourhood is best-effort
            return set()
        types: set[str] = set()
        for r in rows or []:
            labels = r.get("labels") if isinstance(r, dict) else None
            for lab in labels or []:
                if lab not in ("Concept", "Resource"):
                    types.add(str(lab))
        return types

    def questions_for(self, perspective: Perspective, topic_name: str) -> list[str]:
        """The distinct questions this lens asks (templated; LLM-enriched if available)."""
        base = [
            f"{topic_name}: what does the {perspective.lens} know that other views miss?",
            f"{topic_name}: the strongest evidence from the {perspective.lens} view",
        ][:_QUESTIONS_PER_LENS]
        if self.llm_fn is not None:
            try:
                refined = self.llm_fn(
                    f"As a {perspective.lens} researching '{topic_name}', list "
                    f"{_QUESTIONS_PER_LENS} short distinct questions, one per line."
                )
                lines = [
                    q.strip("-• ").strip()
                    for q in str(refined).splitlines()
                    if q.strip()
                ]
                if lines:
                    return lines[:_QUESTIONS_PER_LENS]
            except Exception:  # noqa: BLE001 — fall back to templated questions
                logger.debug("llm question refinement failed", exc_info=True)
        return base

    # -- phases 2-4: fan-out, contradiction map, peer review ------------------ #
    def inquire(
        self, topic: dict[str, Any], acquire: Callable[[str], list[str]]
    ) -> PerspectiveInquiry:
        """Run the four-phase inquiry; ``acquire(question)`` answers one question with
        KG source ids (reuses the single-lens probe per question)."""
        topic_id = str(topic.get("id") or "")
        topic_name = str(topic.get("name") or topic_id).strip()
        inquiry = PerspectiveInquiry(topic_id=topic_id, topic_name=topic_name)
        if not topic_name:
            return inquiry

        perspectives = self.derive_perspectives(topic_id, topic_name)
        for p in perspectives:
            found: list[str] = []
            for question in self.questions_for(p, topic_name):
                try:
                    for sid in acquire(question) or []:
                        if sid not in found and sid != topic_id:
                            found.append(sid)
                except Exception:  # noqa: BLE001 — one bad probe must not abort
                    logger.debug("acquire failed for %r", question, exc_info=True)
            p.source_node_ids = found
        inquiry.perspectives = perspectives
        inquiry.contradiction_map = self._contradiction_map(perspectives, topic_id)
        inquiry.peer_review = self._peer_review(
            perspectives, inquiry.contradiction_map, topic_name
        )
        return inquiry

    def _contradiction_map(
        self, perspectives: list[Perspective], topic_id: str
    ) -> ContradictionMap:
        counts: dict[str, int] = {}
        for p in perspectives:
            for sid in set(p.source_node_ids):
                counts[sid] = counts.get(sid, 0) + 1
        agreements = sorted([s for s, c in counts.items() if c >= 2])
        # Divergences: lens pairs that surfaced entirely non-overlapping evidence.
        divergences: list[tuple[str, str]] = []
        non_empty = [p for p in perspectives if p.source_node_ids]
        for i in range(len(non_empty)):
            for j in range(i + 1, len(non_empty)):
                a, b = non_empty[i], non_empty[j]
                if not (set(a.source_node_ids) & set(b.source_node_ids)):
                    divergences.append((a.lens, b.lens))
        # Blind spot: KG-neighbour types of the topic that none of the found sources cover.
        covered = self._covered_types(counts)
        blind_spot = sorted(self._neighbour_types(topic_id) - covered)
        return ContradictionMap(
            agreements=agreements, divergences=divergences, blind_spot=blind_spot
        )

    def _covered_types(self, source_counts: dict[str, int]) -> set[str]:
        engine = self.engine
        if engine is None or not source_counts:
            return set()
        try:
            rows = engine.query_cypher(
                "MATCH (n) WHERE n.id IN $ids RETURN DISTINCT labels(n) AS labels",
                {"ids": list(source_counts)},
            )
        except Exception:  # noqa: BLE001
            return set()
        out: set[str] = set()
        for r in rows or []:
            for lab in (r.get("labels") if isinstance(r, dict) else None) or []:
                out.add(str(lab))
        return out

    def _peer_review(
        self,
        perspectives: list[Perspective],
        cm: ContradictionMap,
        topic_name: str,
    ) -> PeerReview:
        counts: dict[str, int] = {}
        for p in perspectives:
            for sid in set(p.source_node_ids):
                counts[sid] = counts.get(sid, 0) + 1
        # Confidence 1-10: corroboration across lenses is the reliability signal.
        confidence = {sid: min(10, 2 + 2 * c) for sid, c in counts.items()}
        singletons = sorted([s for s, c in counts.items() if c == 1])
        with_sources = [p for p in perspectives if p.source_node_ids]
        dominant = max(perspectives, key=lambda p: len(p.source_node_ids), default=None)
        missing = next((p for p in perspectives if not p.source_node_ids), None)
        frontier = None
        if cm.blind_spot:
            frontier = f"{topic_name}: what is the role of {cm.blind_spot[0]}?"
        elif missing is not None:
            frontier = f"{topic_name} from the {missing.lens} perspective"
        return PeerReview(
            confidence=confidence,
            weakest_link=singletons[0] if singletons else None,
            dominant_lens=dominant.lens
            if dominant and dominant.source_node_ids
            else None,
            missing_perspective=missing.lens if missing else None,
            frontier_question=frontier
            if (frontier and len(with_sources) >= 2)
            else None,
        )

    # -- materialize ---------------------------------------------------------- #
    def materialize(self, inquiry: PerspectiveInquiry) -> dict[str, Any]:
        """Persist the inquiry as typed KG nodes and submit its frontier question as
        the next research loop (closing the STORM self-critique loop, CONCEPT:AU-KG.research.peer-review-self-critique)."""
        engine = self.engine
        if engine is None:
            return {"materialized": False}
        entities, rels = inquiry.to_entities()
        try:
            engine.ingest_external_batch("research", entities, rels)
        except Exception as exc:  # noqa: BLE001 — materialize is best-effort
            logger.debug("perspective materialize failed: %s", exc)
            return {"materialized": False, "error": str(exc)}
        frontier = inquiry.peer_review.frontier_question
        submitted = None
        if frontier:
            try:
                from .loops import submit_loop

                loop = submit_loop(
                    engine, frontier, kind="research", source="peer-review"
                )
                submitted = loop.get("id") if isinstance(loop, dict) else None
            except Exception:  # noqa: BLE001
                logger.debug("frontier loop submit failed", exc_info=True)
        return {
            "materialized": True,
            "entities": len(entities),
            "frontier_loop": submitted,
        }
