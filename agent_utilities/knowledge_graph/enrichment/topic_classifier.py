#!/usr/bin/python
from __future__ import annotations

"""Intelligent subject/topic classification → topology (CONCEPT:AU-KG.enrichment.topic-classification-topology).

Classifies every ingested document into the canonical **WorldView** subject-domain
taxonomy (``agent_utilities/knowledge_graph/ontology_worldview.ttl``,
CONCEPT:AU-KG.enrichment.worldview-subject-ontology) — a curated, reasonably-MECE
upper ontology of world-knowledge domains (Science, Technology & Computing,
Engineering, Mathematics, Health & Medicine, Society & Politics, Economy &
Business, Law & Governance, Environment & Earth, Arts & Culture, Humanities &
Philosophy, Sports & Recreation) — plus a curated first layer of sub-domains under
each. Every document lands on the MOST SPECIFIC node it can justify; deeper,
context-derived sub-topics are minted as ``skos:narrower``/``NARROWER`` children of
the nearest existing node, so no topic ever floats — the whole ingested corpus
organizes under the ONE worldview, giving the rich cross-topic topology the
document-topology program wants.

Uses the fleet LLM (``create_model``/pydantic-ai ``Agent``, reasoning OFF by
default — a simple structured classification task doesn't need it) with a
deterministic keyword-overlap fallback when no LLM is reachable. Best-effort
throughout: a classification failure never breaks ingestion.

Public entry point: :func:`classify_and_link_topics` — the function the unified
ingestion enrichment seam (``ingestion/engine.py::_enrich_text``) and the
``graph_ingest action=classify_topics`` MCP/REST action both call.
"""

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# The WorldView seed taxonomy — mirrors ontology_worldview.ttl's individuals
# 1:1 (same slugs) so a runtime ``:Topic`` node id always resolves back to the
# canonical OWL individual it hangs from. Extend BOTH the ``.ttl`` and this
# dict together when a genuinely new top-level domain is warranted; deeper
# sub-topics are meant to grow from document context instead (see
# ``_KNOWN_SUBTOPICS`` below).
# ─────────────────────────────────────────────────────────────────────────
WORLDVIEW_TAXONOMY: dict[str, list[str]] = {
    "Science": [
        "Physics",
        "Chemistry",
        "Biology",
        "Astronomy & Space",
        "General & Interdisciplinary Science",
    ],
    "Technology & Computing": [
        "Software Engineering",
        "Artificial Intelligence & Machine Learning",
        "Computer Hardware",
        "Cybersecurity",
        "Networking & Internet",
        "Data & Analytics",
    ],
    "Engineering": [
        "Mechanical Engineering",
        "Civil Engineering",
        "Electrical Engineering",
        "Aerospace Engineering",
        "Robotics & Automation",
        "Industrial & Manufacturing Engineering",
    ],
    "Mathematics": [
        "Pure Mathematics",
        "Applied Mathematics",
        "Statistics & Probability",
        "Logic & Foundations",
    ],
    "Health & Medicine": [
        "Clinical Medicine",
        "Public Health",
        "Mental Health",
        "Biotechnology & Pharmaceuticals",
        "Nutrition & Fitness",
    ],
    "Society & Politics": [
        "Governance & Elections",
        "Geopolitics & International Relations",
        "Public Policy",
        "Social Movements & Activism",
        "Demographics & Social Issues",
    ],
    "Economy & Business": [
        "Finance & Markets",
        "Entrepreneurship & Startups",
        "Management & Strategy",
        "Trade & Industry",
        "Labor & Employment",
    ],
    "Law & Governance": [
        "Legislation & Regulation",
        "Judicial & Legal Systems",
        "Human Rights",
        "Compliance & Regulatory Affairs",
        "Public Administration",
    ],
    "Environment & Earth": [
        "Climate & Climate Change",
        "Ecology & Biodiversity",
        "Energy & Natural Resources",
        "Sustainability",
        "Geography & Geology",
    ],
    "Arts & Culture": [
        "Visual Arts",
        "Music",
        "Film & Television",
        "Literature",
        "Design & Architecture",
    ],
    "Humanities & Philosophy": [
        "Philosophy & Ethics",
        "History",
        "Religion & Belief Systems",
        "Linguistics",
        "Education",
    ],
    "Sports & Recreation": [
        "Competitive Sports",
        "Fitness & Wellness Activities",
        "Games & Esports",
        "Travel & Leisure",
    ],
}

# Process-local registry of known sub-topics per top-level domain, seeded from
# the curated first layer above and grown as the classifier mints genuinely new,
# context-derived sub-topics — so re-classified content tends to converge on the
# SAME narrower node (topology) instead of minting a fresh synonym every time.
_KNOWN_SUBTOPICS: dict[str, set[str]] = {
    top: {sub for sub in subs} for top, subs in WORLDVIEW_TAXONOMY.items()
}


def _slugify(label: str) -> str:
    """``"Artificial Intelligence & Machine Learning"`` → ``"artificial-intelligence-and-machine-learning"``."""
    s = label.strip().lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def topic_node_id(*labels: str) -> str:
    """Deterministic ``:Topic`` node id for a WorldView path, e.g. ``topic:technology-computing/artificial-intelligence-and-machine-learning``."""
    slugs = [_slugify(label) for label in labels if label and label.strip()]
    return "topic:" + "/".join(slugs)


# ─────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────


class TopicAssignment(BaseModel):
    """One document's WorldView classification (CONCEPT:AU-KG.enrichment.topic-classification-topology)."""

    top_level: str = Field(
        description=(
            "The single best-fit top-level WorldView domain label, EXACTLY as "
            "given in the taxonomy (e.g. 'Technology & Computing')."
        )
    )
    sub_topic: str = Field(
        default="",
        description=(
            "The most specific sub-topic under ``top_level`` that fits the "
            "content. Reuse one of the 'existing sub-topics' listed for that "
            "domain when it genuinely fits; otherwise propose a new, concise "
            "(2-5 word) sub-topic label in Title Case. Empty string if the "
            "content is too general for any sub-topic beyond the top level."
        ),
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Classification confidence, 0-1."
    )
    reasoning: str = Field(
        default="", description="One sentence explaining the assignment."
    )


_TAXONOMY_BLOCK = "\n".join(
    f"- {top}: {', '.join(subs)}" for top, subs in WORLDVIEW_TAXONOMY.items()
)

_SYSTEM_PROMPT = f"""\
You are the subject-domain classifier for the agent-utilities Knowledge Graph's
WorldView taxonomy — a curated, reasonably-MECE upper ontology of world-knowledge
domains every ingested document must hang from.

TOP-LEVEL DOMAINS (choose EXACTLY one, verbatim):
{_TAXONOMY_BLOCK}

Rules:
1. Pick the single best-fit top-level domain from the fixed list above — never
   invent a new top-level domain.
2. Pick the MOST SPECIFIC sub-topic that fits. Prefer reusing one of the
   "existing sub-topics" supplied in the user prompt for that domain (this keeps
   related documents converging on the same node instead of near-duplicate
   labels). Only propose a genuinely new sub-topic when none of the existing
   ones fit the content's actual focus.
3. Confidence reflects how clearly the content matches the assignment, not how
   important the content is.
4. Be decisive — always return a top_level even for ambiguous or short content
   (pick the closest fit and lower the confidence instead of refusing).
"""


def _classifier_agent() -> Any:
    """Lazily build the pydantic-ai classification agent (reasoning OFF by default)."""
    try:
        from pydantic_ai import Agent

        from ...core.config import config
        from ...core.model_factory import create_model

        lite = getattr(config, "lite_chat_model", None)
        default = getattr(config, "default_chat_model", None)
        chosen = lite or default
        model = create_model(
            provider=getattr(chosen, "provider", None) if chosen else None,
            model_id=getattr(chosen, "id", None) if chosen else None,
            base_url=getattr(chosen, "base_url", None) if chosen else None,
            api_key=getattr(chosen, "api_key", None) if chosen else None,
            reasoning_effort="none",
        )
        return Agent(
            model=model, output_type=TopicAssignment, system_prompt=_SYSTEM_PROMPT
        )
    except Exception as exc:  # noqa: BLE001 — classification is best-effort
        logger.debug("[topic_classifier] agent construction failed: %s", exc)
        return None


def _existing_subtopics_for(top_level: str, limit: int = 30) -> list[str]:
    key = next(
        (k for k in _KNOWN_SUBTOPICS if k.strip().lower() == top_level.strip().lower()),
        None,
    )
    if key is None:
        return []
    return sorted(_KNOWN_SUBTOPICS[key])[:limit]


def _heuristic_classify(text: str, title: str = "") -> TopicAssignment:
    """Deterministic keyword-overlap fallback when no LLM is reachable.

    Scores every (top-level, sub-topic) label against the text by simple
    case-insensitive substring/word-overlap counting and returns the best hit.
    Never raises; degrades to the first taxonomy entry with low confidence.
    """
    hay = f"{title}\n{text}".lower()
    best_top = next(iter(WORLDVIEW_TAXONOMY))
    best_sub = ""
    best_score = -1
    for top, subs in WORLDVIEW_TAXONOMY.items():
        top_score = hay.count(top.lower().split(" & ")[0].strip().lower())
        for sub in subs:
            words = [w for w in re.split(r"[^a-z0-9]+", sub.lower()) if len(w) > 3]
            sub_score = sum(hay.count(w) for w in words) + top_score
            if sub_score > best_score:
                best_score = sub_score
                best_top, best_sub = top, sub
    confidence = 0.55 if best_score > 0 else 0.2
    return TopicAssignment(
        top_level=best_top,
        sub_topic=best_sub if best_score > 0 else "",
        confidence=confidence,
        reasoning="heuristic keyword-overlap fallback (LLM unavailable)",
    )


async def classify_topic(
    text: str, *, title: str = "", source_type: str = ""
) -> TopicAssignment:
    """Classify ``text`` onto the WorldView taxonomy. Never raises."""
    text = (text or "").strip()
    if not text:
        return TopicAssignment(top_level=next(iter(WORLDVIEW_TAXONOMY)), confidence=0.0)

    agent = _classifier_agent()
    if agent is None:
        return _heuristic_classify(text, title)

    try:
        # A cheap, coarse top-level guess narrows which "existing sub-topics"
        # list to show the model up front (it may still pick a different
        # top-level — this is only a prompt-shaping hint, not a constraint).
        hinted = _heuristic_classify(text, title).top_level
        existing = _existing_subtopics_for(hinted)
        ctx = (
            f"\n\nExisting sub-topics already known for '{hinted}' (reuse when they fit):\n"
            + "\n".join(f"- {s}" for s in existing)
            if existing
            else ""
        )
        prompt = (
            f"TITLE: {title or '(untitled)'}\n"
            f"SOURCE TYPE: {source_type or 'document'}\n"
            f"{ctx}\n\nCONTENT:\n{text[:8000]}"
        )
        result = await agent.run(prompt)
        assignment: TopicAssignment = result.output
        if assignment.top_level not in WORLDVIEW_TAXONOMY:
            # Model drifted from the fixed list — snap to the nearest known
            # label by case-insensitive match, else fall back heuristically.
            match = next(
                (
                    k
                    for k in WORLDVIEW_TAXONOMY
                    if k.strip().lower() == assignment.top_level.strip().lower()
                ),
                None,
            )
            if match is None:
                return _heuristic_classify(text, title)
            assignment.top_level = match
        return assignment
    except Exception as exc:  # noqa: BLE001 — classification never breaks ingest
        logger.debug("[topic_classifier] LLM classification failed: %s", exc)
        return _heuristic_classify(text, title)


# ─────────────────────────────────────────────────────────────────────────
# Graph materialization — :Topic nodes + BROADER/NARROWER + HAS_TOPIC/CLASSIFIED_AS
# ─────────────────────────────────────────────────────────────────────────


def _ensure_topic_node(add_node: Any, add_edge: Any, path: list[str]) -> str:
    """Ensure a ``:Topic`` node exists for ``path`` (idempotent upsert-by-id) and
    its BROADER/NARROWER edge to its parent. Returns the topic node id."""
    node_id = topic_node_id(*path)
    label = path[-1]
    add_node(
        node_id,
        type="Topic",
        name=label,
        level=len(path),
        path="/".join(_slugify(p) for p in path),
        is_worldview_domain=(len(path) == 1),
    )
    if len(path) > 1:
        parent_id = _ensure_topic_node(add_node, add_edge, path[:-1])
        add_edge(node_id, parent_id, rel_type="BROADER")
        add_edge(parent_id, node_id, rel_type="NARROWER")
    return node_id


def _link_document_topic(
    add_edge: Any, doc_id: str, topic_id: str, *, confidence: float, primary: bool
) -> None:
    add_edge(doc_id, topic_id, rel_type="HAS_TOPIC", confidence=confidence)
    if primary:
        add_edge(doc_id, topic_id, rel_type="CLASSIFIED_AS", confidence=confidence)


async def classify_and_link_topics(
    backend: Any,
    doc_id: str,
    text: str,
    *,
    title: str = "",
    source_type: str = "",
) -> dict[str, Any]:
    """Classify ``text`` and materialize the ``:Topic`` topology for ``doc_id``.

    The single entry point the unified ingestion enrichment seam and the
    ``graph_ingest action=classify_topics`` MCP/REST action both call. Mints
    (idempotently) the top-level ``:WorldViewDomain`` node and, when a
    sub-topic is assigned, its child ``:Topic`` node + ``BROADER``/``NARROWER``
    edges, then links ``doc_id`` to both via ``HAS_TOPIC`` (every level) and
    ``CLASSIFIED_AS`` (the primary/most-specific level), each carrying the
    classifier's confidence. Best-effort: never raises, degrades to
    ``status: "skipped"``/``"failed"``.
    """
    result: dict[str, Any] = {
        "doc_id": doc_id,
        "top_level": None,
        "sub_topic": None,
        "confidence": 0.0,
        "topic_ids": [],
        "primary_topic_id": None,
        "status": "skipped",
    }
    if not text or not text.strip():
        return result
    add_node = getattr(backend, "add_node", None)
    add_edge = getattr(backend, "add_edge", None)
    if not callable(add_node) or not callable(add_edge):
        return result

    assignment = await classify_topic(text, title=title, source_type=source_type)

    try:
        top_id = _ensure_topic_node(add_node, add_edge, [assignment.top_level])
        sub_id: str | None = None
        sub_topic = (assignment.sub_topic or "").strip()
        if sub_topic and sub_topic.lower() != assignment.top_level.strip().lower():
            sub_id = _ensure_topic_node(
                add_node, add_edge, [assignment.top_level, sub_topic]
            )
            _KNOWN_SUBTOPICS.setdefault(assignment.top_level, set()).add(sub_topic)

        primary_id = sub_id or top_id
        _link_document_topic(
            add_edge,
            doc_id,
            top_id,
            confidence=assignment.confidence,
            primary=(primary_id == top_id),
        )
        if sub_id:
            _link_document_topic(
                add_edge, doc_id, sub_id, confidence=assignment.confidence, primary=True
            )

        result.update(
            top_level=assignment.top_level,
            sub_topic=sub_topic or None,
            confidence=assignment.confidence,
            reasoning=assignment.reasoning,
            topic_ids=[t for t in (top_id, sub_id) if t],
            primary_topic_id=primary_id,
            status="classified",
        )
    except Exception as exc:  # noqa: BLE001 — enrichment must never break ingest
        logger.warning(
            "[topic_classifier] materialization failed for %s: %s", doc_id, exc
        )
        result["status"] = "failed"
    return result


__all__ = [
    "WORLDVIEW_TAXONOMY",
    "TopicAssignment",
    "topic_node_id",
    "classify_topic",
    "classify_and_link_topics",
]
