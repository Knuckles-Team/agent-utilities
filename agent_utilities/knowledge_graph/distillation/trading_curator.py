"""Trading-knowledge curator — organise (don't dump) book/PDF/notes knowledge.

CONCEPT:EE-036

After the standard Document + chunks + Concept ingestion has extracted concepts
from a trading book, paper, or set of notes, this module *classifies* each
concept into the trading-knowledge taxonomy (Strategy / Risk / Execution) and
materialises typed, queryable nodes with citations and an extraction-confidence
score — rather than leaving the knowledge as an opaque verbatim document. Chapters
about order flow / microstructure additionally seed a ``MicrostructureSignal``
node linked ``DERIVED_FROM`` the concept, so curated reading feeds the same priors
the live signal-fusion consumes.

The classifier (``classify_trading_concept``) and node builder
(``build_knowledge_nodes``) are pure and unit-testable; ``organize_trading_knowledge``
is the thin async wrapper that writes the result through an epistemic-graph client.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

from ...models.domains.finance import (
    ExecutionConceptNode,
    MicrostructureSignalNode,
    RiskConceptNode,
    StrategyConceptNode,
)

logger = logging.getLogger(__name__)

# Category → weighted keyword cues. Deliberately interpretable: the agent (and a
# human) can see *why* a concept was filed under a category.
TRADING_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "execution": (
        "order flow",
        "order book",
        "microstructure",
        "queue",
        "limit order",
        "market maker",
        "market making",
        "bid-ask",
        "bid ask",
        "spread",
        "slippage",
        "imbalance",
        "vwap",
        "twap",
        "fill",
        "latency",
        "tick",
    ),
    "risk": (
        "risk",
        "drawdown",
        "position sizing",
        "kelly",
        "stop loss",
        "stop-loss",
        "var",
        "value at risk",
        "volatility target",
        "leverage",
        "exposure",
        "hedge",
        "capital preservation",
        "ruin",
    ),
    "strategy": (
        "strategy",
        "alpha",
        "factor",
        "mean reversion",
        "momentum",
        "carry",
        "arbitrage",
        "signal",
        "edge",
        "backtest",
        "entry",
        "exit",
        "regime",
        "trend",
    ),
}

# Cues that an Execution concept describes a microstructure signal worth seeding.
_MICROSTRUCTURE_CUES = (
    "order flow",
    "order book",
    "microstructure",
    "queue",
    "imbalance",
    "spread",
    "tick",
)

_NODE_TYPES = {
    "strategy": StrategyConceptNode,
    "risk": RiskConceptNode,
    "execution": ExecutionConceptNode,
}


def classify_trading_concept(text: str) -> tuple[str | None, float]:
    """Classify free text into ``strategy`` / ``risk`` / ``execution``.

    Returns ``(category, confidence)`` where confidence ∈ [0, 1] reflects the
    margin of the winning category over the field. ``(None, 0.0)`` when no cue
    matches (the concept is not trading knowledge and should be skipped).
    """
    if not text:
        return None, 0.0
    low = text.lower()
    scores: dict[str, int] = {}
    for category, cues in TRADING_CATEGORY_KEYWORDS.items():
        scores[category] = sum(1 for cue in cues if cue in low)
    total = sum(scores.values())
    if total == 0:
        return None, 0.0
    category = max(scores, key=lambda k: scores[k])
    # Confidence: winner share of all cue hits, lightly boosted by absolute hits.
    share = scores[category] / total
    confidence = min(1.0, share * (0.6 + 0.1 * scores[category]))
    return category, round(confidence, 4)


def _looks_like_microstructure(text: str) -> bool:
    low = (text or "").lower()
    return any(cue in low for cue in _MICROSTRUCTURE_CUES)


def _concept_field(concept: Any, key: str, default: str = "") -> str:
    if isinstance(concept, Mapping):
        return str(concept.get(key, default) or default)
    return str(getattr(concept, key, default) or default)


def build_knowledge_nodes(
    concepts: Iterable[Any],
    *,
    source_title: str = "",
    min_confidence: float = 0.3,
) -> dict[str, list[Any]]:
    """Classify concepts and build typed knowledge nodes (pure, no I/O).

    ``concepts`` is an iterable of records exposing ``id`` and ``text`` (and
    optionally ``chapter`` / ``page_span``), as a mapping or an object. Returns
    ``{"knowledge": [...], "signals": [...], "skipped": [...]}`` where each
    knowledge node is a Strategy/Risk/Execution concept node, signals are
    ``MicrostructureSignalNode`` seeds for order-flow chapters, and skipped lists
    the ids classified below ``min_confidence`` or as non-trading text.
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    knowledge: list[Any] = []
    signals: list[Any] = []
    skipped: list[str] = []

    for concept in concepts:
        cid = _concept_field(concept, "id") or f"concept:{uuid.uuid4().hex[:8]}"
        text = _concept_field(concept, "text") or _concept_field(concept, "summary")
        category, confidence = classify_trading_concept(text)
        if category is None or confidence < min_confidence:
            skipped.append(cid)
            continue

        node_cls = _NODE_TYPES[category]
        chapter = _concept_field(concept, "chapter")
        page_span = _concept_field(concept, "page_span")
        name = (text[:60].strip() or cid) if text else cid
        node = node_cls(
            id=f"tk:{category}:{cid}",
            name=name,
            topic=category,
            source=source_title,
            chapter=chapter,
            page_span=page_span,
            confidence=confidence,
            timestamp=ts,
        )
        knowledge.append(node)

        # Order-flow / microstructure chapters seed a signal with the book as
        # provenance (priors start at the extraction confidence until backtested).
        if category == "execution" and _looks_like_microstructure(text):
            sig_name = re.sub(r"\s+", " ", name).strip()
            signals.append(
                MicrostructureSignalNode(
                    id=f"sig:book:{cid}",
                    name=sig_name or cid,
                    directional_accuracy=confidence,
                    standalone_sharpe=0.0,  # unproven until backtested
                    provenance=f"{source_title} ch.{chapter}".strip(),
                    timestamp=ts,
                )
            )

    return {"knowledge": knowledge, "signals": signals, "skipped": skipped}


async def organize_trading_knowledge(
    client: Any,
    concepts: Iterable[Any],
    *,
    source_title: str = "",
    min_confidence: float = 0.3,
) -> dict[str, Any]:
    """Classify concepts and write the typed knowledge graph via ``client``.

    ``client`` is an epistemic-graph client exposing ``nodes.add`` and
    ``edges.add``. Creates the typed concept nodes, seeds MicrostructureSignal
    nodes for order-flow chapters with a ``DERIVED_FROM`` edge back to the source
    concept, and returns a summary. Tolerant of a missing client (dry run).
    """
    built = build_knowledge_nodes(
        concepts, source_title=source_title, min_confidence=min_confidence
    )
    written_nodes = 0
    written_edges = 0

    if client is not None:
        for node in built["knowledge"]:
            try:
                await client.nodes.add(node.id, node.model_dump(mode="json"))
                written_nodes += 1
            except Exception as exc:  # noqa: BLE001 — best-effort, keep going
                logger.debug("failed to write knowledge node %s: %s", node.id, exc)
        for sig in built["signals"]:
            try:
                await client.nodes.add(sig.id, sig.model_dump(mode="json"))
                written_nodes += 1
                # sig:book:<cid> derives from tk:execution:<cid>
                cid = sig.id.split(":", 2)[-1]
                try:
                    await client.edges.add(
                        sig.id, f"tk:execution:{cid}", "DERIVED_FROM"
                    )
                    written_edges += 1
                except Exception as exc:  # noqa: BLE001
                    logger.debug("failed DERIVED_FROM edge for %s: %s", sig.id, exc)
            except Exception as exc:  # noqa: BLE001
                logger.debug("failed to write signal node %s: %s", sig.id, exc)

    return {
        "source": source_title,
        "knowledge_nodes": len(built["knowledge"]),
        "signal_seeds": len(built["signals"]),
        "skipped": len(built["skipped"]),
        "written_nodes": written_nodes,
        "written_edges": written_edges,
        "by_category": {
            cat: sum(1 for n in built["knowledge"] if n.topic == cat)
            for cat in TRADING_CATEGORY_KEYWORDS
        },
    }
