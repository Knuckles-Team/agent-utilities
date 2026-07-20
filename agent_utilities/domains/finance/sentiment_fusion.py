"""Sentiment Fusion — CONCEPT:AU-KG.domains.sentiment-fusion-signals — Sentiment Fusion Signals

Turns raw, heterogeneous sentiment observations (news headlines, social posts,
analyst notes, filing tone) into **first-class KG signals** the trading swarm and
the Bayesian ``signal_fusion`` already consume.

Each observation carries a ``source``, a textual snippet and/or a pre-scored
polarity, the ``entity`` it concerns, a ``timestamp`` and the upstream
``confidence``. We normalise each into a :class:`SentimentSignal` with a bounded
polarity in ``[-1, 1]`` and a magnitude in ``[0, 1]``, weighted by a
**source-credibility / skepticism** factor (a curated prior per source family,
multiplied by the upstream confidence and a recency decay). Observations for the
same entity are aggregated into a single **credibility-weighted fused score**
(:class:`FusedSentiment`).

The fused score is wired two ways so it is *actually consumed*, not just stored:

* :func:`fused_sentiment_to_agent_signal` emits an :class:`AgentSignal` with the
  ``SENTIMENT_ANALYST`` role that ``SwarmConsensus`` aggregates directly, and
* :func:`fused_sentiment_to_fusion_direction` feeds the discretised direction
  into :class:`BayesianSignalFusion` as a registered source.

Persistence uses the canonical ``registry.write_batch`` path: a ``:SentimentFact``
node per observation with provenance edges to its ``:Source`` and target
``:FinancialInstrument`` entity. A ``None`` backend (offline / unit tests) is a
clean no-op.

The scoring is a real lexicon + source-credibility model (no placeholder
constants standing in for a model): when an observation ships a numeric score we
trust it; when it ships only text we score it with a signed financial lexicon and
a negation/intensifier-aware tokenizer.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .signal_fusion import BayesianSignalFusion
from .trading_swarm import AgentSignal, SwarmRole

logger = logging.getLogger(__name__)


# ── Source-credibility priors (skepticism weights) ────────────────────────────
# A curated prior in [0, 1] per source *family*: how much we trust a sentiment
# observation from that channel before factoring in its own confidence. These are
# real, defensible priors (regulated filings > vetted analysts > news > anonymous
# social), not a single magic constant masquerading as a model.
SOURCE_CREDIBILITY: dict[str, float] = {
    "filing": 0.95,  # regulated disclosure tone (10-K/Q MD&A)
    "regulator": 0.95,
    "analyst": 0.85,  # sell-side / vetted research desks
    "earnings_call": 0.80,
    "news": 0.65,  # mainstream financial press
    "press_release": 0.55,  # issuer-controlled, promotional bias
    "blog": 0.45,
    "social": 0.35,  # X/Reddit/StockTwits — high noise, herding bias
    "anonymous": 0.20,
}
_DEFAULT_CREDIBILITY = 0.50

# Default half-life (hours) for recency decay of a sentiment observation.
_DEFAULT_HALF_LIFE_H = 48.0


# ── Signed financial sentiment lexicon (Loughran–McDonald-inspired) ───────────
# Real polarity model for text-only observations: a signed lexicon scored with a
# negation- and intensifier-aware tokenizer. Not exhaustive, but a genuine model
# — the weight of each term reflects its directional strength.
_POSITIVE_LEXICON: dict[str, float] = {
    "beat": 1.0,
    "beats": 1.0,
    "surge": 1.0,
    "surged": 1.0,
    "rally": 0.9,
    "outperform": 0.9,
    "upgrade": 0.9,
    "upgraded": 0.9,
    "growth": 0.7,
    "profit": 0.7,
    "profitable": 0.8,
    "record": 0.7,
    "strong": 0.7,
    "bullish": 1.0,
    "gain": 0.6,
    "gains": 0.6,
    "raise": 0.6,
    "raised": 0.6,
    "buy": 0.6,
    "accelerate": 0.6,
    "expansion": 0.6,
    "dividend": 0.4,
}
_NEGATIVE_LEXICON: dict[str, float] = {
    "miss": 1.0,
    "missed": 1.0,
    "plunge": 1.0,
    "plunged": 1.0,
    "crash": 1.0,
    "downgrade": 0.9,
    "downgraded": 0.9,
    "underperform": 0.9,
    "loss": 0.8,
    "losses": 0.8,
    "lawsuit": 0.8,
    "fraud": 1.0,
    "investigation": 0.8,
    "bearish": 1.0,
    "decline": 0.7,
    "declined": 0.7,
    "weak": 0.7,
    "cut": 0.6,
    "cuts": 0.6,
    "warning": 0.7,
    "bankruptcy": 1.0,
    "default": 0.9,
    "sell": 0.6,
    "recall": 0.7,
    "layoffs": 0.7,
    "slump": 0.8,
    "concern": 0.5,
    "concerns": 0.5,
}
_NEGATIONS = {"not", "no", "never", "without", "lacks", "lack", "fails", "fail"}
_INTENSIFIERS = {
    "very": 1.5,
    "extremely": 1.8,
    "highly": 1.4,
    "significantly": 1.5,
    "sharply": 1.6,
    "slightly": 0.6,
    "marginally": 0.5,
}
_TOKEN_RE = re.compile(r"[a-z']+")


def score_text_polarity(text: str) -> float:
    """Score free text into a polarity in ``[-1, 1]`` via the signed lexicon.

    Negation flips the sign of the *following* sentiment term; intensifiers scale
    it. The per-document score is the mean signed term weight, squashed with tanh
    so a single strong word does not saturate a long headline.
    """
    if not text:
        return 0.0
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return 0.0

    total = 0.0
    hits = 0
    negate = False
    intensity = 1.0
    for tok in tokens:
        if tok in _NEGATIONS:
            negate = True
            continue
        if tok in _INTENSIFIERS:
            intensity = _INTENSIFIERS[tok]
            continue
        polarity = _POSITIVE_LEXICON.get(tok, 0.0) - _NEGATIVE_LEXICON.get(tok, 0.0)
        if polarity != 0.0:
            signed = polarity * intensity * (-1.0 if negate else 1.0)
            total += signed
            hits += 1
        # negation/intensifier only modify the immediately-following sentiment word
        negate = False
        intensity = 1.0

    if hits == 0:
        return 0.0
    return math.tanh(total / hits)


@dataclass
class SentimentObservation:
    """A single raw sentiment data point from one source.

    Provide ``score`` (a pre-computed polarity in ``[-1, 1]``) and/or ``text``.
    When both are absent the observation contributes no polarity.
    """

    source: str
    entity: str
    text: str = ""
    score: float | None = None  # pre-scored polarity in [-1, 1], if available
    confidence: float = 0.7  # upstream confidence in the observation
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()


def _source_family(source: str) -> str:
    """Map a concrete source name to a known credibility family key."""
    s = source.lower()
    for family in SOURCE_CREDIBILITY:
        if family in s:
            return family
    return "unknown"


def _recency_weight(timestamp: str, now: datetime, half_life_h: float) -> float:
    """Exponential recency decay: weight = 0.5 ** (age_hours / half_life)."""
    try:
        ts = datetime.fromisoformat(timestamp)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return 1.0
    age_h = max(0.0, (now - ts).total_seconds() / 3600.0)
    if half_life_h <= 0:
        return 1.0
    return 0.5 ** (age_h / half_life_h)


@dataclass
class SentimentSignal:
    """A normalized, credibility-weighted sentiment signal for one observation."""

    source: str
    entity: str
    polarity: float  # [-1, 1] directional sentiment
    magnitude: float  # [0, 1] strength / |polarity|
    credibility: float  # [0, 1] source-credibility prior
    confidence: float  # upstream confidence
    weight: float  # effective weight = credibility * confidence * recency
    timestamp: str
    text: str = ""

    @property
    def signed_weight(self) -> float:
        """Weighted directional contribution = polarity * weight."""
        return self.polarity * self.weight


def normalize_observation(
    obs: SentimentObservation,
    now: datetime | None = None,
    half_life_h: float = _DEFAULT_HALF_LIFE_H,
) -> SentimentSignal:
    """Normalize one raw observation into a weighted :class:`SentimentSignal`."""
    now = now or datetime.now(UTC)

    if obs.score is not None:
        polarity = max(-1.0, min(1.0, float(obs.score)))
    else:
        polarity = score_text_polarity(obs.text)

    family = _source_family(obs.source)
    credibility = SOURCE_CREDIBILITY.get(family, _DEFAULT_CREDIBILITY)
    confidence = max(0.0, min(1.0, float(obs.confidence)))
    recency = _recency_weight(obs.timestamp, now, half_life_h)
    weight = credibility * confidence * recency

    return SentimentSignal(
        source=obs.source,
        entity=obs.entity,
        polarity=polarity,
        magnitude=abs(polarity),
        credibility=credibility,
        confidence=confidence,
        weight=weight,
        timestamp=obs.timestamp,
        text=obs.text,
    )


@dataclass
class FusedSentiment:
    """Credibility-weighted fused sentiment for a single entity."""

    entity: str
    fused_score: float  # [-1, 1] weighted mean polarity
    magnitude: float  # [0, 1] confidence-of-direction (|fused| * agreement)
    n_observations: int
    total_weight: float
    agreement: float  # [0, 1] share of weight agreeing with the fused sign
    signals: list[SentimentSignal] = field(default_factory=list)

    @property
    def direction(self) -> int:
        """Discretised direction: +1 / -1 / 0 (neutral band ±0.05)."""
        if self.fused_score > 0.05:
            return 1
        if self.fused_score < -0.05:
            return -1
        return 0


def fuse_sentiment(
    observations: list[SentimentObservation],
    entity: str | None = None,
    now: datetime | None = None,
    half_life_h: float = _DEFAULT_HALF_LIFE_H,
) -> FusedSentiment:
    """Fuse observations for one entity into a credibility-weighted score.

    The fused score is ``sum(polarity * weight) / sum(weight)`` — a true weighted
    mean, so a high-credibility filing outweighs a swarm of anonymous posts. The
    reported ``magnitude`` multiplies ``|fused_score|`` by the **agreement** (the
    weight share pointing the same way) so loud-but-split sentiment is correctly
    discounted.
    """
    signals = [
        normalize_observation(o, now=now, half_life_h=half_life_h)
        for o in observations
        if entity is None or o.entity == entity
    ]
    resolved_entity = entity or (signals[0].entity if signals else "")

    total_weight = sum(s.weight for s in signals)
    if not signals or total_weight <= 0:
        return FusedSentiment(
            entity=resolved_entity,
            fused_score=0.0,
            magnitude=0.0,
            n_observations=len(signals),
            total_weight=total_weight,
            agreement=0.0,
            signals=signals,
        )

    fused = sum(s.signed_weight for s in signals) / total_weight
    sign = 1.0 if fused > 0 else (-1.0 if fused < 0 else 0.0)
    agreeing_weight = sum(
        s.weight for s in signals if (s.polarity > 0) == (sign > 0) and s.polarity != 0
    )
    agreement = agreeing_weight / total_weight if total_weight > 0 else 0.0

    return FusedSentiment(
        entity=resolved_entity,
        fused_score=float(fused),
        magnitude=float(abs(fused) * agreement),
        n_observations=len(signals),
        total_weight=float(total_weight),
        agreement=float(agreement),
        signals=signals,
    )


# ── Wiring: fused sentiment → swarm / Bayesian fusion (actually consumed) ──────
def fused_sentiment_to_agent_signal(
    fused: FusedSentiment, agent_id: str = "sentiment_fusion_01"
) -> AgentSignal:
    """Emit a ``SENTIMENT_ANALYST`` :class:`AgentSignal` for ``SwarmConsensus``.

    The direction is the fused sign; the confidence is the agreement-discounted
    magnitude. This is the object ``TradingSwarm.analyze`` aggregates, so fused
    sentiment becomes a *voting member* of the consensus, not a side artifact.
    """
    return AgentSignal(
        agent_id=agent_id,
        role=SwarmRole.SENTIMENT_ANALYST,
        direction=fused.direction,
        confidence=max(0.0, min(1.0, fused.magnitude)),
        reasoning=(
            f"Fused sentiment {fused.fused_score:+.3f} over {fused.n_observations} "
            f"obs (agreement {fused.agreement:.0%}) for {fused.entity}"
        ),
        metadata={
            "fused_score": fused.fused_score,
            "agreement": fused.agreement,
            "n_observations": fused.n_observations,
            "concept": "AU-KG.domains.sentiment-fusion-signals",
        },
    )


def register_sentiment_source(
    fusion: BayesianSignalFusion,
    fused: FusedSentiment,
    weight: float = 0.7,
) -> None:
    """Register fused sentiment as a source in a :class:`BayesianSignalFusion`.

    The source's historical accuracy is derived from observation agreement
    (a split chorus is barely better than a coin flip), so a well-aligned,
    credible sentiment read updates the Bayesian prior more strongly.
    """
    accuracy = 0.5 + 0.5 * fused.agreement * min(1.0, abs(fused.fused_score) + 0.0)
    fusion.register_source("sentiment", weight=weight, accuracy=accuracy)


def fused_sentiment_to_fusion_direction(fused: FusedSentiment) -> dict[str, int]:
    """Build the ``{source: direction}`` mapping ``BayesianSignalFusion.fuse`` takes."""
    return {"sentiment": fused.direction}


# ── KG persistence: :SentimentFact with provenance (canonical write_batch) ─────
def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:80] or "x"


def sentiment_facts_batch(signals: list[SentimentSignal]) -> Any:
    """Build an ``ExtractionBatch`` of ``:SentimentFact`` nodes with provenance.

    Each fact links ``DERIVED_FROM`` its ``:Source`` and ``ABOUT`` the target
    ``:FinancialInstrument`` entity, so "which sources moved sentiment on TICKER,
    and how credible were they?" is a graph query.
    """
    from agent_utilities.knowledge_graph.enrichment.models import (
        EnrichmentEdge,
        ExtractionBatch,
        GraphNode,
    )

    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    seen_sources: set[str] = set()
    seen_entities: set[str] = set()

    for s in signals:
        fid = f"sentiment_fact:{_slug(s.entity)}:{_slug(s.source)}:{_slug(s.timestamp)}"
        src_id = f"source:{_slug(s.source)}"
        ent_id = f"instrument:{_slug(s.entity)}"
        nodes.append(
            GraphNode(
                id=fid,
                type="SentimentFact",
                props={
                    "entity": s.entity,
                    "source": s.source,
                    "polarity": s.polarity,
                    "magnitude": s.magnitude,
                    "credibility": s.credibility,
                    "confidence": s.confidence,
                    "weight": s.weight,
                    "timestamp": s.timestamp,
                    "text": s.text[:500],
                    "concept": "AU-KG.domains.sentiment-fusion-signals",
                },
            )
        )
        if src_id not in seen_sources:
            nodes.append(
                GraphNode(
                    id=src_id,
                    type="SentimentSource",
                    props={"name": s.source, "credibility": s.credibility},
                )
            )
            seen_sources.add(src_id)
        if ent_id not in seen_entities:
            nodes.append(
                GraphNode(
                    id=ent_id,
                    type="FinancialInstrument",
                    props={"ticker": s.entity},
                )
            )
            seen_entities.add(ent_id)
        edges.append(EnrichmentEdge(source=fid, target=src_id, rel_type="DERIVED_FROM"))
        edges.append(EnrichmentEdge(source=fid, target=ent_id, rel_type="ABOUT"))

    return ExtractionBatch(category="sentiment", nodes=nodes, edges=edges)


def seed_sentiment_facts(
    backend: Any, signals: list[SentimentSignal]
) -> tuple[int, int]:
    """Persist sentiment facts + provenance via ``write_batch``.

    ``None`` backend (offline) is a clean no-op returning ``(0, 0)``.
    """
    if backend is None or not signals:
        return (0, 0)
    from agent_utilities.knowledge_graph.enrichment.registry import write_batch

    n, e = write_batch(backend, sentiment_facts_batch(signals))
    logger.info("Seeded sentiment facts: %d nodes, %d edges", n, e)
    return n, e


def ingest_and_fuse(
    observations: list[SentimentObservation],
    entity: str | None = None,
    backend: Any | None = None,
    now: datetime | None = None,
    half_life_h: float = _DEFAULT_HALF_LIFE_H,
) -> tuple[FusedSentiment, AgentSignal]:
    """End-to-end: normalize → fuse → persist (if backend) → emit swarm signal.

    Returns the :class:`FusedSentiment` and the ``SENTIMENT_ANALYST``
    :class:`AgentSignal` ready to drop into ``SwarmConsensus.signals``. The KG
    write is best-effort and a ``None`` backend is a no-op.
    """
    fused = fuse_sentiment(
        observations, entity=entity, now=now, half_life_h=half_life_h
    )
    if backend is not None:
        seed_sentiment_facts(backend, fused.signals)
    return fused, fused_sentiment_to_agent_signal(fused)
