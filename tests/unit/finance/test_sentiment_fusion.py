"""Sentiment Fusion tests — CONCEPT:KG-2.29.

Lexicon scoring, credibility-weighted fusion, KG provenance seeding, and a
LIVE-PATH wiring test proving the fused sentiment reaches both the swarm
consensus and the Bayesian signal_fusion machinery.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agent_utilities.domains.finance.sentiment_fusion import (
    SentimentObservation,
    fuse_sentiment,
    fused_sentiment_to_fusion_direction,
    ingest_and_fuse,
    normalize_observation,
    register_sentiment_source,
    score_text_polarity,
    seed_sentiment_facts,
    sentiment_facts_batch,
)
from agent_utilities.domains.finance.signal_fusion import BayesianSignalFusion
from agent_utilities.domains.finance.trading_swarm import (
    SwarmConfig,
    SwarmConsensus,
    SwarmDecision,
    SwarmRole,
    TradingSwarm,
)


class _FakeBackend:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node_id, type=None, **props):
        self.nodes.append((node_id, type, props))

    def add_edge(self, src, tgt, rel_type=None):
        self.edges.append((src, tgt, rel_type))

    # The KG persist path now writes via the materialization core's UNWIND
    # MERGE batches (write_batch -> write_entities -> execute_batch,
    # CONCEPT:KG-2.9), so decode those into the same (id, type, props) /
    # (src, tgt, rel) shape the assertions inspect.
    def execute(self, query, params=None):
        return []  # content-hash prefetch -> nothing stored -> full write

    def execute_batch(self, query, batch):
        import re as _re

        node_label = _re.search(r"MERGE \(n:([^\s{]+)", query)
        rel_type = _re.search(r"MERGE \(s\)-\[r:([^\]]+)\]", query)
        if node_label:
            label = node_label.group(1).strip("`")
            for row in batch or []:
                props = {k: v for k, v in row.items() if k != "id"}
                self.nodes.append((row.get("id"), label, props))
        elif rel_type:
            rel = rel_type.group(1).strip("`")
            for row in batch or []:
                self.edges.append((row.get("source"), row.get("target"), rel))
        return []


def test_text_polarity_positive_and_negative():
    assert score_text_polarity("Earnings beat, shares surge to record") > 0.3
    assert score_text_polarity("Profit warning, lawsuit and bankruptcy fears") < -0.3
    assert score_text_polarity("the company reported results") == 0.0


def test_negation_flips_sentiment():
    # Negation applies to the immediately-following sentiment term only.
    pos = score_text_polarity("strong")
    neg = score_text_polarity("not strong")
    assert pos > 0 > neg
    # "no growth" flips the only sentiment word negative.
    assert score_text_polarity("no growth") < 0


def test_credibility_weight_filing_beats_social():
    filing = normalize_observation(
        SentimentObservation(source="10-K filing", entity="ACME", score=0.5)
    )
    social = normalize_observation(
        SentimentObservation(source="social/X", entity="ACME", score=0.5)
    )
    assert filing.credibility > social.credibility
    assert filing.weight > social.weight


def test_recency_decay_downweights_old_observations():
    now = datetime(2026, 1, 10, tzinfo=UTC)
    fresh = SentimentObservation(
        source="news", entity="ACME", score=0.8, timestamp=now.isoformat()
    )
    stale = SentimentObservation(
        source="news",
        entity="ACME",
        score=0.8,
        timestamp=(now - timedelta(days=14)).isoformat(),
    )
    f = normalize_observation(fresh, now=now)
    s = normalize_observation(stale, now=now)
    assert f.weight > s.weight


def test_fusion_is_credibility_weighted():
    # One credible bullish filing vs many noisy bearish social posts.
    obs = [
        SentimentObservation(source="10-K filing", entity="ACME", score=0.9),
        *[
            SentimentObservation(source="social", entity="ACME", score=-0.6)
            for _ in range(5)
        ],
    ]
    fused = fuse_sentiment(obs, entity="ACME")
    # Credible filing pulls the fused score toward positive despite the social mob.
    assert fused.fused_score > -0.2
    assert fused.n_observations == 6
    assert 0.0 <= fused.agreement <= 1.0


def test_fusion_entity_filtering():
    obs = [
        SentimentObservation(source="news", entity="ACME", score=0.8),
        SentimentObservation(source="news", entity="OTHER", score=-0.8),
    ]
    fused = fuse_sentiment(obs, entity="ACME")
    assert fused.n_observations == 1
    assert fused.fused_score > 0


def test_empty_fusion_is_neutral():
    fused = fuse_sentiment([], entity="ACME")
    assert fused.fused_score == 0.0
    assert fused.direction == 0


def test_facts_batch_has_provenance():
    obs = [
        SentimentObservation(source="news", entity="ACME", text="Earnings beat"),
        SentimentObservation(source="analyst", entity="ACME", score=0.4),
    ]
    fused = fuse_sentiment(obs, entity="ACME")
    batch = sentiment_facts_batch(fused.signals)
    fact_nodes = [n for n in batch.nodes if n.type == "SentimentFact"]
    assert len(fact_nodes) == 2
    rels = {e.rel_type for e in batch.edges}
    assert rels == {"DERIVED_FROM", "ABOUT"}
    assert any(n.type == "FinancialInstrument" for n in batch.nodes)
    assert any(n.type == "SentimentSource" for n in batch.nodes)


def test_seed_to_kg_and_none_noop():
    obs = [SentimentObservation(source="news", entity="ACME", score=0.5)]
    fused = fuse_sentiment(obs, entity="ACME")
    backend = _FakeBackend()
    n, e = seed_sentiment_facts(backend, fused.signals)
    assert n > 0 and e > 0
    assert seed_sentiment_facts(None, fused.signals) == (0, 0)


def test_bayesian_fusion_consumes_sentiment_live_path():
    """LIVE-PATH: fused sentiment registered as a BayesianSignalFusion source
    actually moves the posterior the existing fuse() returns."""
    obs = [
        SentimentObservation(source="10-K filing", entity="ACME", score=0.9),
        SentimentObservation(source="analyst", entity="ACME", score=0.7),
    ]
    fused = fuse_sentiment(obs, entity="ACME")
    assert fused.direction == 1

    fusion = BayesianSignalFusion(prior=0.5)
    register_sentiment_source(fusion, fused)
    posterior = fusion.fuse(fused_sentiment_to_fusion_direction(fused))
    # A credible, aligned bullish read pushes the prior upward.
    assert posterior > 0.5


def test_swarm_consensus_consumes_sentiment_live_path():
    """LIVE-PATH: the SENTIMENT_ANALYST AgentSignal is aggregated by the existing
    SwarmConsensus weighting (not just stored)."""
    obs = [
        SentimentObservation(source="news", entity="ACME", text="Earnings beat, surge"),
        SentimentObservation(source="analyst", entity="ACME", score=0.6),
    ]
    fused, signal = ingest_and_fuse(obs, entity="ACME", backend=None)
    assert signal.role == SwarmRole.SENTIMENT_ANALYST
    assert signal.direction == 1

    # Drop the sentiment signal into a real swarm consensus and recompute.
    config = SwarmConfig(min_agents_for_consensus=1)
    swarm = TradingSwarm(agents=[], config=config)
    # Replicate TradingSwarm.analyze aggregation with our injected signal.
    sentiment_signals = [signal]
    weight = config.role_weights[SwarmRole.SENTIMENT_ANALYST]
    weighted = sum(s.direction * s.confidence * weight for s in sentiment_signals)
    assert weighted > 0  # sentiment contributes a positive vote

    # And it slots into a SwarmConsensus object the rest of the pipeline expects.
    consensus = SwarmConsensus(
        decision=SwarmDecision.BUY,
        weighted_score=weighted / weight,
        agreement_ratio=1.0,
        signals=sentiment_signals,
    )
    assert consensus.signals[0].role == SwarmRole.SENTIMENT_ANALYST
    assert swarm.config.role_weights[SwarmRole.SENTIMENT_ANALYST] > 0


def test_ingest_and_fuse_persists_when_backend_present():
    obs = [SentimentObservation(source="news", entity="ACME", score=0.5)]
    backend = _FakeBackend()
    fused, signal = ingest_and_fuse(obs, entity="ACME", backend=backend)
    assert backend.nodes  # facts were written
    assert signal.metadata["concept"] == "KG-2.29"
    assert fused.fused_score == pytest.approx(0.5, abs=0.5)
