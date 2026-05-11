"""Tests for CONCEPT:KG-2.6 — Strategy Sharing System."""

import json

import pytest

from agent_utilities.domains.finance.strategy_sharing import (
    PerformanceSummary,
    StrategyCard,
    StrategyCategory,
    StrategyPreset,
    StrategyRegistry,
    StrategyVisibility,
)


@pytest.fixture
def sample_card():
    return StrategyCard(
        card_id="card:001",
        name="RSI Reversal",
        author="quant_agent",
        category=StrategyCategory.MEAN_REVERSION,
        tags=["rsi", "reversal", "equity"],
        parameters={"rsi_period": 14, "overbought": 70, "oversold": 30},
        entry_rules=["RSI < 30"],
        exit_rules=["RSI > 70"],
        risk_params={"stop_loss_pct": 2.0, "take_profit_pct": 5.0},
        performance=PerformanceSummary(
            sharpe_ratio=1.5, total_return=0.25, max_drawdown=-0.08, win_rate=0.58
        ),
    )


@pytest.fixture
def momentum_card():
    return StrategyCard(
        card_id="card:002",
        name="Trend Rider",
        author="alpha_team",
        category=StrategyCategory.TREND_FOLLOWING,
        tags=["momentum", "trend"],
        performance=PerformanceSummary(sharpe_ratio=2.1, total_return=0.40),
    )


class TestStrategyCard:
    def test_creation(self, sample_card):
        assert sample_card.card_id == "card:001"
        assert sample_card.created_at != ""

    def test_content_hash(self, sample_card):
        assert len(sample_card.content_hash) == 16

    def test_deterministic_hash(self, sample_card):
        h1 = sample_card.content_hash
        h2 = sample_card.content_hash
        assert h1 == h2

    def test_to_dict(self, sample_card):
        d = sample_card.to_dict()
        assert d["name"] == "RSI Reversal"
        assert d["category"] == "mean_reversion"
        assert "content_hash" in d

    def test_to_json(self, sample_card):
        j = sample_card.to_json()
        parsed = json.loads(j)
        assert parsed["name"] == "RSI Reversal"

    def test_from_dict_roundtrip(self, sample_card):
        d = sample_card.to_dict()
        restored = StrategyCard.from_dict(d)
        assert restored.name == sample_card.name
        assert restored.category == sample_card.category


class TestStrategyRegistry:
    def test_publish(self, sample_card):
        registry = StrategyRegistry()
        card_id = registry.publish(sample_card)
        assert card_id == "card:001"
        assert sample_card.visibility == StrategyVisibility.SHARED

    def test_get(self, sample_card):
        registry = StrategyRegistry()
        registry.publish(sample_card)
        retrieved = registry.get("card:001")
        assert retrieved is not None
        assert retrieved.name == "RSI Reversal"

    def test_search_by_category(self, sample_card, momentum_card):
        registry = StrategyRegistry()
        registry.publish(sample_card)
        registry.publish(momentum_card)
        results = registry.search(category=StrategyCategory.MEAN_REVERSION)
        assert len(results) == 1
        assert results[0].name == "RSI Reversal"

    def test_search_by_tags(self, sample_card, momentum_card):
        registry = StrategyRegistry()
        registry.publish(sample_card)
        registry.publish(momentum_card)
        results = registry.search(tags=["rsi"])
        assert len(results) == 1

    def test_search_by_min_sharpe(self, sample_card, momentum_card):
        registry = StrategyRegistry()
        registry.publish(sample_card)
        registry.publish(momentum_card)
        results = registry.search(min_sharpe=2.0)
        assert len(results) == 1
        assert results[0].name == "Trend Rider"

    def test_search_by_author(self, sample_card, momentum_card):
        registry = StrategyRegistry()
        registry.publish(sample_card)
        registry.publish(momentum_card)
        results = registry.search(author="quant_agent")
        assert len(results) == 1

    def test_private_excluded_from_search(self):
        registry = StrategyRegistry()
        card = StrategyCard(
            card_id="priv",
            name="Secret",
            author="me",
            visibility=StrategyVisibility.PRIVATE,
        )
        registry._cards[card.card_id] = card  # Add without publishing
        results = registry.search()
        assert len(results) == 0

    def test_presets(self, sample_card):
        registry = StrategyRegistry()
        registry.publish(sample_card)
        preset = StrategyPreset(
            preset_id="p1", name="Conservative", parameters={"rsi_period": 21}
        )
        registry.add_preset("card:001", preset)
        presets = registry.get_presets("card:001")
        assert len(presets) == 1
        assert presets[0].name == "Conservative"

    def test_fork(self, sample_card):
        registry = StrategyRegistry()
        registry.publish(sample_card)
        forked = registry.fork("card:001", "new_author")
        assert forked is not None
        assert forked.author == "new_author"
        assert "Fork" in forked.name
        assert forked.visibility == StrategyVisibility.PRIVATE

    def test_fork_nonexistent(self):
        registry = StrategyRegistry()
        assert registry.fork("nope", "author") is None

    def test_card_count(self, sample_card, momentum_card):
        registry = StrategyRegistry()
        registry.publish(sample_card)
        registry.publish(momentum_card)
        assert registry.card_count == 2

    def test_public_count(self, sample_card):
        registry = StrategyRegistry()
        registry.publish(sample_card)
        assert registry.public_count == 1
