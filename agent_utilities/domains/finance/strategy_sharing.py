"""
Strategy Sharing System — CONCEPT:AU-KG.research.research-pipeline-runner

Provides shareable strategy cards with metadata, configuration presets,
and strategy marketplace primitives.

Source: Vibe-Trading community strategy sharing
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

logger = logging.getLogger(__name__)


class StrategyVisibility(StrEnum):
    PRIVATE = "private"
    SHARED = "shared"
    PUBLIC = "public"


class StrategyCategory(StrEnum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    FACTOR_BASED = "factor_based"
    SENTIMENT = "sentiment"
    MULTI_STRATEGY = "multi_strategy"


@dataclass
class PerformanceSummary:
    """Condensed performance metrics for a strategy card."""

    sharpe_ratio: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    backtest_period: str = ""
    last_updated: str = ""


@dataclass
class StrategyCard:
    """
    A shareable strategy card — the atomic unit of strategy sharing.
    Contains everything needed to replicate a strategy.
    """

    card_id: str
    name: str
    author: str
    description: str = ""
    category: StrategyCategory = StrategyCategory.MOMENTUM
    visibility: StrategyVisibility = StrategyVisibility.PRIVATE
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    parameters: dict[str, float | str] = field(default_factory=dict)
    entry_rules: list[str] = field(default_factory=list)
    exit_rules: list[str] = field(default_factory=list)
    risk_params: dict[str, float] = field(default_factory=dict)
    performance: PerformanceSummary = field(default_factory=PerformanceSummary)
    asset_classes: list[str] = field(default_factory=lambda: ["equity"])
    timeframes: list[str] = field(default_factory=lambda: ["1D"])
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        now = datetime.now(UTC).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    @property
    def content_hash(self) -> str:
        """Deterministic hash for deduplication."""
        content = (
            f"{self.name}:{self.version}:{json.dumps(self.parameters, sort_keys=True)}"
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Serialize to a shareable dictionary."""
        return {
            "card_id": self.card_id,
            "name": self.name,
            "author": self.author,
            "description": self.description,
            "category": self.category.value,
            "visibility": self.visibility.value,
            "version": self.version,
            "tags": self.tags,
            "parameters": self.parameters,
            "entry_rules": self.entry_rules,
            "exit_rules": self.exit_rules,
            "risk_params": self.risk_params,
            "performance": {
                "sharpe_ratio": self.performance.sharpe_ratio,
                "total_return": self.performance.total_return,
                "max_drawdown": self.performance.max_drawdown,
                "win_rate": self.performance.win_rate,
            },
            "asset_classes": self.asset_classes,
            "timeframes": self.timeframes,
            "content_hash": self.content_hash,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyCard":
        perf = data.get("performance", {})
        return cls(
            card_id=data.get("card_id", ""),
            name=data.get("name", ""),
            author=data.get("author", ""),
            description=data.get("description", ""),
            category=StrategyCategory(data.get("category", "momentum")),
            visibility=StrategyVisibility(data.get("visibility", "private")),
            version=data.get("version", "1.0.0"),
            tags=data.get("tags", []),
            parameters=data.get("parameters", {}),
            entry_rules=data.get("entry_rules", []),
            exit_rules=data.get("exit_rules", []),
            risk_params=data.get("risk_params", {}),
            performance=PerformanceSummary(
                sharpe_ratio=perf.get("sharpe_ratio", 0.0),
                total_return=perf.get("total_return", 0.0),
                max_drawdown=perf.get("max_drawdown", 0.0),
                win_rate=perf.get("win_rate", 0.0),
            ),
            asset_classes=data.get("asset_classes", []),
            timeframes=data.get("timeframes", []),
        )


@dataclass
class StrategyPreset:
    """A named configuration preset for a strategy."""

    preset_id: str
    name: str
    description: str = ""
    parameters: dict[str, float | str] = field(default_factory=dict)
    suitable_for: list[str] = field(default_factory=list)  # market conditions


class StrategyRegistry:
    """
    Registry for sharing, discovering, and managing strategy cards.

    Usage:
        registry = StrategyRegistry()
        registry.publish(card)
        results = registry.search(category=StrategyCategory.MOMENTUM)
    """

    def __init__(self):
        self._cards: dict[str, StrategyCard] = {}
        self._presets: dict[str, list[StrategyPreset]] = {}

    def publish(self, card: StrategyCard) -> str:
        """Publish a strategy card to the registry."""
        card.visibility = StrategyVisibility.SHARED
        card.updated_at = datetime.now(UTC).isoformat()
        self._cards[card.card_id] = card
        logger.info(f"Published strategy card: {card.name} [{card.card_id}]")
        return card.card_id

    def get(self, card_id: str) -> StrategyCard | None:
        """Get a strategy card by ID."""
        return self._cards.get(card_id)

    def search(
        self,
        category: StrategyCategory | None = None,
        tags: list[str] | None = None,
        min_sharpe: float | None = None,
        author: str | None = None,
    ) -> list[StrategyCard]:
        """Search for strategy cards matching criteria."""
        results = list(self._cards.values())

        # Only show shared/public cards
        results = [c for c in results if c.visibility != StrategyVisibility.PRIVATE]

        if category:
            results = [c for c in results if c.category == category]

        if tags:
            results = [c for c in results if any(t in c.tags for t in tags)]

        if min_sharpe is not None:
            results = [c for c in results if c.performance.sharpe_ratio >= min_sharpe]

        if author:
            results = [c for c in results if c.author == author]

        # Sort by Sharpe descending
        results.sort(key=lambda c: c.performance.sharpe_ratio, reverse=True)
        return results

    def add_preset(self, card_id: str, preset: StrategyPreset) -> None:
        """Add a configuration preset to a strategy card."""
        if card_id not in self._presets:
            self._presets[card_id] = []
        self._presets[card_id].append(preset)

    def get_presets(self, card_id: str) -> list[StrategyPreset]:
        """Get all presets for a strategy card."""
        return list(self._presets.get(card_id, []))

    def fork(self, card_id: str, new_author: str) -> StrategyCard | None:
        """Fork a strategy card — creates a copy with a new author."""
        original = self._cards.get(card_id)
        if not original:
            return None

        import uuid

        forked = StrategyCard(
            card_id=f"fork:{uuid.uuid4().hex[:8]}",
            name=f"{original.name} (Fork)",
            author=new_author,
            description=f"Forked from {original.author}'s {original.name}",
            category=original.category,
            visibility=StrategyVisibility.PRIVATE,
            tags=list(original.tags),
            parameters=dict(original.parameters),
            entry_rules=list(original.entry_rules),
            exit_rules=list(original.exit_rules),
            risk_params=dict(original.risk_params),
            asset_classes=list(original.asset_classes),
            timeframes=list(original.timeframes),
        )
        self._cards[forked.card_id] = forked
        return forked

    @property
    def card_count(self) -> int:
        return len(self._cards)

    @property
    def public_count(self) -> int:
        return sum(
            1
            for c in self._cards.values()
            if c.visibility != StrategyVisibility.PRIVATE
        )
