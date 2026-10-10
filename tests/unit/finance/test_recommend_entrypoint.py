"""AU-CONTEXT-R008.2/.3: a registered entry point produces recommendations
from real holdings, and per-strategy scorecards persist for comparison.

AU-CONTEXT-R008.2: ``register_recommend_tool`` wires
:func:`recommend_for_holding` to the real holdings/watchlist read path
(:class:`HoldingsRegistry` + a market observation source) and a registered
MCP tool entry point, replacing any placeholder recommendation output.

AU-CONTEXT-R008.3: :class:`ScorecardStore` persists each strategy version's
``StrategyScorecard`` across separate ``recommend_for_holding`` calls rather
than recomputing it in memory per call, and it stays queryable per strategy
version.
"""

from __future__ import annotations

import pytest

from agent_utilities.domains.finance.analysis_snapshot import (
    AnalysisSnapshot,
    RecommendationAction,
)
from agent_utilities.domains.finance.recommendation_entrypoint import (
    HoldingsRegistry,
    MarketObservation,
    Position,
    ScorecardStore,
    recommend_for_holding,
    register_recommend_tool,
)


class _FakeMarket:
    """A real-shaped market observation source with a fixed momentum reading."""

    def __init__(self, momentum: float) -> None:
        self._momentum = momentum

    def observe(self, ticker: str) -> MarketObservation:
        return MarketObservation(
            ticker=ticker, momentum=self._momentum, observed_at="2026-10-09T00:00:00Z"
        )


class _CollectingMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def _deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return _deco


@pytest.mark.spec("AU-CONTEXT-R008.2")
def test_registered_entry_point_returns_snapshot_for_real_holding() -> None:
    registry = HoldingsRegistry()
    registry.register(Position(holding_id="acct-1:AAPL", ticker="AAPL"))
    tool = register_recommend_tool(
        _CollectingMCP(),
        registry=registry,
        market=_FakeMarket(momentum=0.4),
        store=ScorecardStore(),
    )

    snapshot = tool(holding_id="acct-1:AAPL", strategy_version="momentum-v1")

    assert isinstance(snapshot, AnalysisSnapshot)
    assert snapshot.holding_id == "acct-1:AAPL"
    assert snapshot.strategy_version == "momentum-v1"
    assert snapshot.action is RecommendationAction.ACCUMULATE
    assert snapshot.evidence_ids
    assert snapshot.informational_only is True


@pytest.mark.spec("AU-CONTEXT-R008.2")
def test_unknown_holding_abstains_rather_than_fabricating_a_position() -> None:
    tool = register_recommend_tool(
        _CollectingMCP(),
        registry=HoldingsRegistry(),
        market=_FakeMarket(momentum=0.9),
        store=ScorecardStore(),
    )

    snapshot = tool(holding_id="does-not-exist", strategy_version="momentum-v1")

    assert snapshot.action is RecommendationAction.ABSTAIN


@pytest.mark.spec("AU-CONTEXT-R008.3")
def test_scorecard_persists_across_separate_recommend_calls() -> None:
    registry = HoldingsRegistry()
    registry.register(Position(holding_id="acct-1:AAPL", ticker="AAPL"))
    registry.register(Position(holding_id="acct-1:TSLA", ticker="TSLA"))
    store = ScorecardStore()

    recommend_for_holding(
        "acct-1:AAPL",
        registry=registry,
        market=_FakeMarket(momentum=0.5),
        store=store,
        strategy_version="momentum-v1",
    )
    recommend_for_holding(
        "acct-1:TSLA",
        registry=registry,
        market=_FakeMarket(momentum=-0.5),
        store=store,
        strategy_version="momentum-v1",
    )

    scorecard = store.get("momentum-v1")
    assert scorecard.total == 2
    assert scorecard.by_action == {"accumulate": 1, "de_risk": 1}
    # The same persisted instance is returned on a later, separate query --
    # it was not recomputed in memory for this call.
    assert store.get("momentum-v1") is scorecard


@pytest.mark.spec("AU-CONTEXT-R008.3")
def test_scorecards_are_queryable_independently_per_strategy_version() -> None:
    registry = HoldingsRegistry()
    registry.register(Position(holding_id="acct-1:AAPL", ticker="AAPL"))
    store = ScorecardStore()

    recommend_for_holding(
        "acct-1:AAPL",
        registry=registry,
        market=_FakeMarket(momentum=0.5),
        store=store,
        strategy_version="momentum-v1",
    )
    recommend_for_holding(
        "acct-1:AAPL",
        registry=registry,
        market=_FakeMarket(momentum=0.5),
        store=store,
        strategy_version="mean-reversion-v1",
    )

    assert store.get("momentum-v1").total == 1
    assert store.get("mean-reversion-v1").total == 1
