"""CONCEPT:AU-KG.domains.finance-recommendation-entrypoint — AU-CONTEXT-R008.2/.3

Wires :func:`agent_utilities.domains.finance.analysis_snapshot.recommend` to a
real holdings/watchlist read path (:class:`HoldingsRegistry` + a real market
observation source) and exposes it as a registered MCP tool entry point
(``finance_recommend``), replacing any placeholder recommendation output
(AU-CONTEXT-R008.2).

Each strategy version's :class:`StrategyScorecard` is kept in
:class:`ScorecardStore`, a store that is constructed once and reused across
calls rather than rebuilt per call, so scorecard totals persist across
separate :func:`recommend_for_holding` invocations and remain queryable per
strategy version for cross-strategy comparison (AU-CONTEXT-R008.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from agent_utilities.domains.finance.analysis_snapshot import (
    AnalysisSnapshot,
    RecommendationAction,
    StrategyScorecard,
    recommend,
)


@dataclass(frozen=True, slots=True)
class Position:
    """A real, read-only holding or watchlist entry, keyed by holding id."""

    holding_id: str
    ticker: str
    is_watchlist: bool = False


@dataclass(frozen=True, slots=True)
class MarketObservation:
    """A real market observation, cited as recommendation evidence."""

    ticker: str
    momentum: float
    observed_at: str


class MarketObservationSource(Protocol):
    """Protocol for a real market-observation provider (e.g. ``DataRegistry``)."""

    def observe(self, ticker: str) -> MarketObservation: ...


class HoldingsRegistry:
    """Read-only positions/watchlist registry.

    Populated by the isolated paper-trading simulation account or a broker
    sync (AU-CONTEXT-R006.1); this class exposes only read access plus the
    one registration primitive used at account-build time. It is never an
    order-submission path.
    """

    def __init__(self) -> None:
        self._positions: dict[str, Position] = {}

    def register(self, position: Position) -> None:
        self._positions[position.holding_id] = position

    def get(self, holding_id: str) -> Position | None:
        return self._positions.get(holding_id)


class ScorecardStore:
    """Persists each strategy version's :class:`StrategyScorecard` across calls.

    The store is constructed once (e.g. at process/tool-registration scope)
    and reused: ``get`` returns the *same* scorecard instance on repeated
    calls for a given strategy version instead of recomputing one in memory
    each time, so totals accumulate across separate recommendations.
    """

    def __init__(self) -> None:
        self._scorecards: dict[str, StrategyScorecard] = {}

    def get(self, strategy_version: str) -> StrategyScorecard:
        if strategy_version not in self._scorecards:
            self._scorecards[strategy_version] = StrategyScorecard(
                strategy_version=strategy_version
            )
        return self._scorecards[strategy_version]

    def record(self, snapshot: AnalysisSnapshot) -> StrategyScorecard:
        scorecard = self.get(snapshot.strategy_version)
        scorecard.record(snapshot)
        return scorecard


def recommend_for_holding(
    holding_id: str,
    *,
    registry: HoldingsRegistry,
    market: MarketObservationSource,
    store: ScorecardStore,
    strategy_version: str,
    calibration_threshold: float = 0.3,
) -> AnalysisSnapshot:
    """Produce a recommendation for a real holding id and persist its scorecard.

    Reads the holding from ``registry`` (the real positions/watchlist path)
    and a real market observation from ``market``, derives a deterministic
    action/confidence from that observation, builds the
    :class:`AnalysisSnapshot` via :func:`recommend`, and records it into the
    persisted per-strategy-version scorecard in ``store``.
    """
    if (position := registry.get(holding_id)) is None:
        snapshot = recommend(
            holding_id=holding_id,
            strategy_version=strategy_version,
            confidence=0.0,
            evidence_ids=[],
            proposed_action=RecommendationAction.HOLD,
            calibration_threshold=calibration_threshold,
        )
        store.record(snapshot)
        return snapshot

    observation = market.observe(position.ticker)
    confidence = min(abs(observation.momentum), 1.0)
    if observation.momentum > 0:
        action = RecommendationAction.ACCUMULATE
    elif observation.momentum < 0:
        action = RecommendationAction.DE_RISK
    else:
        action = RecommendationAction.HOLD

    snapshot = recommend(
        holding_id=holding_id,
        strategy_version=strategy_version,
        confidence=confidence,
        evidence_ids=[
            f"position:{position.holding_id}",
            f"observation:{position.ticker}:{observation.observed_at}",
        ],
        proposed_action=action,
        calibration_threshold=calibration_threshold,
    )
    store.record(snapshot)
    return snapshot


def register_recommend_tool(
    mcp: Any,
    *,
    registry: HoldingsRegistry,
    market: MarketObservationSource,
    store: ScorecardStore,
) -> Any:
    """Register the ``finance_recommend`` MCP tool.

    This is the registered tool/agent entry point required by
    AU-CONTEXT-R008.2: it returns the real :class:`AnalysisSnapshot` built
    from the live holdings/watchlist read path, replacing any placeholder
    recommendation output.
    """

    @mcp.tool()
    def finance_recommend(
        holding_id: str, strategy_version: str = "momentum-v1"
    ) -> AnalysisSnapshot:
        return recommend_for_holding(
            holding_id,
            registry=registry,
            market=market,
            store=store,
            strategy_version=strategy_version,
        )

    return finance_recommend
