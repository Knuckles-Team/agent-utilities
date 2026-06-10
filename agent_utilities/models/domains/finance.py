from ...models.knowledge_graph import RegistryNode, RegistryNodeType


class TradingStrategyNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.TRADING_STRATEGY
    asset_class: str = "equities"
    timeframe: str = "1d"
    expected_alpha: float = 0.0


class TradingSwarmEntityNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.TRADING_SWARM_ENTITY
    swarm_size: int = 3
    primary_strategy_id: str = ""


class PortfolioAllocationNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.PORTFOLIO_ALLOCATION
    target_weight: float = 0.0
    current_weight: float = 0.0
    asset_ticker: str = ""


class MarketDataSourceNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.MARKET_DATA_SOURCE
    provider_name: str = ""
    data_type: str = "price"
    update_frequency_ms: int = 1000


class MarketRegimeNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.MARKET_REGIME
    regime_type: str = "bull_volatile"
    confidence: float = 0.0


class ExecutionSignalNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.EXECUTION_SIGNAL
    direction: str = "buy"
    strength: float = 1.0
    target_price: float = 0.0


class OrderCommitRecordNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.ORDER_COMMIT_RECORD
    order_id: str = ""
    status: str = "filled"
    fill_price: float = 0.0


class OrderVersionNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.ORDER_VERSION
    version_number: int = 1
    modification_reason: str = ""


class KellySizingNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.KELLY_SIZING
    win_probability: float = 0.0
    win_loss_ratio: float = 0.0
    suggested_fraction: float = 0.0


class VaREstimateNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.VAR_ESTIMATE
    confidence_interval: float = 0.95
    time_horizon_days: int = 1
    estimated_loss: float = 0.0


class MarkovRegimeStateNode(RegistryNode):
    """A detected market regime observation.

    CONCEPT:KG-2.6 — Markov Regime Detection
    """

    type: RegistryNodeType = RegistryNodeType.MARKOV_REGIME_STATE
    regime_type: str = "sideways"  # bull, bear, sideways
    confidence: float = 0.0
    returns_window: int = 20
    asset_class: str = "equities"


class MarkovTransitionMatrixNode(RegistryNode):
    """A serialized Markov transition matrix.

    CONCEPT:KG-2.6 — Markov Regime Forecasting
    """

    type: RegistryNodeType = RegistryNodeType.MARKOV_TRANSITION_MATRIX
    dimension: int = 3
    matrix_json: str = "{}"
    sample_count: int = 0
    estimation_window: int = 252
    asset_class: str = "equities"


class RegimeSignalNode(RegistryNode):
    """A generated trading signal from regime probabilities.

    CONCEPT:KG-2.6 — Regime-Based Signal Generation
    """

    type: RegistryNodeType = RegistryNodeType.REGIME_SIGNAL
    signal_value: float = 0.0
    bull_prob: float = 0.0
    bear_prob: float = 0.0
    sideways_prob: float = 0.0


class StrategyCardEntityNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.STRATEGY_CARD_ENTITY
    author: str = ""
    status: str = "draft"
    review_score: float = 0.0


class MicrostructureSignalNode(RegistryNode):
    """A short-horizon microstructure alpha with measured statistical priors.

    CONCEPT:EE-033 — the priors (directional accuracy, standalone Sharpe, decay)
    are written back by the backtester and drive signal-fusion weights, so the
    fusion self-adjusts as evidence accumulates. ``provenance`` cites the backtest
    run id or the book/paper the signal was distilled from.
    """

    type: RegistryNodeType = RegistryNodeType.MICROSTRUCTURE_SIGNAL
    prediction_horizon: str = "1m"  # e.g. "30s", "1m", "5m"
    directional_accuracy: float = 0.5  # measured prior ∈ [0, 1]
    standalone_sharpe: float = 0.0  # measured prior (deflated Sharpe)
    pbo: float = 0.0  # probability of backtest overfit (last validation)
    decay_regime: str = "stationary"  # stationary | decaying | regime_dependent
    half_life_days: float = 0.0
    provenance: str = ""  # backtest run id / book citation
    asset_class: str = "equities"
    last_validated: str | None = None  # ISO timestamp of last priors write-back


class _TradingKnowledgeBase(RegistryNode):
    """Shared shape for knowledge distilled & classified from books/PDFs/notes.

    CONCEPT:EE-036 — concrete subclasses (Strategy/Risk/Execution concept)
    organise curated knowledge into typed, queryable nodes with citations and an
    extraction-confidence score, rather than leaving it as a verbatim document.
    """

    topic: str = ""
    source: str = ""  # book / paper / PDF title
    chapter: str = ""
    page_span: str = ""
    confidence: float = 0.5  # extraction confidence ∈ [0, 1]


class StrategyConceptNode(_TradingKnowledgeBase):
    type: RegistryNodeType = RegistryNodeType.STRATEGY_CONCEPT


class RiskConceptNode(_TradingKnowledgeBase):
    type: RegistryNodeType = RegistryNodeType.RISK_CONCEPT


class ExecutionConceptNode(_TradingKnowledgeBase):
    type: RegistryNodeType = RegistryNodeType.EXECUTION_CONCEPT


class TradeJournalNode(RegistryNode):
    """A single trade decision recorded for the feedback loop.

    CONCEPT:EE-034 — the expert agent writes one per decision; a nightly distill
    pass promotes recurring profitable patterns into reusable strategy concepts.
    """

    type: RegistryNodeType = RegistryNodeType.TRADE_JOURNAL
    instrument: str = ""
    stage: str = "paper"  # paper | advisory | bounded_autonomous
    direction: str = "hold"  # buy | sell | hold
    size: float = 0.0
    signals_used: list[str] = []  # MicrostructureSignal ids fused for this decision
    priors_snapshot: str = "{}"  # JSON of the priors at decision time
    rationale: str = ""
    regime_id: str = ""
    outcome_pnl: float = 0.0
    status: str = "open"  # open | closed | cancelled
