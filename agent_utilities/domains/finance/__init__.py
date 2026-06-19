"""
KG-Native Quantitative Finance Framework.

Provides deep learning models (LSTMs), stationary feature engineering,
walk-forward validation, execution algorithms (Kelly Criterion), alpha
factor analysis (IC/IR), risk management (VaR/Stress), portfolio
optimization (MVO/RP/BL), versioned order management, market data
abstraction, x402 AI payments, profit attribution, and universal
real-time streaming — all mapped to the Agent Intelligence Graph.
"""

from .alpha_factors import (
    FACTOR_REGISTRY,
    AlphaFactorLibrary,
    compute_factor_ic,
    compute_factor_ir,
    rank_factors,
)
from .calibration_tracker import (
    CalibrationScore,
    CalibrationTracker,
    CallRecord,
    apply_calibration_to_swarm,
    brier_score,
    calibrated_role_weights,
)
from .composite_backtest import (
    CompositeBacktester,
    CompositeBacktestResult,
    MarketAttribution,
    MarketSpec,
    run_composite_backtest,
)
from .copy_trade import (
    COPY_TRADE_AGENTS,
    CopyTradeConfig,
    CopyTradeIntent,
    CopyTradePipeline,
    aggregate_consensus,
    build_copy_trade_workflow,
    exit_check,
    position_multiplier,
    score_market,
    seed_copy_trade_workflow,
    size_position,
)
from .credit_quality import (
    CreditDividendReport,
    CreditQuality,
    DividendQuality,
    assess_credit_quality,
    assess_dividend_quality,
    attach_to_debate_context,
    emit_credit_dividend_report,
    merton_distance_to_default,
    normal_cdf,
)

try:  # torch-backed LSTM eval — optional extra; data-science-mcp owns ML training (dep rule)
    from .evaluation import evaluate_trading_signal, walk_forward_validation
except ImportError:  # noqa: BLE001 - torch absent in the lean serving install
    evaluate_trading_signal = walk_forward_validation = None  # type: ignore[assignment]
from .execution import calculate_kelly_fraction, check_regime_shift
from .features import StationaryFeatureEngineer, check_stationarity
from .filing_diff import (
    FilingDiff,
    FilingDiffAgent,
    FilingDiffFinding,
    FilingDiffResult,
    diff_filing_sections,
)
from .forensic_screener import ForensicScreener, ForensicVerdict
from .geopolitical_risk import (
    AssetExposure,
    GeopoliticalRiskFactor,
    GeopoliticalRiskScore,
    RiskCategory,
    apply_geopolitical_stress,
    asset_exposure_to_factor,
    exposed_holdings,
    geopolitical_facts_batch,
    risk_to_regime_flag,
    risk_to_stress_shocks,
    score_portfolio,
    seed_geopolitical_risk,
)
from .investor_debate import (
    DEFAULT_BEAR_PERSONA,
    DEFAULT_BULL_PERSONA,
    INVESTOR_PERSONAS,
    SPECIALIST_ROLES,
    PersonaRole,
    build_financial_debate_team,
    load_persona_prompt,
    persona_archetype,
    persona_for_role,
    persona_system_prompt,
    seed_financial_debate_team,
)
from .kronos_forecaster import (
    CandleType,
    ForecastResult,
    KLineToken,
    KLineTokenizer,
    KronosForecaster,
    KronosPredictor,
)
from .market_data import (
    DataFetchResult,
    DataRegistry,
    MarketDataProvider,
    SyntheticProvider,
    YFinanceProvider,
    normalize_ohlcv,
)
from .market_feeds import (
    FeedSubscription,
    LiveBar,
    MarketFeedBus,
    Tick,
    TickAggregator,
)

try:  # torch-backed LSTM model — optional extra; data-science-mcp owns ML training (dep rule)
    from .models import TradingLSTM, prepare_sequences
except ImportError:  # noqa: BLE001 - torch absent in the lean serving install
    TradingLSTM = prepare_sequences = None  # type: ignore[assignment,misc]
from .pattern_classifier import (
    Candle,
    EdgeLabel,
    PatternClassification,
    PatternClassifier,
    PricePattern,
)
from .payments import (
    BudgetLimits,
    PaymentChallenge,
    PaymentGuard,
    PaymentProof,
    PaymentRecord,
    PaymentStatus,
    X402PaymentClient,
)
from .persona_heuristics import (
    PERSONA_HEURISTICS,
    Heuristic,
    HeuristicResult,
    PersonaEvaluation,
    evaluate_all,
    evaluate_persona,
    list_personas,
    persona_heuristics_batch,
    seed_persona_heuristics,
)
from .portfolio_optimizer import (
    BlackLittermanOptimizer,
    MeanVarianceOptimizer,
    OptimizationResult,
    RiskParityOptimizer,
)
from .profit_attribution import (
    AttributionResult,
    BenchmarkComparison,
    PerformanceReport,
    ProfitAttributor,
    compare_to_benchmark,
    compute_performance_report,
)
from .research_autopilot import (
    AutopilotConfig,
    BacktestMetrics,
    Hypothesis,
    HypothesisResult,
    HypothesisStatus,
    ResearchAutopilot,
    ResearchReport,
    SimpleBacktester,
)
from .risk_manager import (
    PreTradeGuard,
    PreTradeResult,
    RiskLimits,
    RiskManager,
    StressTestEngine,
    StressTestResult,
    VaRCalculator,
    VaRResult,
)
from .sentiment_fusion import (
    FusedSentiment,
    SentimentObservation,
    SentimentSignal,
    fuse_sentiment,
    fused_sentiment_to_agent_signal,
    fused_sentiment_to_fusion_direction,
    ingest_and_fuse,
    normalize_observation,
    register_sentiment_source,
    score_text_polarity,
    seed_sentiment_facts,
    sentiment_facts_batch,
)
from .strategy_export import (
    ExportPlatform,
    ExportResult,
    MQL5Exporter,
    PineScriptExporter,
    StrategyCondition,
    StrategyExporter,
    StrategySpec,
    TDXExporter,
)
from .strategy_sharing import (
    PerformanceSummary,
    StrategyCard,
    StrategyCategory,
    StrategyPreset,
    StrategyRegistry,
    StrategyVisibility,
)
from .streaming import (
    CallbackSubscriber,
    StreamBus,
    StreamMessage,
    StreamSubscriber,
    WebSocketStreamAdapter,
)
from .trade_journal import (
    BiasDiagnostic,
    Roundtrip,
    TradeJournalAuditor,
    TraderProfile,
    audit_trade_journal,
)
from .trading_swarm import (
    AgentSignal,
    SwarmAgent,
    SwarmConfig,
    SwarmConsensus,
    SwarmDecision,
    SwarmRole,
    TradingSwarm,
)
from .versioned_orders import (
    OrderCommit,
    OrderHistory,
    OrderStage,
    OrderStatus,
    PreCommitGuard,
)
from .visual_ta import (
    DetectedPattern,
    PatternDetector,
    PatternType,
    SupportResistanceDetector,
    SupportResistanceLevel,
    TrendAnalysis,
    TrendDirection,
    VisualTAEngine,
)

__all__ = [
    # Investor-persona debate wiring (KG-2.6)
    "persona_system_prompt",
    "load_persona_prompt",
    "persona_archetype",
    "DEFAULT_BULL_PERSONA",
    "DEFAULT_BEAR_PERSONA",
    # Copy-Trade Pipeline (KG-2.6)
    "CopyTradePipeline",
    "CopyTradeConfig",
    "CopyTradeIntent",
    "COPY_TRADE_AGENTS",
    "score_market",
    "aggregate_consensus",
    "position_multiplier",
    "exit_check",
    "size_position",
    "build_copy_trade_workflow",
    "seed_copy_trade_workflow",
    # Original modules
    "StationaryFeatureEngineer",
    "check_stationarity",
    "TradingLSTM",
    "prepare_sequences",
    "evaluate_trading_signal",
    "walk_forward_validation",
    "calculate_kelly_fraction",
    "check_regime_shift",
    # Alpha Factors (KG-2.6)
    "AlphaFactorLibrary",
    "FACTOR_REGISTRY",
    "compute_factor_ic",
    "compute_factor_ir",
    "rank_factors",
    # Risk Management (KG-2.6)
    "RiskManager",
    "RiskLimits",
    "PreTradeGuard",
    "PreTradeResult",
    "VaRCalculator",
    "VaRResult",
    "StressTestEngine",
    "StressTestResult",
    # Portfolio Optimization (KG-2.6)
    "MeanVarianceOptimizer",
    "RiskParityOptimizer",
    "BlackLittermanOptimizer",
    "OptimizationResult",
    # Versioned Orders (KG-2.6)
    "OrderHistory",
    "OrderStage",
    "OrderCommit",
    "OrderStatus",
    "PreCommitGuard",
    # Market Data (KG-2.6)
    "MarketDataProvider",
    "YFinanceProvider",
    "SyntheticProvider",
    "DataRegistry",
    "DataFetchResult",
    "normalize_ohlcv",
    # x402 Payments (KG-2.6)
    "X402PaymentClient",
    "PaymentChallenge",
    "PaymentProof",
    "PaymentRecord",
    "PaymentGuard",
    "PaymentStatus",
    "BudgetLimits",
    # Profit Attribution (KG-2.6)
    "ProfitAttributor",
    "AttributionResult",
    "PerformanceReport",
    "BenchmarkComparison",
    "compute_performance_report",
    "compare_to_benchmark",
    # Streaming (KG-2.6)
    "StreamBus",
    "StreamMessage",
    "StreamSubscriber",
    "CallbackSubscriber",
    "WebSocketStreamAdapter",
    # Kronos Forecaster (KG-2.6)
    "KronosForecaster",
    "KLineTokenizer",
    "KLineToken",
    "KronosPredictor",
    "CandleType",
    "ForecastResult",
    # Trading Swarm (KG-2.6)
    "TradingSwarm",
    "SwarmAgent",
    "SwarmRole",
    "SwarmDecision",
    "SwarmConsensus",
    "SwarmConfig",
    "AgentSignal",
    # Visual TA (KG-2.6)
    "VisualTAEngine",
    "PatternDetector",
    "SupportResistanceDetector",
    "TrendAnalysis",
    "TrendDirection",
    "PatternType",
    "DetectedPattern",
    "SupportResistanceLevel",
    # Market Feeds (KG-2.6)
    "MarketFeedBus",
    "Tick",
    "LiveBar",
    "TickAggregator",
    "FeedSubscription",
    # Strategy Export (KG-2.6)
    "StrategyExporter",
    "PineScriptExporter",
    "MQL5Exporter",
    "TDXExporter",
    "StrategySpec",
    "StrategyCondition",
    "ExportPlatform",
    "ExportResult",
    # Research Autopilot (KG-2.6)
    "ResearchAutopilot",
    "Hypothesis",
    "HypothesisStatus",
    "BacktestMetrics",
    "HypothesisResult",
    "ResearchReport",
    "AutopilotConfig",
    "SimpleBacktester",
    # Strategy Sharing (KG-2.6)
    "StrategyRegistry",
    "StrategyCard",
    "StrategyPreset",
    "StrategyCategory",
    "StrategyVisibility",
    "PerformanceSummary",
    # Forensic Screener (KG-2.6)
    "ForensicScreener",
    "ForensicVerdict",
    # Year-over-Year Filing Diff (KG-2.6)
    "FilingDiff",
    "FilingDiffAgent",
    "FilingDiffFinding",
    "FilingDiffResult",
    "diff_filing_sections",
    # Price-Action Pattern Classifier (KG-2.6)
    "PatternClassifier",
    "PatternClassification",
    "PricePattern",
    "EdgeLabel",
    "Candle",
    # Investor Debate Team (KG-2.6)
    "INVESTOR_PERSONAS",
    "SPECIALIST_ROLES",
    "PersonaRole",
    "build_financial_debate_team",
    "seed_financial_debate_team",
    "persona_for_role",
    # Trade-Journal Bias Auditor + Shadow Account (KG-2.26)
    "TradeJournalAuditor",
    "TraderProfile",
    "Roundtrip",
    "BiasDiagnostic",
    "audit_trade_journal",
    # Agent Calibration / Reputation Tracking (KG-2.27)
    "CalibrationTracker",
    "CalibrationScore",
    "CallRecord",
    "brier_score",
    "calibrated_role_weights",
    "apply_calibration_to_swarm",
    # Persona Decision-Heuristic Enrichment (KG-2.28)
    "Heuristic",
    "HeuristicResult",
    "PersonaEvaluation",
    "PERSONA_HEURISTICS",
    "evaluate_persona",
    "evaluate_all",
    "list_personas",
    "persona_heuristics_batch",
    "seed_persona_heuristics",
    # Dividend Sustainability + Credit/Fixed-Income Quality (KG-2.31)
    "DividendQuality",
    "CreditQuality",
    "CreditDividendReport",
    "assess_dividend_quality",
    "assess_credit_quality",
    "merton_distance_to_default",
    "normal_cdf",
    "emit_credit_dividend_report",
    "attach_to_debate_context",
    # Multi-Market Composite Backtester (KG-2.32)
    "CompositeBacktester",
    "CompositeBacktestResult",
    "MarketSpec",
    "MarketAttribution",
    "run_composite_backtest",
    # Sentiment Fusion Signals (KG-2.29)
    "SentimentObservation",
    "SentimentSignal",
    "FusedSentiment",
    "score_text_polarity",
    "normalize_observation",
    "fuse_sentiment",
    "fused_sentiment_to_agent_signal",
    "register_sentiment_source",
    "fused_sentiment_to_fusion_direction",
    "sentiment_facts_batch",
    "seed_sentiment_facts",
    "ingest_and_fuse",
    # Geopolitical Risk Scoring (KG-2.30)
    "GeopoliticalRiskFactor",
    "GeopoliticalRiskScore",
    "AssetExposure",
    "RiskCategory",
    "asset_exposure_to_factor",
    "score_portfolio",
    "exposed_holdings",
    "risk_to_stress_shocks",
    "risk_to_regime_flag",
    "apply_geopolitical_stress",
    "geopolitical_facts_batch",
    "seed_geopolitical_risk",
]
