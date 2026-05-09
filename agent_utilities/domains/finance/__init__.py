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
from .evaluation import evaluate_trading_signal, walk_forward_validation
from .execution import calculate_kelly_fraction, check_regime_shift
from .features import StationaryFeatureEngineer, check_stationarity
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
from .models import TradingLSTM, prepare_sequences
from .payments import (
    BudgetLimits,
    PaymentChallenge,
    PaymentGuard,
    PaymentProof,
    PaymentRecord,
    PaymentStatus,
    X402PaymentClient,
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
    # Original modules
    "StationaryFeatureEngineer",
    "check_stationarity",
    "TradingLSTM",
    "prepare_sequences",
    "evaluate_trading_signal",
    "walk_forward_validation",
    "calculate_kelly_fraction",
    "check_regime_shift",
    # Alpha Factors (KG-2.60)
    "AlphaFactorLibrary",
    "FACTOR_REGISTRY",
    "compute_factor_ic",
    "compute_factor_ir",
    "rank_factors",
    # Risk Management (KG-2.61)
    "RiskManager",
    "RiskLimits",
    "PreTradeGuard",
    "PreTradeResult",
    "VaRCalculator",
    "VaRResult",
    "StressTestEngine",
    "StressTestResult",
    # Portfolio Optimization (KG-2.62)
    "MeanVarianceOptimizer",
    "RiskParityOptimizer",
    "BlackLittermanOptimizer",
    "OptimizationResult",
    # Versioned Orders (KG-2.63)
    "OrderHistory",
    "OrderStage",
    "OrderCommit",
    "OrderStatus",
    "PreCommitGuard",
    # Market Data (KG-2.64)
    "MarketDataProvider",
    "YFinanceProvider",
    "SyntheticProvider",
    "DataRegistry",
    "DataFetchResult",
    "normalize_ohlcv",
    # x402 Payments (KG-2.65)
    "X402PaymentClient",
    "PaymentChallenge",
    "PaymentProof",
    "PaymentRecord",
    "PaymentGuard",
    "PaymentStatus",
    "BudgetLimits",
    # Profit Attribution (KG-2.66)
    "ProfitAttributor",
    "AttributionResult",
    "PerformanceReport",
    "BenchmarkComparison",
    "compute_performance_report",
    "compare_to_benchmark",
    # Streaming (KG-2.67)
    "StreamBus",
    "StreamMessage",
    "StreamSubscriber",
    "CallbackSubscriber",
    "WebSocketStreamAdapter",
    # Kronos Forecaster (KG-2.70)
    "KronosForecaster",
    "KLineTokenizer",
    "KLineToken",
    "KronosPredictor",
    "CandleType",
    "ForecastResult",
    # Trading Swarm (KG-2.71)
    "TradingSwarm",
    "SwarmAgent",
    "SwarmRole",
    "SwarmDecision",
    "SwarmConsensus",
    "SwarmConfig",
    "AgentSignal",
    # Visual TA (KG-2.72)
    "VisualTAEngine",
    "PatternDetector",
    "SupportResistanceDetector",
    "TrendAnalysis",
    "TrendDirection",
    "PatternType",
    "DetectedPattern",
    "SupportResistanceLevel",
    # Market Feeds (KG-2.73)
    "MarketFeedBus",
    "Tick",
    "LiveBar",
    "TickAggregator",
    "FeedSubscription",
    # Strategy Export (KG-2.74)
    "StrategyExporter",
    "PineScriptExporter",
    "MQL5Exporter",
    "TDXExporter",
    "StrategySpec",
    "StrategyCondition",
    "ExportPlatform",
    "ExportResult",
    # Research Autopilot (KG-2.75)
    "ResearchAutopilot",
    "Hypothesis",
    "HypothesisStatus",
    "BacktestMetrics",
    "HypothesisResult",
    "ResearchReport",
    "AutopilotConfig",
    "SimpleBacktester",
    # Strategy Sharing (KG-2.76)
    "StrategyRegistry",
    "StrategyCard",
    "StrategyPreset",
    "StrategyCategory",
    "StrategyVisibility",
    "PerformanceSummary",
]
