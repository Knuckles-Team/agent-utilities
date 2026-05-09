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


class StrategyCardEntityNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.STRATEGY_CARD_ENTITY
    author: str = ""
    status: str = "draft"
    review_score: float = 0.0
