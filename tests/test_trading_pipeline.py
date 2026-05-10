from __future__ import annotations
"""Tests for CONCEPT:KG-2.6 — Financial Trading Pipeline KG Primitives."""


import pytest

from agent_utilities.models.knowledge_graph import (
    BacktestRunNode,
    OrderNode,
    PortfolioNode,
    PositionNode,
    RegistryEdgeType,
    RegistryNodeType,
    StrategyNode,
    TradingSignalNode,
)


class TestTradingSignalNode:
    def test_creation_defaults(self):
        node = TradingSignalNode(id="sig:001", name="Buy AAPL")
        assert node.type == RegistryNodeType.TRADING_SIGNAL
        assert node.signal_type == "hold"
        assert node.confidence == 0.5

    def test_full_creation(self):
        node = TradingSignalNode(
            id="sig:002",
            name="Strong Buy MSFT",
            signal_type="buy",
            confidence=0.95,
            instrument_id="inst:MSFT",
            attribution="technical",
            price_at_signal=420.50,
            expiry="2024-12-31T23:59:59Z",
        )
        assert node.signal_type == "buy"
        assert node.confidence == 0.95
        assert node.instrument_id == "inst:MSFT"
        assert node.price_at_signal == 420.50

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            TradingSignalNode(id="sig:bad", name="Bad", confidence=1.5)
        with pytest.raises(Exception):
            TradingSignalNode(id="sig:bad2", name="Bad", confidence=-0.1)

    def test_serialization(self):
        node = TradingSignalNode(
            id="sig:003", name="Test", signal_type="sell", confidence=0.8
        )
        data = node.model_dump()
        restored = TradingSignalNode.model_validate(data)
        assert restored.signal_type == "sell"
        assert restored.confidence == 0.8


class TestOrderNode:
    def test_creation_defaults(self):
        node = OrderNode(id="ord:001", name="Market Buy")
        assert node.type == RegistryNodeType.ORDER
        assert node.order_type == "market"
        assert node.side == "buy"
        assert node.status == "pending"

    def test_full_lifecycle(self):
        node = OrderNode(
            id="ord:002",
            name="Limit Buy AAPL",
            order_type="limit",
            side="buy",
            quantity=100,
            price=150.00,
            filled_price=149.95,
            status="filled",
            instrument_id="inst:AAPL",
            exchange="NASDAQ",
            submitted_at="2024-06-01T10:00:00Z",
            filled_at="2024-06-01T10:00:05Z",
        )
        assert node.filled_price == 149.95
        assert node.status == "filled"

    def test_serialization(self):
        node = OrderNode(id="ord:003", name="Test", quantity=50, price=200.0)
        data = node.model_dump()
        restored = OrderNode.model_validate(data)
        assert restored.quantity == 50
        assert restored.price == 200.0


class TestPositionNode:
    def test_creation(self):
        node = PositionNode(
            id="pos:001",
            name="AAPL Long",
            instrument_id="inst:AAPL",
            side="long",
            quantity=100,
            entry_price=150.00,
            current_price=155.00,
            unrealized_pnl=500.00,
        )
        assert node.type == RegistryNodeType.POSITION
        assert node.side == "long"
        assert node.unrealized_pnl == 500.00
        assert node.status == "open"

    def test_closed_position(self):
        node = PositionNode(
            id="pos:002",
            name="MSFT Short",
            side="short",
            quantity=50,
            entry_price=300.00,
            exit_price=290.00,
            realized_pnl=500.00,
            status="closed",
        )
        assert node.status == "closed"
        assert node.realized_pnl == 500.00


class TestPortfolioNode:
    def test_creation(self):
        node = PortfolioNode(
            id="port:001",
            name="Growth Portfolio",
            total_value=1_000_000.0,
            cash_balance=50_000.0,
            position_count=15,
            total_pnl=25_000.0,
            allocation_weights={"AAPL": 0.2, "MSFT": 0.15, "GOOG": 0.1},
            benchmark_id="bench:SP500",
        )
        assert node.type == RegistryNodeType.PORTFOLIO
        assert node.allocation_weights["AAPL"] == 0.2
        assert node.position_count == 15


class TestStrategyNode:
    def test_creation(self):
        node = StrategyNode(
            id="strat:001",
            name="Momentum Factor V2",
            strategy_type="momentum",
            parameters={"lookback": 20, "threshold": 0.02},
            version=2,
            sharpe_ratio=1.45,
            max_drawdown=0.12,
            win_rate=0.58,
            universes=["US_LARGE_CAP", "EU_STOXX"],
        )
        assert node.type == RegistryNodeType.STRATEGY
        assert node.strategy_type == "momentum"
        assert node.sharpe_ratio == 1.45
        assert len(node.universes) == 2

    def test_serialization(self):
        node = StrategyNode(
            id="strat:002",
            name="Test",
            parameters={"alpha": 0.5},
        )
        data = node.model_dump()
        restored = StrategyNode.model_validate(data)
        assert restored.parameters["alpha"] == 0.5


class TestTradingEdgeTypes:
    def test_edge_types_exist(self):
        assert RegistryEdgeType.GENERATED_SIGNAL == "generated_signal"
        assert RegistryEdgeType.PLACED_ORDER == "placed_order"
        assert RegistryEdgeType.OPENED_POSITION == "opened_position"
        assert RegistryEdgeType.BELONGS_TO_PORTFOLIO == "belongs_to_portfolio"
        assert RegistryEdgeType.EXECUTES_STRATEGY == "executes_strategy"
        assert RegistryEdgeType.BACKTESTED_WITH == "backtested_with"


class TestTradingPipelineIntegration:
    """Test the full trading pipeline graph: Strategy → Signal → Order → Position → Portfolio."""

    def test_full_pipeline_graph(self):
        import networkx as nx

        g = nx.MultiDiGraph()

        strat = StrategyNode(id="strat:mom", name="Momentum", strategy_type="momentum")
        signal = TradingSignalNode(
            id="sig:buy_aapl", name="Buy AAPL", signal_type="buy", confidence=0.9
        )
        order = OrderNode(
            id="ord:001", name="Market Buy AAPL", quantity=100, status="filled"
        )
        pos = PositionNode(
            id="pos:001", name="AAPL Long", entry_price=150.0, quantity=100
        )
        port = PortfolioNode(id="port:main", name="Main", position_count=1)

        # Add nodes
        for n in [strat, signal, order, pos, port]:
            g.add_node(n.id, **n.model_dump())

        # Add pipeline edges
        g.add_edge(strat.id, signal.id, type=RegistryEdgeType.GENERATED_SIGNAL)
        g.add_edge(signal.id, order.id, type=RegistryEdgeType.PLACED_ORDER)
        g.add_edge(order.id, pos.id, type=RegistryEdgeType.OPENED_POSITION)
        g.add_edge(pos.id, port.id, type=RegistryEdgeType.BELONGS_TO_PORTFOLIO)
        g.add_edge(port.id, strat.id, type=RegistryEdgeType.EXECUTES_STRATEGY)

        # Verify graph structure
        assert g.number_of_nodes() == 5
        assert g.number_of_edges() == 5

        # Verify traversal: Strategy → Signal → Order → Position → Portfolio
        path = nx.shortest_path(g, strat.id, port.id)
        assert len(path) == 5
        assert path[0] == strat.id
        assert path[-1] == port.id

    def test_strategy_backtest_link(self):
        import networkx as nx

        g = nx.MultiDiGraph()
        strat = StrategyNode(id="strat:v1", name="V1")
        bt = BacktestRunNode(id="bt:001", name="Backtest V1", strategy_id="strat:v1")
        g.add_node(strat.id, **strat.model_dump())
        g.add_node(bt.id, **bt.model_dump())
        g.add_edge(strat.id, bt.id, type=RegistryEdgeType.BACKTESTED_WITH)

        assert g.has_edge(strat.id, bt.id)
