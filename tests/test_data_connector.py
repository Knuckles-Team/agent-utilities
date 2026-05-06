"""Tests for CONCEPT:ECO-4.4 — Market Data Connector Protocol."""

from __future__ import annotations

from agent_utilities.models.knowledge_graph import (
    DataConnectorNode,
    DataFetchRecordNode,
    RegistryEdgeType,
    RegistryNodeType,
)
from agent_utilities.protocols.data_connector import (
    DataConnectorRegistry,
    DataFetchResult,
)


class FakeConnector:
    """Test connector implementing the DataConnectorProtocol."""

    def __init__(self, name: str, priority: int = 0, healthy: bool = True, data: list | None = None):
        self.name = name
        self.provider = f"fake_{name}"
        self.priority = priority
        self.supported_instruments = ["equity", "etf"]
        self._healthy = healthy
        self._data = data or []

    def fetch(self, query: str, **kwargs) -> DataFetchResult:
        if not self._data:
            return DataFetchResult(query=query, error="no data")
        return DataFetchResult(
            rows=[{"symbol": query, "price": d} for d in self._data],
            row_count=len(self._data),
            query=query,
        )

    def health_check(self) -> bool:
        return self._healthy


class FailingConnector(FakeConnector):
    """Connector that always raises an exception."""

    def fetch(self, query: str, **kwargs) -> DataFetchResult:
        raise ConnectionError("Network error")


class TestDataFetchResult:
    def test_defaults(self):
        r = DataFetchResult()
        assert r.row_count == 0
        assert r.rows == []
        assert r.is_fallback is False
        assert r.fetched_at  # auto-generated timestamp

    def test_with_data(self):
        r = DataFetchResult(
            rows=[{"a": 1}], row_count=1, connector_name="test"
        )
        assert r.row_count == 1
        assert r.connector_name == "test"


class TestDataConnectorRegistry:
    def test_register(self):
        reg = DataConnectorRegistry()
        c = FakeConnector("alpha", priority=1)
        reg.register(c)
        assert len(reg.list_connectors()) == 1
        assert reg.list_connectors()[0]["name"] == "alpha"

    def test_register_maintains_priority_order(self):
        reg = DataConnectorRegistry()
        reg.register(FakeConnector("high", priority=10))
        reg.register(FakeConnector("low", priority=1))
        reg.register(FakeConnector("mid", priority=5))
        names = [c["name"] for c in reg.list_connectors()]
        assert names == ["low", "mid", "high"]

    def test_unregister(self):
        reg = DataConnectorRegistry()
        reg.register(FakeConnector("test"))
        assert reg.unregister("test") is True
        assert reg.unregister("nonexistent") is False
        assert len(reg.list_connectors()) == 0

    def test_get_connector(self):
        reg = DataConnectorRegistry()
        c = FakeConnector("find_me")
        reg.register(c)
        assert reg.get_connector("find_me") is c
        assert reg.get_connector("missing") is None

    def test_fetch_success(self):
        reg = DataConnectorRegistry()
        reg.register(FakeConnector("primary", data=[100.0, 101.0]))
        result = reg.fetch_with_fallback("AAPL")
        assert result.row_count == 2
        assert result.connector_name == "primary"
        assert result.is_fallback is False
        assert result.error is None

    def test_fetch_fallback(self):
        reg = DataConnectorRegistry()
        reg.register(FakeConnector("primary", priority=0, data=[]))  # empty result
        reg.register(FakeConnector("secondary", priority=1, data=[200.0]))
        result = reg.fetch_with_fallback("AAPL")
        assert result.row_count == 1
        assert result.connector_name == "secondary"
        assert result.is_fallback is True

    def test_fetch_skips_unhealthy(self):
        reg = DataConnectorRegistry()
        reg.register(FakeConnector("sick", priority=0, healthy=False, data=[100.0]))
        reg.register(FakeConnector("healthy", priority=1, data=[200.0]))
        result = reg.fetch_with_fallback("AAPL")
        assert result.connector_name == "healthy"

    def test_fetch_handles_exceptions(self):
        reg = DataConnectorRegistry()
        reg.register(FailingConnector("broken", priority=0))
        reg.register(FakeConnector("backup", priority=1, data=[300.0]))
        result = reg.fetch_with_fallback("AAPL")
        assert result.row_count == 1
        assert result.connector_name == "backup"

    def test_fetch_all_fail(self):
        reg = DataConnectorRegistry()
        reg.register(FailingConnector("broken1", priority=0))
        reg.register(FailingConnector("broken2", priority=1))
        result = reg.fetch_with_fallback("AAPL")
        assert result.error is not None
        assert "All connectors failed" in result.error

    def test_instrument_type_filter(self):
        reg = DataConnectorRegistry()
        c = FakeConnector("equity_only", data=[100.0])
        c.supported_instruments = ["equity"]
        reg.register(c)
        result = reg.fetch_with_fallback("AAPL", instrument_type="equity")
        assert result.row_count == 1
        result2 = reg.fetch_with_fallback("BTC", instrument_type="crypto")
        assert result2.error is not None

    def test_fetch_history(self):
        reg = DataConnectorRegistry()
        reg.register(FakeConnector("tracker", data=[100.0]))
        reg.fetch_with_fallback("AAPL")
        reg.fetch_with_fallback("MSFT")
        assert len(reg.fetch_history) == 2
        reg.clear_history()
        assert len(reg.fetch_history) == 0


class TestDataConnectorKGNodes:
    def test_connector_node(self):
        node = DataConnectorNode(
            id="dc:yahoo",
            name="Yahoo Finance",
            connector_type="market_data",
            provider="yahoo_finance",
            base_url="https://api.yahoo.com",
            rate_limit_rpm=100,
            supported_instruments=["equity", "etf", "index"],
            priority=0,
        )
        assert node.type == RegistryNodeType.DATA_CONNECTOR
        assert node.is_healthy is True
        assert node.provider == "yahoo_finance"

    def test_fetch_record_node(self):
        node = DataFetchRecordNode(
            id="fetch:001",
            name="AAPL Fetch",
            connector_id="dc:yahoo",
            query="AAPL",
            row_count=252,
            latency_ms=150.5,
            status_code=200,
        )
        assert node.type == RegistryNodeType.DATA_FETCH_RECORD
        assert node.row_count == 252

    def test_edge_types(self):
        assert RegistryEdgeType.FETCHED_FROM == "fetched_from"
        assert RegistryEdgeType.FALLS_BACK_TO == "falls_back_to"

    def test_fallback_chain_graph(self):
        import networkx as nx

        g = nx.MultiDiGraph()
        for name, priority in [("yahoo", 0), ("polygon", 1), ("alpha", 2)]:
            node = DataConnectorNode(
                id=f"dc:{name}", name=name, priority=priority
            )
            g.add_node(node.id, **node.model_dump())

        g.add_edge("dc:yahoo", "dc:polygon", type=RegistryEdgeType.FALLS_BACK_TO)
        g.add_edge("dc:polygon", "dc:alpha", type=RegistryEdgeType.FALLS_BACK_TO)

        # Verify transitive fallback chain
        path = nx.shortest_path(g, "dc:yahoo", "dc:alpha")
        assert len(path) == 3
