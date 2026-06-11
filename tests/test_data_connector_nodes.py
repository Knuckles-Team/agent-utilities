from __future__ import annotations

"""Tests for the data-connector KG node models (DataConnectorNode et al.).

The row-oriented ``DataConnectorRegistry`` runtime was strangled (zero live
callers); the KG ontology surface — ``DataConnectorNode``,
``DataFetchRecordNode``, and the ``fetched_from`` / ``falls_back_to`` edge
types — remains live (owl_bridge, archimate_layer, standardization, hydration).
"""


from agent_utilities.models.knowledge_graph import (
    DataConnectorNode,
    DataFetchRecordNode,
    RegistryEdgeType,
    RegistryNodeType,
)


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
        from agent_utilities.knowledge_graph.core.graph_compute import (
            GraphComputeEngine,
        )

        g = GraphComputeEngine(backend_type="rust")
        for name, priority in [("yahoo", 0), ("polygon", 1), ("alpha", 2)]:
            node = DataConnectorNode(id=f"dc:{name}", name=name, priority=priority)
            g.add_node(node.id, **node.model_dump())

        g.add_edge("dc:yahoo", "dc:polygon", type=RegistryEdgeType.FALLS_BACK_TO)
        g.add_edge("dc:polygon", "dc:alpha", type=RegistryEdgeType.FALLS_BACK_TO)

        # Verify transitive fallback chain
        path = g.get_shortest_path("dc:yahoo", "dc:alpha")
        assert path is not None
        assert len(path) == 3
