from __future__ import annotations

"""Tests for CONCEPT:AU-KG.research.research-pipeline-runner — Financial Trading Pipeline KG Primitives."""


from agent_utilities.models.knowledge_graph import (
    RegistryEdgeType,
)


class TestTradingEdgeTypes:
    def test_edge_types_exist(self):
        assert RegistryEdgeType.GENERATED_SIGNAL == "generated_signal"
        assert RegistryEdgeType.PLACED_ORDER == "placed_order"
        assert RegistryEdgeType.OPENED_POSITION == "opened_position"
        assert RegistryEdgeType.BELONGS_TO_PORTFOLIO == "belongs_to_portfolio"
        assert RegistryEdgeType.EXECUTES_STRATEGY == "executes_strategy"
        assert RegistryEdgeType.BACKTESTED_WITH == "backtested_with"
