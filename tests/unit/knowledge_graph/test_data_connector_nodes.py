from __future__ import annotations

"""Tests for the data-connector KG edge types.

The row-oriented ``DataConnectorRegistry`` runtime was strangled (zero live
callers) and its unused Pydantic node models were deleted (EH-380); the
``fetched_from`` / ``falls_back_to`` edge types remain live (owl_bridge,
archimate_layer, standardization, hydration).
"""


from agent_utilities.models.knowledge_graph import (
    RegistryEdgeType,
)


class TestDataConnectorKGNodes:
    def test_edge_types(self):
        assert RegistryEdgeType.FETCHED_FROM == "fetched_from"
        assert RegistryEdgeType.FALLS_BACK_TO == "falls_back_to"
