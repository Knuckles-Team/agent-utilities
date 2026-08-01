"""Tests for the trading-knowledge curator — CONCEPT:AU-AHE.assimilation.trading-curator."""

import asyncio

from agent_utilities.knowledge_graph.distillation.trading_curator import (
    build_knowledge_nodes,
    classify_trading_concept,
    organize_trading_knowledge,
)


class TestClassifyTradingConcept:
    def test_execution_microstructure(self):
        cat, conf = classify_trading_concept(
            "Order book imbalance and queue position predict short-horizon fills."
        )
        assert cat == "execution" and conf > 0.0

    def test_risk(self):
        cat, conf = classify_trading_concept(
            "Position sizing via the Kelly criterion controls drawdown and risk of ruin."
        )
        assert cat == "risk" and conf > 0.0

    def test_strategy(self):
        cat, _ = classify_trading_concept(
            "A momentum factor captures trend; backtest the alpha before trading."
        )
        assert cat == "strategy"

    def test_non_trading_text_skipped(self):
        cat, conf = classify_trading_concept("The cat sat on the mat in the sun.")
        assert cat is None and conf == 0.0


class TestBuildKnowledgeNodes:
    def test_classifies_and_seeds_microstructure(self):
        concepts = [
            {
                "id": "c1",
                "text": "Order flow imbalance in the limit order book signals short-term direction.",
                "chapter": "3",
            },
            {
                "id": "c2",
                "text": "Use volatility-targeted position sizing to cap drawdown.",
                "chapter": "5",
            },
            {"id": "c3", "text": "Completely unrelated prose about gardening."},
        ]
        out = build_knowledge_nodes(concepts, source_title="Trading and Exchanges")

        # c1 -> execution + a microstructure signal seed; c2 -> risk; c3 -> skipped
        assert {n.topic for n in out["knowledge"]} == {"execution", "risk"}
        assert len(out["signals"]) == 1
        sig = out["signals"][0]
        assert sig.id == "sig:book:c1"
        assert "Trading and Exchanges" in sig.provenance
        assert "c3" in out["skipped"]
        # provenance + citation preserved on the knowledge node
        exec_node = next(n for n in out["knowledge"] if n.topic == "execution")
        assert exec_node.source == "Trading and Exchanges" and exec_node.chapter == "3"


class _FakeNodes:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def add(self, node_id, properties=None):
        self.calls.append((node_id, properties))


class _FakeEdges:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def add(self, source_id, target_id, properties=None):
        # Mirrors the real epistemic_graph.client.EdgeClient.add contract:
        # `properties` must be a dict (or None), never a bare relationship
        # string — a str here is a caller bug (D-W2N-2).
        if properties is not None and not isinstance(properties, dict):
            raise TypeError(
                f"properties must be a dict or None, got {type(properties).__name__}"
            )
        self.calls.append((source_id, target_id, properties))


class _FakeClient:
    def __init__(self) -> None:
        self.nodes = _FakeNodes()
        self.edges = _FakeEdges()


class TestOrganizeTradingKnowledgeWritesDerivedFromEdge:
    def test_derived_from_edge_is_a_properties_dict_not_a_bare_string(self):
        # D-W2N-2: organize_trading_knowledge used to call
        # client.edges.add(source, target, "DERIVED_FROM") — a bare string in
        # the properties slot. A real client TypeErrors on that (caught by
        # the surrounding best-effort except and only logged at debug), so
        # the DERIVED_FROM provenance edge silently never landed. Assert the
        # edge write now succeeds and carries a proper relationship dict.
        concepts = [
            {
                "id": "c1",
                "text": "Order flow imbalance in the limit order book signals short-term direction.",
                "chapter": "3",
            }
        ]
        client = _FakeClient()
        result = asyncio.run(
            organize_trading_knowledge(client, concepts, source_title="Book")
        )

        assert result["signal_seeds"] == 1
        assert client.edges.calls, "expected a DERIVED_FROM edge write attempt"
        source_id, target_id, properties = client.edges.calls[0]
        assert source_id == "sig:book:c1"
        assert target_id == "tk:execution:c1"
        assert properties == {"relationship": "DERIVED_FROM"}
