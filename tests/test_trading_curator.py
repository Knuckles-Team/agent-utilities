"""Tests for the trading-knowledge curator — CONCEPT:AU-AHE.assimilation.trading-curator."""

from agent_utilities.knowledge_graph.distillation.trading_curator import (
    build_knowledge_nodes,
    classify_trading_concept,
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
