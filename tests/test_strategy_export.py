"""Tests for CONCEPT:AU-KG.research.research-pipeline-runner — Multi-Platform Strategy Export."""

import pytest

from agent_utilities.domains.finance.strategy_export import (
    ExportPlatform,
    MQL5Exporter,
    PineScriptExporter,
    StrategyCondition,
    StrategyExporter,
    StrategySpec,
    TDXExporter,
)


@pytest.fixture
def simple_strategy():
    return StrategySpec(
        name="RSI Momentum",
        description="Buy when RSI crosses above 30, sell when above 70",
        entry_long=[
            StrategyCondition(indicator="rsi", operator="crosses_above", value=30),
        ],
        entry_short=[
            StrategyCondition(indicator="rsi", operator="crosses_below", value=70),
        ],
        exit_long=[
            StrategyCondition(indicator="rsi", operator=">", value=70),
        ],
        stop_loss_pct=2.0,
        take_profit_pct=5.0,
    )


@pytest.fixture
def multi_condition_strategy():
    return StrategySpec(
        name="SMA Cross with Volume",
        entry_long=[
            StrategyCondition(indicator="sma_20", operator=">", value="sma_50"),
            StrategyCondition(indicator="volume", operator=">", value=1000000),
        ],
        stop_loss_pct=3.0,
        take_profit_pct=8.0,
    )


class TestPineScriptExporter:
    def test_basic_export(self, simple_strategy):
        exporter = PineScriptExporter()
        result = exporter.export(simple_strategy)
        assert "//@version=6" in result.code
        assert "strategy(" in result.code
        assert "rsi_momentum.pine" in result.filename

    def test_entry_conditions(self, simple_strategy):
        result = PineScriptExporter().export(simple_strategy)
        assert "ta.crossover" in result.code
        assert "strategy.entry" in result.code

    def test_risk_management(self, simple_strategy):
        result = PineScriptExporter().export(simple_strategy)
        assert "strategy.exit" in result.code

    def test_multi_conditions(self, multi_condition_strategy):
        result = PineScriptExporter().export(multi_condition_strategy)
        assert " and " in result.code


class TestMQL5Exporter:
    def test_basic_export(self, simple_strategy):
        exporter = MQL5Exporter()
        result = exporter.export(simple_strategy)
        assert ".mq5" in result.filename
        assert "void OnTick()" in result.code
        assert "OrderSend" in result.code

    def test_inputs(self, simple_strategy):
        result = MQL5Exporter().export(simple_strategy)
        assert "input double LotSize" in result.code
        assert "input double StopLoss" in result.code


class TestTDXExporter:
    def test_basic_export(self, simple_strategy):
        result = TDXExporter().export(simple_strategy)
        assert ".tdx" in result.filename
        assert "买入信号" in result.code or "卖出信号" in result.code


class TestStrategyExporter:
    def test_export_pine(self, simple_strategy):
        exporter = StrategyExporter()
        result = exporter.export(simple_strategy, ExportPlatform.PINE_SCRIPT_V6)
        assert result.platform == ExportPlatform.PINE_SCRIPT_V6
        assert len(result.code) > 0

    def test_export_mql5(self, simple_strategy):
        exporter = StrategyExporter()
        result = exporter.export(simple_strategy, ExportPlatform.MQL5)
        assert result.platform == ExportPlatform.MQL5

    def test_export_tdx(self, simple_strategy):
        exporter = StrategyExporter()
        result = exporter.export(simple_strategy, ExportPlatform.TDX)
        assert result.platform == ExportPlatform.TDX

    def test_export_all(self, simple_strategy):
        exporter = StrategyExporter()
        results = exporter.export_all(simple_strategy)
        assert len(results) == 3
        platforms = {r.platform for r in results}
        assert ExportPlatform.PINE_SCRIPT_V6 in platforms
        assert ExportPlatform.MQL5 in platforms
        assert ExportPlatform.TDX in platforms

    def test_supported_platforms(self):
        exporter = StrategyExporter()
        assert len(exporter.supported_platforms) == 3
