"""
Multi-Platform Strategy Export — CONCEPT:AU-KG.research.research-pipeline-runner

Generates platform-specific trading strategy code from a universal
strategy specification: Pine Script v6 (TradingView), MQL5 (MetaTrader 5),
and TDX formula language.

Source: Vibe-Trading multi-platform export
"""

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class ExportPlatform(StrEnum):
    PINE_SCRIPT_V6 = "pine_v6"
    MQL5 = "mql5"
    TDX = "tdx"
    PYTHON = "python"


@dataclass
class StrategyCondition:
    """A single entry or exit condition."""

    indicator: str  # e.g. "rsi", "sma_cross", "macd_signal"
    operator: str  # ">", "<", "crosses_above", "crosses_below"
    value: float | str = 0.0
    timeframe: str = "1D"


@dataclass
class StrategySpec:
    """Universal strategy specification — platform-agnostic."""

    name: str
    description: str = ""
    entry_long: list[StrategyCondition] = field(default_factory=list)
    entry_short: list[StrategyCondition] = field(default_factory=list)
    exit_long: list[StrategyCondition] = field(default_factory=list)
    exit_short: list[StrategyCondition] = field(default_factory=list)
    position_size_pct: float = 10.0
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 5.0
    timeframe: str = "1D"


@dataclass
class ExportResult:
    """Result of a strategy export."""

    platform: ExportPlatform
    code: str
    filename: str
    warnings: list[str] = field(default_factory=list)


# ── Indicator → Platform Expression Maps ────────────────────────────

PINE_INDICATORS = {
    "rsi": "ta.rsi(close, 14)",
    "sma_20": "ta.sma(close, 20)",
    "sma_50": "ta.sma(close, 50)",
    "ema_12": "ta.ema(close, 12)",
    "ema_26": "ta.ema(close, 26)",
    "macd_signal": "ta.macd(close, 12, 26, 9)",
    "bollinger_upper": "ta.bb(close, 20, 2).1",
    "bollinger_lower": "ta.bb(close, 20, 2).2",
    "volume": "volume",
    "close": "close",
    "open": "open",
    "high": "high",
    "low": "low",
}

MQL5_INDICATORS = {
    "rsi": "iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE)",
    "sma_20": "iMA(_Symbol, PERIOD_CURRENT, 20, 0, MODE_SMA, PRICE_CLOSE)",
    "sma_50": "iMA(_Symbol, PERIOD_CURRENT, 50, 0, MODE_SMA, PRICE_CLOSE)",
    "ema_12": "iMA(_Symbol, PERIOD_CURRENT, 12, 0, MODE_EMA, PRICE_CLOSE)",
    "ema_26": "iMA(_Symbol, PERIOD_CURRENT, 26, 0, MODE_EMA, PRICE_CLOSE)",
    "macd_signal": "iMACD(_Symbol, PERIOD_CURRENT, 12, 26, 9, PRICE_CLOSE)",
    "volume": "iVolume(_Symbol, PERIOD_CURRENT, 0)",
    "close": "iClose(_Symbol, PERIOD_CURRENT, 0)",
}

TDX_INDICATORS = {
    "rsi": "RSI(CLOSE, 14)",
    "sma_20": "MA(CLOSE, 20)",
    "sma_50": "MA(CLOSE, 50)",
    "ema_12": "EMA(CLOSE, 12)",
    "ema_26": "EMA(CLOSE, 26)",
    "macd_signal": "MACD.DEA",
    "volume": "VOL",
    "close": "CLOSE",
}


def _condition_to_pine(cond: StrategyCondition) -> str:
    """Convert a condition to Pine Script expression."""
    indicator = PINE_INDICATORS.get(cond.indicator, cond.indicator)
    if cond.operator == "crosses_above":
        return f"ta.crossover({indicator}, {cond.value})"
    elif cond.operator == "crosses_below":
        return f"ta.crossunder({indicator}, {cond.value})"
    return f"{indicator} {cond.operator} {cond.value}"


def _condition_to_mql5(cond: StrategyCondition) -> str:
    """Convert a condition to MQL5 expression."""
    indicator = MQL5_INDICATORS.get(cond.indicator, cond.indicator)
    if cond.operator in ("crosses_above", "crosses_below"):
        return f"{indicator} {'>=' if cond.operator == 'crosses_above' else '<='} {cond.value}"
    return f"{indicator} {cond.operator} {cond.value}"


def _condition_to_tdx(cond: StrategyCondition) -> str:
    """Convert a condition to TDX formula."""
    indicator = TDX_INDICATORS.get(cond.indicator, cond.indicator)
    op_map = {
        ">": ">",
        "<": "<",
        ">=": ">=",
        "<=": "<=",
        "crosses_above": ">",
        "crosses_below": "<",
    }
    op = op_map.get(cond.operator, cond.operator)
    return f"{indicator} {op} {cond.value}"


class PineScriptExporter:
    """Generates Pine Script v6 code from a StrategySpec."""

    def export(self, spec: StrategySpec) -> ExportResult:
        lines = [
            "//@version=6",
            f'strategy("{spec.name}", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value={spec.position_size_pct})',
            "",
            "// === Entry Conditions ===",
        ]

        # Entry long
        if spec.entry_long:
            conditions = " and ".join(_condition_to_pine(c) for c in spec.entry_long)
            lines.append(f"longCondition = {conditions}")
            lines.append("if longCondition")
            lines.append('    strategy.entry("Long", strategy.long)')

        # Entry short
        if spec.entry_short:
            conditions = " and ".join(_condition_to_pine(c) for c in spec.entry_short)
            lines.append(f"shortCondition = {conditions}")
            lines.append("if shortCondition")
            lines.append('    strategy.entry("Short", strategy.short)')

        # Exit long
        if spec.exit_long:
            conditions = " and ".join(_condition_to_pine(c) for c in spec.exit_long)
            lines.append(f"exitLongCondition = {conditions}")
            lines.append("if exitLongCondition")
            lines.append('    strategy.close("Long")')

        # Stop loss / take profit
        if spec.stop_loss_pct > 0:
            lines.append("")
            lines.append("// === Risk Management ===")
            lines.append(
                f'strategy.exit("SL/TP Long", "Long", stop=strategy.position_avg_price * (1 - {spec.stop_loss_pct / 100}), limit=strategy.position_avg_price * (1 + {spec.take_profit_pct / 100}))'
            )

        return ExportResult(
            platform=ExportPlatform.PINE_SCRIPT_V6,
            code="\n".join(lines),
            filename=f"{spec.name.lower().replace(' ', '_')}.pine",
        )


class MQL5Exporter:
    """Generates MQL5 Expert Advisor code from a StrategySpec."""

    def export(self, spec: StrategySpec) -> ExportResult:
        lines = [
            "//+------------------------------------------------------------------+",
            f"//| {spec.name}.mq5",
            "//| Auto-generated by agent-utilities Strategy Export",
            "//+------------------------------------------------------------------+",
            "#property strict",
            "",
            f"input double LotSize = {spec.position_size_pct / 100:.2f};",
            f"input double StopLoss = {spec.stop_loss_pct};",
            f"input double TakeProfit = {spec.take_profit_pct};",
            "",
            "void OnTick() {",
        ]

        if spec.entry_long:
            conditions = " && ".join(_condition_to_mql5(c) for c in spec.entry_long)
            lines.append("    // Long entry")
            lines.append(f"    if ({conditions}) {{")
            lines.append(
                "        OrderSend(_Symbol, OP_BUY, LotSize, Ask, 3, Ask - StopLoss * _Point, Ask + TakeProfit * _Point);"
            )
            lines.append("    }")

        if spec.entry_short:
            conditions = " && ".join(_condition_to_mql5(c) for c in spec.entry_short)
            lines.append("    // Short entry")
            lines.append(f"    if ({conditions}) {{")
            lines.append(
                "        OrderSend(_Symbol, OP_SELL, LotSize, Bid, 3, Bid + StopLoss * _Point, Bid - TakeProfit * _Point);"
            )
            lines.append("    }")

        lines.append("}")

        return ExportResult(
            platform=ExportPlatform.MQL5,
            code="\n".join(lines),
            filename=f"{spec.name.lower().replace(' ', '_')}.mq5",
        )


class TDXExporter:
    """Generates TDX (通达信) formula code."""

    def export(self, spec: StrategySpec) -> ExportResult:
        lines = [
            f"{{策略: {spec.name}}}",
            f"{{Description: {spec.description}}}",
            "",
        ]

        if spec.entry_long:
            conditions = " AND ".join(_condition_to_tdx(c) for c in spec.entry_long)
            lines.append(f"买入信号: {conditions};")

        if spec.entry_short:
            conditions = " AND ".join(_condition_to_tdx(c) for c in spec.entry_short)
            lines.append(f"卖出信号: {conditions};")

        return ExportResult(
            platform=ExportPlatform.TDX,
            code="\n".join(lines),
            filename=f"{spec.name.lower().replace(' ', '_')}.tdx",
        )


class StrategyExporter:
    """
    Unified strategy export engine supporting multiple platforms.

    Usage:
        exporter = StrategyExporter()
        results = exporter.export_all(spec)
    """

    def __init__(self):
        self._exporters: dict[ExportPlatform, Any] = {
            ExportPlatform.PINE_SCRIPT_V6: PineScriptExporter(),
            ExportPlatform.MQL5: MQL5Exporter(),
            ExportPlatform.TDX: TDXExporter(),
        }

    def export(self, spec: StrategySpec, platform: ExportPlatform) -> ExportResult:
        """Export to a specific platform."""
        exporter = self._exporters.get(platform)
        if not exporter:
            return ExportResult(
                platform=platform,
                code="",
                filename="",
                warnings=[f"Unsupported platform: {platform}"],
            )
        return exporter.export(spec)

    def export_all(self, spec: StrategySpec) -> list[ExportResult]:
        """Export to all supported platforms."""
        return [exporter.export(spec) for exporter in self._exporters.values()]

    @property
    def supported_platforms(self) -> list[ExportPlatform]:
        return list(self._exporters.keys())
