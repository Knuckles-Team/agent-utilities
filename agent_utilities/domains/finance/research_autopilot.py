"""
Research Autopilot — CONCEPT:AU-KG.research.research-pipeline-runner

Automated hypothesis → backtest → report loop for overnight
strategy research and discovery.

Source: Vibe-Trading Research Autopilot
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from agent_utilities.domains.finance.debate_engine import DebateContext, DebateEngine
from agent_utilities.numeric import xp as np

logger = logging.getLogger(__name__)


class HypothesisStatus(StrEnum):
    PENDING = "pending"
    TESTING = "testing"
    VALIDATED = "validated"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass
class Hypothesis:
    """A trading hypothesis to be tested."""

    hypothesis_id: str
    title: str
    description: str = ""
    entry_rule: str = ""
    exit_rule: str = ""
    expected_edge: str = ""
    asset_class: str = "equity"
    timeframe: str = "1D"
    status: HypothesisStatus = HypothesisStatus.PENDING
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()


@dataclass
class BacktestMetrics:
    """Metrics from a hypothesis backtest."""

    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_duration: float = 0.0


@dataclass
class HypothesisResult:
    """Result of testing a single hypothesis."""

    hypothesis: Hypothesis
    metrics: BacktestMetrics
    passed: bool = False
    pass_criteria: dict[str, bool] = field(default_factory=dict)
    report: str = ""
    tested_at: str = ""

    def __post_init__(self):
        if not self.tested_at:
            self.tested_at = datetime.now(UTC).isoformat()


@dataclass
class ResearchReport:
    """Complete research report from an autopilot session."""

    session_id: str
    hypotheses_tested: int = 0
    hypotheses_passed: int = 0
    hypotheses_rejected: int = 0
    results: list[HypothesisResult] = field(default_factory=list)
    best_hypothesis: str = ""
    started_at: str = ""
    completed_at: str = ""

    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now(UTC).isoformat()


@dataclass
class AutopilotConfig:
    """Configuration for the research autopilot."""

    min_sharpe: float = 0.5
    max_drawdown: float = -0.20
    min_trades: int = 30
    min_win_rate: float = 0.45
    min_profit_factor: float = 1.2
    backtest_periods: int = 252  # 1 year of daily bars


class SimpleBacktester:
    """
    Minimal backtesting engine for hypothesis validation.
    Evaluates entry/exit rules against synthetic or historical returns.
    """

    def run(
        self, entry_signals: np.ndarray, exit_signals: np.ndarray, returns: np.ndarray
    ) -> BacktestMetrics:
        """
        Run a backtest given entry/exit signal arrays and period returns.

        Args:
            entry_signals: Boolean array — True when entry is triggered.
            exit_signals: Boolean array — True when exit is triggered.
            returns: Array of period returns.
        """
        n = min(len(entry_signals), len(exit_signals), len(returns))
        if n < 10:
            return BacktestMetrics()

        in_position = False
        trade_returns = []
        current_trade = 0.0
        trade_lengths = []
        current_length = 0

        for i in range(n):
            if not in_position and entry_signals[i]:
                in_position = True
                current_trade = 0.0
                current_length = 0
            elif in_position:
                current_trade += returns[i]
                current_length += 1
                if exit_signals[i] or i == n - 1:
                    trade_returns.append(current_trade)
                    trade_lengths.append(current_length)
                    in_position = False

        if not trade_returns:
            return BacktestMetrics()

        trades = np.array(trade_returns)
        wins = trades[trades > 0]
        losses = trades[trades < 0]

        total_return = float(np.sum(trades))
        sharpe = (
            float(np.mean(trades) / np.std(trades) * np.sqrt(252))
            if np.std(trades) > 0
            else 0.0
        )

        # Max drawdown
        cumulative = np.cumsum(trades)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        win_rate = float(len(wins) / len(trades)) if len(trades) > 0 else 0.0
        profit_factor = (
            float(np.sum(wins) / abs(np.sum(losses)))
            if len(losses) > 0 and np.sum(losses) != 0
            else float("inf")
        )
        avg_duration = float(np.mean(trade_lengths)) if trade_lengths else 0.0

        return BacktestMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_duration=avg_duration,
        )


class ResearchAutopilot:
    """
    Automated research loop: generates hypotheses, backtests them,
    and produces a summary report.

    Usage:
        autopilot = ResearchAutopilot()
        autopilot.add_hypothesis(Hypothesis(...))
        report = autopilot.run()
    """

    def __init__(
        self, config: AutopilotConfig | None = None, engine: Any | None = None
    ):
        self.config = config or AutopilotConfig()
        self.backtester = SimpleBacktester()
        self.debate_engine = DebateEngine(engine=engine) if engine else None
        self._hypotheses: list[Hypothesis] = []
        self._session_count = 0

    def add_hypothesis(self, hypothesis: Hypothesis) -> None:
        """Add a hypothesis to test."""
        self._hypotheses.append(hypothesis)

    def _evaluate_hypothesis(
        self,
        hypothesis: Hypothesis,
        entry_signals: np.ndarray,
        exit_signals: np.ndarray,
        returns: np.ndarray,
    ) -> HypothesisResult:
        """Test a single hypothesis against data."""
        hypothesis.status = HypothesisStatus.TESTING

        try:
            # 1. Peer Review via Debate Engine (if wired to KG engine)
            debate_summary = ""
            if self.debate_engine:
                context = DebateContext(
                    ticker="HYPOTHETICAL",
                    asset_class=hypothesis.asset_class,
                    market_report=f"Hypothesis: {hypothesis.title}\n{hypothesis.description}",
                )
                debate_session = self.debate_engine.run_debate(
                    session_id=f"peer_review_{hypothesis.hypothesis_id}",
                    context=context,
                    num_rounds=2,
                )

                # If risk manager vetoes the hypothesis, reject early before expensive backtest
                if (
                    debate_session.risk_assessment
                    and not debate_session.risk_assessment.approved
                ):
                    hypothesis.status = HypothesisStatus.REJECTED
                    return HypothesisResult(
                        hypothesis=hypothesis,
                        metrics=BacktestMetrics(),
                        passed=False,
                        report=f"❌ REJECTED IN PEER REVIEW\nReasoning: {debate_session.risk_assessment.reasoning}",
                    )
                reasoning = (
                    debate_session.risk_assessment.reasoning
                    if debate_session.risk_assessment
                    else "No risk assessment available"
                )
                debate_summary = (
                    f"\n\n**Peer Review**: ✅ APPROVED\nReasoning: {reasoning}\n"
                )

            # 2. Backtest Phase
            metrics = self.backtester.run(entry_signals, exit_signals, returns)

            # Evaluate pass criteria
            criteria = {
                "sharpe": metrics.sharpe_ratio >= self.config.min_sharpe,
                "drawdown": metrics.max_drawdown >= self.config.max_drawdown,
                "trades": metrics.total_trades >= self.config.min_trades,
                "win_rate": metrics.win_rate >= self.config.min_win_rate,
                "profit_factor": metrics.profit_factor >= self.config.min_profit_factor,
            }

            passed = all(criteria.values())
            hypothesis.status = (
                HypothesisStatus.VALIDATED if passed else HypothesisStatus.REJECTED
            )

            # Generate report
            report_lines = [
                f"## {hypothesis.title}",
                "",
                f"**Status**: {'✅ VALIDATED' if passed else '❌ REJECTED'}",
                "",
                "| Metric | Value | Threshold | Pass |",
                "|--------|-------|-----------|------|",
                f"| Sharpe | {metrics.sharpe_ratio:.2f} | ≥{self.config.min_sharpe} | {'✅' if criteria['sharpe'] else '❌'} |",
                f"| Max DD | {metrics.max_drawdown:.1%} | ≥{self.config.max_drawdown:.0%} | {'✅' if criteria['drawdown'] else '❌'} |",
                f"| Trades | {metrics.total_trades} | ≥{self.config.min_trades} | {'✅' if criteria['trades'] else '❌'} |",
                f"| Win Rate | {metrics.win_rate:.1%} | ≥{self.config.min_win_rate:.0%} | {'✅' if criteria['win_rate'] else '❌'} |",
                f"| Profit Factor | {metrics.profit_factor:.2f} | ≥{self.config.min_profit_factor} | {'✅' if criteria['profit_factor'] else '❌'} |",
                debate_summary,
            ]

            return HypothesisResult(
                hypothesis=hypothesis,
                metrics=metrics,
                passed=passed,
                pass_criteria=criteria,
                report="\n".join(report_lines),
            )

        except Exception as e:
            hypothesis.status = HypothesisStatus.ERROR
            return HypothesisResult(
                hypothesis=hypothesis,
                metrics=BacktestMetrics(),
                passed=False,
                report=f"Error: {e}",
            )

    def run(self, data: dict[str, Any] | None = None) -> ResearchReport:
        """
        Run all pending hypotheses and produce a research report.

        Args:
            data: Dict with keys 'entry_signals', 'exit_signals', 'returns'
                  per hypothesis_id. If None, uses synthetic data.
        """
        self._session_count += 1
        session_id = f"research:{self._session_count:04d}"

        results = []
        for hyp in self._hypotheses:
            if hyp.status != HypothesisStatus.PENDING:
                continue

            if data and hyp.hypothesis_id in data:
                d = data[hyp.hypothesis_id]
                entry = d.get("entry_signals", np.array([]))
                exit_ = d.get("exit_signals", np.array([]))
                rets = d.get("returns", np.array([]))
            else:
                # Generate synthetic test data
                rng = np.random.default_rng(hash(hyp.hypothesis_id) % 2**32)
                n = self.config.backtest_periods
                rets = rng.normal(0.0005, 0.02, n)
                entry = rng.random(n) > 0.95
                exit_ = rng.random(n) > 0.90

            result = self._evaluate_hypothesis(hyp, entry, exit_, rets)
            results.append(result)

        passed = [r for r in results if r.passed]
        best = (
            max(passed, key=lambda r: r.metrics.sharpe_ratio).hypothesis.title
            if passed
            else ""
        )

        report = ResearchReport(
            session_id=session_id,
            hypotheses_tested=len(results),
            hypotheses_passed=len(passed),
            hypotheses_rejected=len(results) - len(passed),
            results=results,
            best_hypothesis=best,
            completed_at=datetime.now(UTC).isoformat(),
        )
        return report

    @property
    def pending_count(self) -> int:
        return sum(1 for h in self._hypotheses if h.status == HypothesisStatus.PENDING)
