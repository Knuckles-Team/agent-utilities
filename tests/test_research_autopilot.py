"""Tests for CONCEPT:KG-2.6 — Research Autopilot."""

from agent_utilities.numeric import xp as np

from agent_utilities.domains.finance.research_autopilot import (
    AutopilotConfig,
    Hypothesis,
    HypothesisStatus,
    ResearchAutopilot,
    SimpleBacktester,
)


class TestSimpleBacktester:
    def test_profitable_strategy(self):
        rng = np.random.default_rng(42)
        n = 300
        returns = rng.normal(0.001, 0.01, n)
        entry = np.zeros(n, dtype=bool)
        exit_ = np.zeros(n, dtype=bool)
        # Enter every 10 bars, exit 5 bars later
        for i in range(0, n, 10):
            entry[i] = True
            if i + 5 < n:
                exit_[i + 5] = True

        bt = SimpleBacktester()
        metrics = bt.run(entry, exit_, returns)
        assert metrics.total_trades > 0
        assert metrics.win_rate > 0

    def test_no_trades(self):
        n = 100
        bt = SimpleBacktester()
        metrics = bt.run(np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), np.zeros(n))
        assert metrics.total_trades == 0

    def test_short_data(self):
        bt = SimpleBacktester()
        metrics = bt.run(np.array([True]), np.array([True]), np.array([0.01]))
        assert metrics.total_trades == 0

    def test_metrics_values(self):
        rng = np.random.default_rng(42)
        n = 200
        returns = rng.normal(0.002, 0.01, n)
        entry = rng.random(n) > 0.9
        exit_ = rng.random(n) > 0.85
        bt = SimpleBacktester()
        metrics = bt.run(entry, exit_, returns)
        if metrics.total_trades > 0:
            assert 0.0 <= metrics.win_rate <= 1.0
            assert metrics.max_drawdown <= 0


class TestResearchAutopilot:
    def test_add_hypothesis(self):
        autopilot = ResearchAutopilot()
        autopilot.add_hypothesis(Hypothesis(hypothesis_id="h1", title="Test"))
        assert autopilot.pending_count == 1

    def test_run_with_synthetic_data(self):
        autopilot = ResearchAutopilot()
        autopilot.add_hypothesis(
            Hypothesis(
                hypothesis_id="h1",
                title="Momentum Test",
                entry_rule="momentum > 0",
                exit_rule="momentum < 0",
            )
        )
        autopilot.add_hypothesis(
            Hypothesis(
                hypothesis_id="h2",
                title="Mean Reversion",
                entry_rule="rsi < 30",
                exit_rule="rsi > 70",
            )
        )
        report = autopilot.run()
        assert report.hypotheses_tested == 2
        assert report.session_id == "research:0001"

    def test_hypothesis_status_updated(self):
        autopilot = ResearchAutopilot()
        h = Hypothesis(hypothesis_id="h1", title="Test")
        autopilot.add_hypothesis(h)
        autopilot.run()
        assert h.status in (HypothesisStatus.VALIDATED, HypothesisStatus.REJECTED)

    def test_strict_config_rejects_more(self):
        strict = AutopilotConfig(
            min_sharpe=2.0, min_win_rate=0.7, min_profit_factor=3.0
        )
        autopilot = ResearchAutopilot(config=strict)
        for i in range(5):
            autopilot.add_hypothesis(
                Hypothesis(hypothesis_id=f"h{i}", title=f"Test {i}")
            )
        report = autopilot.run()
        # Strict criteria = most should be rejected
        assert report.hypotheses_rejected >= report.hypotheses_passed

    def test_report_has_results(self):
        autopilot = ResearchAutopilot()
        autopilot.add_hypothesis(Hypothesis(hypothesis_id="h1", title="Test"))
        report = autopilot.run()
        assert len(report.results) == 1
        assert report.results[0].report != ""

    def test_already_tested_skipped(self):
        autopilot = ResearchAutopilot()
        h = Hypothesis(
            hypothesis_id="h1", title="Test", status=HypothesisStatus.VALIDATED
        )
        autopilot.add_hypothesis(h)
        report = autopilot.run()
        assert report.hypotheses_tested == 0

    def test_session_counter_increments(self):
        autopilot = ResearchAutopilot()
        autopilot.add_hypothesis(Hypothesis(hypothesis_id="h1", title="T1"))
        r1 = autopilot.run()
        autopilot.add_hypothesis(Hypothesis(hypothesis_id="h2", title="T2"))
        r2 = autopilot.run()
        assert r1.session_id == "research:0001"
        assert r2.session_id == "research:0002"

    def test_with_custom_data(self):
        autopilot = ResearchAutopilot(config=AutopilotConfig(min_trades=5))
        autopilot.add_hypothesis(
            Hypothesis(hypothesis_id="h1", title="Custom Data Test")
        )

        rng = np.random.default_rng(42)
        n = 200
        data = {
            "h1": {
                "entry_signals": rng.random(n) > 0.9,
                "exit_signals": rng.random(n) > 0.85,
                "returns": rng.normal(0.002, 0.01, n),
            }
        }
        report = autopilot.run(data=data)
        assert report.hypotheses_tested == 1
