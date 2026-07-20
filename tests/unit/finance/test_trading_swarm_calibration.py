"""Calibration read-back in TradingSwarm aggregation (CONCEPT:AU-KG.domains.agent-calibration-reputation-tracking).

calibrated_role_weights existed but was never consulted by trading_swarm —
reputation was tracked, then ignored. Aggregation now EMA-blends each role's
static weight toward its agents' tracked calibration, gated by
SwarmConfig.calibration_weighting, and analyze()/record_market_outcome() close
the call→outcome loop natively.

@pytest.mark.concept("AU-KG.domains.agent-calibration-reputation-tracking")
"""

from __future__ import annotations

import pytest

from agent_utilities.domains.finance.calibration_tracker import CalibrationTracker
from agent_utilities.domains.finance.trading_swarm import (
    AgentSignal,
    SwarmAgent,
    SwarmConfig,
    SwarmRole,
    TradingSwarm,
)


class _FixedAgent(SwarmAgent):
    """Agent emitting a fixed (direction, confidence) signal."""

    def __init__(self, agent_id, role, direction, confidence):
        super().__init__(agent_id, role)
        self._fixed = (direction, confidence)

    def analyze(self, market_data):
        direction, confidence = self._fixed
        signal = AgentSignal(
            agent_id=self.agent_id,
            role=self.role,
            direction=direction,
            confidence=confidence,
        )
        self._signal_history.append(signal)
        return signal


pytestmark = pytest.mark.concept("AU-KG.domains.agent-calibration-reputation-tracking")

BULLISH_DATA = {
    "symbol": "ACME",
    "momentum": 0.06,
    "trend": 0.05,
    "volume_signal": 0.6,
    "rsi": 55.0,
    "volatility": 0.01,
}


def _swarm(**config_kw) -> TradingSwarm:
    return TradingSwarm.create_default(config=SwarmConfig(**config_kw))


class TestEffectiveWeights:
    def test_no_resolved_calls_keeps_base_weights(self):
        swarm = _swarm()
        weights = swarm._effective_role_weights()
        for role, base in swarm.config.role_weights.items():
            assert weights[role] == pytest.approx(base)

    def test_bad_calibration_lowers_role_weight(self):
        swarm = _swarm()
        # The quant is reliably wrong with high confidence.
        for i in range(6):
            swarm.calibration.record_call(
                "quant_01", direction=+1, confidence=0.9, subject=f"s{i}"
            )
            swarm.calibration.record_outcome(
                "quant_01", realized_direction=-1, subject=f"s{i}"
            )
        weights = swarm._effective_role_weights()
        base = swarm.config.role_weights
        assert weights[SwarmRole.QUANT_ANALYST] < base[SwarmRole.QUANT_ANALYST]
        # untouched roles keep their base weight
        assert weights[SwarmRole.DIRECTOR] == pytest.approx(base[SwarmRole.DIRECTOR])

    def test_config_gate_pins_static_weights(self):
        swarm = _swarm(calibration_weighting=False)
        for i in range(6):
            swarm.calibration.record_call(
                "quant_01", direction=+1, confidence=0.9, subject=f"s{i}"
            )
            swarm.calibration.record_outcome(
                "quant_01", realized_direction=-1, subject=f"s{i}"
            )
        assert swarm._effective_role_weights() == dict(swarm.config.role_weights)

    def test_alpha_bounds_blend(self):
        swarm = _swarm(calibration_alpha=0.0)
        for i in range(4):
            swarm.calibration.record_call(
                "quant_01", direction=+1, confidence=0.9, subject=f"s{i}"
            )
            swarm.calibration.record_outcome(
                "quant_01", realized_direction=-1, subject=f"s{i}"
            )
        # alpha=0 → fully static even with terrible calibration
        assert swarm._effective_role_weights() == dict(swarm.config.role_weights)


class TestAnalyzeLoop:
    def test_analyze_records_open_calls(self):
        swarm = _swarm()
        swarm.analyze(BULLISH_DATA)
        open_calls = [
            rec
            for calls in swarm.calibration._calls.values()
            for rec in calls
            if rec.correct is None
        ]
        assert open_calls, "directional signals must open calibration calls"
        assert all(c.subject == "ACME" for c in open_calls)

    def test_record_market_outcome_resolves_calls(self):
        swarm = _swarm()
        swarm.analyze(BULLISH_DATA)
        resolved = swarm.record_market_outcome(+1, subject="ACME")
        assert resolved > 0
        assert all(
            rec.correct is not None
            for calls in swarm.calibration._calls.values()
            for rec in calls
        )

    def test_reputation_shifts_consensus_score(self):
        # Two bullish agents vs one confident BEARISH quant. In one swarm the
        # quant has a proven-bad record; discounting its dissent must move the
        # consensus toward the well-calibrated bullish majority.
        def _make():
            return [
                _FixedAgent("dir_01", SwarmRole.DIRECTOR, +1, 0.8),
                _FixedAgent("trend_01", SwarmRole.TREND_ANALYST, +1, 0.7),
                _FixedAgent("quant_01", SwarmRole.QUANT_ANALYST, -1, 0.9),
            ]

        neutral = TradingSwarm(agents=_make(), config=SwarmConfig())
        skeptical = TradingSwarm(agents=_make(), config=SwarmConfig())
        for i in range(8):
            skeptical.calibration.record_call(
                "quant_01", direction=-1, confidence=0.95, subject=f"s{i}"
            )
            skeptical.calibration.record_outcome(
                "quant_01", realized_direction=+1, subject=f"s{i}"
            )
        score_neutral = neutral.analyze(BULLISH_DATA).weighted_score
        score_skeptical = skeptical.analyze(BULLISH_DATA).weighted_score
        # the bearish dissenter's discounted voice raises the bullish consensus
        assert score_skeptical > score_neutral

    def test_injected_tracker_is_used(self):
        tracker = CalibrationTracker()
        swarm = TradingSwarm.create_default()
        swarm_with = TradingSwarm(agents=swarm.agents, calibration=tracker)
        assert swarm_with.calibration is tracker
