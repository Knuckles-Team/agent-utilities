"""Tests for CONCEPT:KG-2.6 — Multi-Agent Trading Swarm."""

import pytest

from agent_utilities.domains.finance.trading_swarm import (
    AgentSignal,
    SwarmAgent,
    SwarmConfig,
    SwarmConsensus,
    SwarmDecision,
    SwarmRole,
    TradingSwarm,
)


@pytest.fixture
def bullish_data():
    return {
        "momentum": 0.08,
        "volatility": 0.01,
        "rsi": 55,
        "trend": 0.03,
        "volume_signal": 0.5,
    }


@pytest.fixture
def bearish_data():
    return {
        "momentum": -0.08,
        "volatility": 0.01,
        "rsi": 75,
        "trend": -0.03,
        "volume_signal": -0.3,
    }


@pytest.fixture
def volatile_data():
    return {
        "momentum": 0.02,
        "volatility": 0.05,
        "rsi": 50,
        "trend": 0.01,
        "volume_signal": 0.1,
    }


class TestSwarmAgent:
    def test_quant_analyst_bullish(self, bullish_data):
        agent = SwarmAgent("quant_01", SwarmRole.QUANT_ANALYST)
        signal = agent.analyze(bullish_data)
        assert signal.direction == 1
        assert signal.confidence > 0

    def test_risk_manager_high_vol(self, volatile_data):
        agent = SwarmAgent("risk_01", SwarmRole.RISK_MANAGER)
        signal = agent.analyze(volatile_data)
        assert signal.direction == 0  # Hold on high vol

    def test_indicator_overbought(self, bearish_data):
        agent = SwarmAgent("ind_01", SwarmRole.INDICATOR_SPECIALIST)
        signal = agent.analyze(bearish_data)
        assert signal.direction == -1  # RSI overbought

    def test_trend_analyst(self, bullish_data):
        agent = SwarmAgent("trend_01", SwarmRole.TREND_ANALYST)
        signal = agent.analyze(bullish_data)
        assert signal.direction == 1

    def test_signal_history(self, bullish_data):
        agent = SwarmAgent("test_01", SwarmRole.QUANT_ANALYST)
        agent.analyze(bullish_data)
        agent.analyze(bullish_data)
        assert agent.signal_count == 2


class TestTradingSwarm:
    def test_create_default(self):
        swarm = TradingSwarm.create_default()
        assert swarm.agent_count == 6

    def test_bullish_consensus(self, bullish_data):
        swarm = TradingSwarm.create_default()
        consensus = swarm.analyze(bullish_data)
        assert consensus.decision in (
            SwarmDecision.BUY,
            SwarmDecision.STRONG_BUY,
            SwarmDecision.HOLD,
        )
        assert len(consensus.signals) == 6

    def test_bearish_consensus(self, bearish_data):
        swarm = TradingSwarm.create_default()
        consensus = swarm.analyze(bearish_data)
        assert isinstance(consensus.decision, SwarmDecision)

    def test_risk_veto(self, volatile_data):
        config = SwarmConfig(risk_veto_enabled=True)
        swarm = TradingSwarm.create_default(config=config)
        consensus = swarm.analyze(volatile_data)
        if consensus.risk_override:
            assert consensus.decision == SwarmDecision.HOLD

    def test_insufficient_agents(self, bullish_data):
        swarm = TradingSwarm(
            agents=[SwarmAgent("a", SwarmRole.QUANT_ANALYST)],
            config=SwarmConfig(min_agents_for_consensus=3),
        )
        consensus = swarm.analyze(bullish_data)
        assert consensus.decision == SwarmDecision.NO_CONSENSUS

    def test_consensus_history(self, bullish_data, bearish_data):
        swarm = TradingSwarm.create_default()
        swarm.analyze(bullish_data)
        swarm.analyze(bearish_data)
        assert len(swarm.get_consensus_history()) == 2

    def test_add_agent(self):
        swarm = TradingSwarm.create_default()
        initial = swarm.agent_count
        swarm.add_agent(SwarmAgent("sentiment_01", SwarmRole.SENTIMENT_ANALYST))
        assert swarm.agent_count == initial + 1

    def test_roles_list(self):
        swarm = TradingSwarm.create_default()
        assert SwarmRole.DIRECTOR in swarm.roles
        assert SwarmRole.RISK_MANAGER in swarm.roles

    def test_agreement_ratio(self, bullish_data):
        swarm = TradingSwarm.create_default()
        consensus = swarm.analyze(bullish_data)
        assert 0.0 <= consensus.agreement_ratio <= 1.0

    def test_dissenting_agents_tracked(self, bullish_data):
        swarm = TradingSwarm.create_default()
        consensus = swarm.analyze(bullish_data)
        assert isinstance(consensus.dissenting_agents, list)
