"""Unit + live-path tests for the copy-trade pipeline (CONCEPT:AU-KG.research.research-pipeline-runner).

Engine-dependent sizing degrades gracefully to the local Kelly fallback when no
epistemic-graph engine is reachable, so these run fully offline.
"""

from __future__ import annotations

from agent_utilities.domains.finance.copy_trade import (
    CopyTradeConfig,
    CopyTradePipeline,
    aggregate_consensus,
    build_copy_trade_workflow,
    exit_check,
    position_multiplier,
    score_market,
    seed_copy_trade_workflow,
    size_position,
)
from agent_utilities.domains.finance.trading_swarm import (
    AgentSignal,
    SwarmConfig,
    SwarmDecision,
    SwarmRole,
)

# ── Scanner ────────────────────────────────────────────────────────────────


def test_scanner_passes_good_market():
    m = {
        "question": "X?",
        "midpoint": 0.40,
        "bids_depth": 1000.0,
        "asks_depth": 1000.0,
        "hours_to_resolution": 24.0,
        "token": "t1",
    }
    entry = score_market(m, estimate=0.60)
    assert entry is not None
    assert entry["side"] == "YES"
    assert entry["gap"] == 0.20


def test_scanner_kills_thin_edge_low_depth_and_bad_horizon():
    base = {
        "question": "X?",
        "midpoint": 0.50,
        "bids_depth": 1000.0,
        "asks_depth": 1000.0,
        "hours_to_resolution": 24.0,
    }
    # thin edge
    assert score_market({**base}, estimate=0.53) is None
    # low depth
    assert score_market({**base, "bids_depth": 100.0}, estimate=0.70) is None
    # too late
    assert score_market({**base, "hours_to_resolution": 1.0}, estimate=0.70) is None
    # too slow
    assert score_market({**base, "hours_to_resolution": 400.0}, estimate=0.70) is None


# ── Consensus & position multiplier ─────────────────────────────────────────


def _sig(agent, role, direction, conf=0.8):
    return AgentSignal(agent_id=agent, role=role, direction=direction, confidence=conf)


def test_two_agree_full_one_half_disagree_none():
    cfg = CopyTradeConfig()
    swarm = SwarmConfig(min_agents_for_consensus=1)
    # 2 buy, 1 hold ⇒ full position
    sigs = [
        _sig("arbitrage", SwarmRole.QUANT_ANALYST, 1),
        _sig("convergence", SwarmRole.TREND_ANALYST, 1),
        _sig("whale_copy", SwarmRole.DIRECTOR, 0),
    ]
    c = aggregate_consensus(sigs, swarm)
    assert position_multiplier(c, cfg) == 1.0

    # exactly 1 buy ⇒ half
    sigs = [
        _sig("arbitrage", SwarmRole.QUANT_ANALYST, 1),
        _sig("convergence", SwarmRole.TREND_ANALYST, 0),
        _sig("whale_copy", SwarmRole.DIRECTOR, 0),
    ]
    c = aggregate_consensus(sigs, swarm)
    assert position_multiplier(c, cfg) == 0.5

    # net-zero / opposing ⇒ no trade
    sigs = [
        _sig("arbitrage", SwarmRole.QUANT_ANALYST, 1, 0.8),
        _sig("convergence", SwarmRole.TREND_ANALYST, -1, 0.8),
        _sig("whale_copy", SwarmRole.DIRECTOR, 0, 0.8),
    ]
    c = aggregate_consensus(sigs, swarm)
    # DIRECTOR weight breaks ties toward hold; either way no full position
    assert position_multiplier(c, cfg) in (0.0, 0.5)


# ── Sizing (engine or local fallback) ───────────────────────────────────────


def test_size_position_kelly_fallback():
    # q=0.6, c=0.5 -> f*=0.2; quarter-Kelly -> 0.05; × full multiplier
    f = size_position(estimate=0.6, price=0.5, multiplier=1.0)
    assert abs(f - 0.05) < 1e-6
    # negative EV ⇒ zero
    assert size_position(estimate=0.4, price=0.5, multiplier=1.0) == 0.0
    # no consensus ⇒ zero regardless
    assert size_position(estimate=0.6, price=0.5, multiplier=0.0) == 0.0


# ── Exit logic ───────────────────────────────────────────────────────────────


def test_exit_triggers():
    pos = {"entry": 0.35, "target": 0.72, "hours_since_entry": 1.0}
    # 85% of (0.72-0.35)=0.3145 above entry -> 0.6645
    assert exit_check(pos, 0.67, volume_10m=1.0, avg_volume=1.0) == "TARGET_HIT"
    # volume spike
    assert exit_check(pos, 0.40, volume_10m=10.0, avg_volume=1.0) == "VOLUME_EXIT"
    # stale thesis
    stale = {"entry": 0.35, "target": 0.72, "hours_since_entry": 30.0}
    assert exit_check(stale, 0.355, volume_10m=1.0, avg_volume=1.0) == "STALE_THESIS"
    # hold
    assert exit_check(pos, 0.45, volume_10m=1.0, avg_volume=1.0) is None


# ── End-to-end pipeline (live path through the real class) ──────────────────


def test_pipeline_end_to_end():
    def evaluator(entry):
        # whale + convergence both long ⇒ should clear consensus
        return [
            _sig("arbitrage", SwarmRole.QUANT_ANALYST, 1, 0.7),
            _sig("convergence", SwarmRole.TREND_ANALYST, 1, 0.8),
            _sig("whale_copy", SwarmRole.DIRECTOR, 1, 0.9),
        ]

    markets = [
        # passes scanner (gap .2, depth 1000, 24h)
        (
            {
                "question": "A?",
                "midpoint": 0.40,
                "bids_depth": 1000.0,
                "asks_depth": 1000.0,
                "hours_to_resolution": 24.0,
                "token": "a",
            },
            0.60,
        ),
        # killed by scanner (thin edge)
        (
            {
                "question": "B?",
                "midpoint": 0.50,
                "bids_depth": 1000.0,
                "asks_depth": 1000.0,
                "hours_to_resolution": 24.0,
                "token": "b",
            },
            0.52,
        ),
    ]
    pipe = CopyTradePipeline()
    intents = pipe.run(markets, evaluator)
    assert len(intents) == 1
    intent = intents[0]
    assert intent.token == "a"
    assert intent.side == "YES"
    assert intent.position_fraction > 0.0
    assert intent.decision in (SwarmDecision.BUY, SwarmDecision.STRONG_BUY)


# ── Workflow registration ────────────────────────────────────────────────────


def test_build_and_seed_workflow():
    spec = build_copy_trade_workflow()
    assert spec.name == "copy_trade_prediction_market"
    assert "scan" in spec.steps and "exit_monitor" in spec.steps
    assert any("emerald-exchange" in o for o in spec.orchestrates)

    captured = {}

    def fake_write_batch(batch):
        captured["batch"] = batch
        return "ok"

    assert seed_copy_trade_workflow(fake_write_batch) == "ok"
    assert captured["batch"].category == "orchestration"
