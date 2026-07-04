"""Agent Calibration / Reputation tests — CONCEPT:AU-KG.domains.agent-calibration-reputation-tracking.

Brier maths (local fallback), KG persistence, and the LIVE wire into
SwarmConsensus weighting (calibration changes the swarm's aggregated decision).
"""

from __future__ import annotations

from agent_utilities.domains.finance.calibration_tracker import (
    CalibrationTracker,
    apply_calibration_to_swarm,
    brier_score,
    calibrated_role_weights,
)
from agent_utilities.domains.finance.trading_swarm import (
    AgentSignal,
    SwarmAgent,
    SwarmDecision,
    SwarmRole,
    TradingSwarm,
)


class _FakeBackend:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node_id, type=None, **props):
        self.nodes.append((node_id, type, props))

    def add_edge(self, src, tgt, rel_type=None):
        self.edges.append((src, tgt, rel_type))

    # The KG persist path now writes via the materialization core's UNWIND
    # MERGE batches (write_batch -> write_entities -> execute_batch,
    # CONCEPT:AU-KG.ingest.enterprise-source-extractor), so decode those into the same (id, type, props) /
    # (src, tgt, rel) shape the assertions inspect.
    def execute(self, query, params=None):
        return []  # content-hash prefetch -> nothing stored -> full write

    def execute_batch(self, query, batch):
        import re as _re

        node_label = _re.search(r"MERGE \(n:([^\s{]+)", query)
        rel_type = _re.search(r"MERGE \(s\)-\[r:([^\]]+)\]", query)
        if node_label:
            label = node_label.group(1).strip("`")
            for row in batch or []:
                props = {k: v for k, v in row.items() if k != "id"}
                self.nodes.append((row.get("id"), label, props))
        elif rel_type:
            rel = rel_type.group(1).strip("`")
            for row in batch or []:
                self.edges.append((row.get("source"), row.get("target"), rel))
        return []


def test_brier_local_matches_formula():
    # forecasts perfectly right -> 0; coin flips -> 0.25.
    assert brier_score([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert brier_score([0.5, 0.5], [1.0, 0.0]) == 0.25


def test_perfect_agent_high_calibration():
    t = CalibrationTracker()
    for i in range(5):
        t.record_call("good", direction=1, confidence=0.9, subject=f"s{i}")
        t.record_outcome("good", realized_direction=1, subject=f"s{i}")
    sc = t.score("good")
    assert sc.n_calls == 5
    assert sc.accuracy == 1.0
    assert sc.calibration > 0.5
    assert sc.brier < 0.25


def test_wrong_overconfident_agent_low_calibration():
    t = CalibrationTracker()
    for i in range(5):
        t.record_call("bad", direction=1, confidence=0.9, subject=f"s{i}")
        t.record_outcome("bad", realized_direction=-1, subject=f"s{i}")  # always wrong
    sc = t.score("bad")
    assert sc.accuracy == 0.0
    assert sc.brier > 0.25  # overconfident + wrong
    assert sc.calibration == 0.0


def test_no_calls_neutral_prior():
    sc = CalibrationTracker().score("unknown")
    assert sc.n_calls == 0
    assert sc.calibration == 0.5


def test_calibrated_role_weights_scale_by_reputation():
    t = CalibrationTracker()
    # quant is well-calibrated; sentiment is poorly calibrated.
    for i in range(4):
        t.record_call("quant_01", 1, 0.85, f"q{i}")
        t.record_outcome("quant_01", 1, f"q{i}")
        t.record_call("sent_01", 1, 0.85, f"se{i}")
        t.record_outcome("sent_01", -1, f"se{i}")
    agent_roles = {
        "quant_01": SwarmRole.QUANT_ANALYST,
        "sent_01": SwarmRole.SENTIMENT_ANALYST,
    }
    base = {SwarmRole.QUANT_ANALYST: 1.0, SwarmRole.SENTIMENT_ANALYST: 1.0}
    w = calibrated_role_weights(t, agent_roles, base_weights=base)
    assert w[SwarmRole.QUANT_ANALYST] > w[SwarmRole.SENTIMENT_ANALYST]


def test_live_path_calibration_changes_swarm_decision():
    """LIVE-PATH: applying calibration to a swarm changes its aggregated vote.

    Build a swarm whose only signals are a bullish quant and a bearish sentiment
    agent of equal base weight (net hold). After down-weighting the (proven
    miscalibrated) sentiment agent, the bullish quant should win.
    """

    class _FixedAgent(SwarmAgent):
        def __init__(self, agent_id, role, direction, confidence):
            super().__init__(agent_id, role)
            self._dir = direction
            self._conf = confidence

        def analyze(self, market_data):
            sig = AgentSignal(
                agent_id=self.agent_id,
                role=self.role,
                direction=self._dir,
                confidence=self._conf,
            )
            self._signal_history.append(sig)
            return sig

    agents = [
        _FixedAgent("quant_01", SwarmRole.QUANT_ANALYST, +1, 0.8),
        _FixedAgent("sent_01", SwarmRole.SENTIMENT_ANALYST, -1, 0.8),
        _FixedAgent("trend_01", SwarmRole.TREND_ANALYST, 0, 0.5),
    ]
    swarm = TradingSwarm(agents=agents)
    # Equal base weights so the bull and bear cancel -> HOLD before calibration.
    swarm.config.role_weights = {
        SwarmRole.QUANT_ANALYST: 1.0,
        SwarmRole.SENTIMENT_ANALYST: 1.0,
        SwarmRole.TREND_ANALYST: 1.0,
    }
    before = swarm.analyze({})
    assert before.decision == SwarmDecision.HOLD

    # Sentiment agent has a terrible track record; quant is excellent.
    t = CalibrationTracker()
    for i in range(5):
        t.record_call("quant_01", 1, 0.85, f"q{i}")
        t.record_outcome("quant_01", 1, f"q{i}")
        t.record_call("sent_01", 1, 0.85, f"s{i}")
        t.record_outcome("sent_01", -1, f"s{i}")

    new_weights = apply_calibration_to_swarm(swarm, t)
    assert (
        new_weights[SwarmRole.QUANT_ANALYST] > new_weights[SwarmRole.SENTIMENT_ANALYST]
    )

    after = swarm.analyze({})
    # With the bear voice down-weighted, the bullish quant now carries the vote.
    assert after.weighted_score > before.weighted_score
    assert after.decision in (SwarmDecision.BUY, SwarmDecision.STRONG_BUY)


def test_persist_calibration_to_kg():
    t = CalibrationTracker()
    t.record_call("a1", 1, 0.7, "x")
    t.record_outcome("a1", 1, "x")
    backend = _FakeBackend()
    n, e = t.persist(backend)
    assert n == 1 and e == 1
    assert backend.nodes[0][1] == "AgentCalibration"
    assert backend.edges[0][2] == "CALIBRATION_OF"


def test_persist_none_backend_noop():
    assert CalibrationTracker().persist(None) == (0, 0)
