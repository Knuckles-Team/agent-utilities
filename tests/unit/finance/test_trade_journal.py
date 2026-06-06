"""Trade-Journal Bias Auditor + Shadow Account tests — CONCEPT:KG-2.26.

Real metric maths + KG persistence (via a fake backend), offline.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from agent_utilities.domains.finance.trade_journal import (
    Roundtrip,
    TradeJournalAuditor,
    audit_trade_journal,
)


def _rt(symbol, day, hold_days, pnl, entry=100.0, side="buy"):
    entry_time = datetime(2026, 1, 1) + timedelta(days=day)
    exit_time = entry_time + timedelta(days=hold_days)
    return Roundtrip(
        symbol=symbol,
        entry_time=entry_time.isoformat(),
        exit_time=exit_time.isoformat(),
        entry_price=entry,
        exit_price=entry + pnl / 10,
        size=10,
        pnl=pnl,
        side=side,
    )


class _FakeBackend:
    """Duck-typed GraphBackend capturing add_node/add_edge calls."""

    def __init__(self):
        self.nodes: list[tuple] = []
        self.edges: list[tuple] = []

    def add_node(self, node_id, type=None, **props):
        self.nodes.append((node_id, type, props))

    def add_edge(self, src, tgt, rel_type=None):
        self.edges.append((src, tgt, rel_type))


def test_profile_metrics_are_real():
    rts = [
        _rt("AAA", 0, 2, 300),
        _rt("AAA", 5, 1, 100),
        _rt("BBB", 10, 4, -200),
        _rt("BBB", 20, 3, -50),
    ]
    profile = TradeJournalAuditor().audit("acct_1", rts)
    assert profile.total_roundtrips == 4
    assert profile.win_rate == 0.5  # 2 of 4 positive
    assert profile.total_pnl == 150.0  # 300+100-200-50
    # avg win = 200, avg loss = -125 -> ratio 1.6
    assert round(profile.profit_loss_ratio, 2) == 1.6
    # equity curve 300, 400, 200, 150 -> peak 400, worst trough -250
    assert profile.max_drawdown == -250.0


def test_disposition_effect_detected():
    # Losers held 8d, winners held 1d -> ratio 8 -> high.
    rts = [
        _rt("AAA", 0, 1, 100),
        _rt("AAA", 5, 1, 100),
        _rt("BBB", 10, 8, -100),
        _rt("BBB", 25, 8, -100),
    ]
    profile = TradeJournalAuditor().audit("acct_2", rts)
    disp = profile.bias("disposition_effect")
    assert disp is not None
    assert disp.severity == "high"
    assert disp.metric >= 1.5


def test_momentum_chasing_detected():
    # Repeated buys of same symbol at ever-higher prices -> chase.
    rts = [
        _rt("AAA", 0, 1, 10, entry=100.0),
        _rt("AAA", 2, 1, 10, entry=110.0),
        _rt("AAA", 4, 1, 10, entry=125.0),
        _rt("AAA", 6, 1, 10, entry=140.0),
    ]
    profile = TradeJournalAuditor().audit("acct_3", rts)
    chase = profile.bias("momentum_chasing")
    assert chase is not None
    assert chase.severity in ("medium", "high")
    assert chase.metric > 0.5


def test_anchoring_detected():
    # >=5 trades of one symbol all within a <5% price band -> anchored.
    rts = [_rt("AAA", i, 1, 5, entry=100.0 + (i % 2)) for i in range(6)]
    profile = TradeJournalAuditor().audit("acct_4", rts)
    anchor = profile.bias("anchoring")
    assert anchor is not None
    assert anchor.severity == "high"


def test_empty_journal_degrades():
    profile = TradeJournalAuditor().audit("empty", [])
    assert profile.total_roundtrips == 0
    assert profile.win_rate == 0.0


def test_persist_to_kg_backend():
    rts = [_rt("AAA", 0, 2, 300), _rt("BBB", 5, 4, -100)]
    auditor = TradeJournalAuditor()
    profile = auditor.audit("acct_kg", rts)
    backend = _FakeBackend()
    n_nodes, n_edges = auditor.persist(profile, backend)
    # 1 profile node + 4 bias nodes; 4 EXHIBITED_BY edges.
    assert n_nodes == 5
    assert n_edges == 4
    node_types = {t for _, t, _ in backend.nodes}
    assert node_types == {"TraderProfile", "BehavioralBias"}
    assert all(rel == "EXHIBITED_BY" for _, _, rel in backend.edges)


def test_persist_none_backend_is_noop():
    profile = TradeJournalAuditor().audit("x", [_rt("AAA", 0, 1, 10)])
    assert TradeJournalAuditor().persist(profile, None) == (0, 0)


def test_convenience_audit_with_dicts_and_citation():
    rts = [
        {
            "symbol": "AAA",
            "entry_time": "2026-01-01T00:00:00",
            "exit_time": "2026-01-03T00:00:00",
            "entry_price": 100.0,
            "exit_price": 130.0,
            "size": 10,
            "pnl": 300.0,
        }
    ]
    profile = audit_trade_journal("acct_dict", rts)
    assert profile.total_roundtrips == 1
    assert "KG-2.26" in profile.citation()
