"""Recursive-improvement velocity ledger (CONCEPT:AHE-3.26, SAFE-1.3).

Reads the self-evolution loop's own persisted audit streams (EvolutionCycle +
ProposalPublication + CapabilityRatchetResult) back into one velocity reading: cycle
cadence, the genotypic-vs-prose mechanism split, capability pass-rate, and an
improving/stalling verdict whose signals flag the paper's research-gets-harder modes.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.research.improvement_ledger import (
    ImprovementLedger,
    improvement_velocity,
)

pytestmark = pytest.mark.concept("AHE-3.26")


class LedgerEngine:
    """Fake engine answering the three label queries the ledger issues."""

    def __init__(self, cycles=(), pubs=(), ratchet=()):
        self.nodes: dict[str, dict] = {}
        self._cycles = list(cycles)
        self._pubs = list(pubs)
        self._ratchet = list(ratchet)

    def add_node(self, node_id, node_type, properties=None):
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def query_cypher(self, query, params=None):
        if "EvolutionCycle" in query:
            return [
                {"ts": ts, "metadata": json.dumps({"duration_ms": ms})}
                for ts, ms in self._cycles
            ]
        if "ProposalPublication" in query:
            return [{"kind": k, "ok": ok, "ts": ts} for k, ok, ts in self._pubs]
        if "CapabilityRatchetResult" in query:
            return [{"result": res, "ts": ts} for res, ts in self._ratchet]
        return []


def test_empty_is_idle():
    v = ImprovementLedger(LedgerEngine()).summarize()
    assert v.verdict == "idle" and v.cycles == 0
    assert v.capability_pass_rate == 1.0  # vacuous


def test_improving_with_code_and_passes():
    eng = LedgerEngine(
        cycles=[("2026-06-13T01", 100.0), ("2026-06-13T02", 110.0)],
        pubs=[("code", True, "t1"), ("code", True, "t2"), ("sdd_plan", True, "t3")],
        ratchet=[("pass", "t1"), ("pass", "t2")],
    )
    v = ImprovementLedger(eng).summarize()
    assert v.verdict == "improving"
    assert v.code_publications == 2 and v.prose_publications == 1
    assert round(v.code_fraction, 2) == 0.67
    assert v.capability_pass_rate == 1.0


@pytest.mark.concept("SAFE-1.3")
def test_prose_only_flags_no_genotypic():
    eng = LedgerEngine(
        cycles=[("a", 100.0)],
        pubs=[("sdd_plan", True, "t1"), ("sdd_plan", True, "t2")],
    )
    v = ImprovementLedger(eng).summarize()
    assert v.verdict == "stalling"
    assert any("prose-only" in s for s in v.signals)
    assert v.code_fraction == 0.0


@pytest.mark.concept("SAFE-1.3")
def test_recent_holds_flag_regression_pressure():
    eng = LedgerEngine(
        cycles=[("a", 100.0)],
        pubs=[("code", True, "t1")],
        ratchet=[("pass", "t1"), ("hold", "t2"), ("hold", "t3")],
    )
    v = ImprovementLedger(eng).summarize()
    assert v.verdict == "stalling"
    assert any("regressions outnumber" in s for s in v.signals)
    assert v.capability_pass == 1 and v.capability_hold == 2


@pytest.mark.concept("SAFE-1.3")
def test_rising_cadence_flags_research_harder():
    eng = LedgerEngine(
        # 10 cycles: first 5 fast (~100ms), last 5 slow (~400ms) → recent > 1.5×prior.
        cycles=[(f"t{i:02d}", 100.0 if i < 5 else 400.0) for i in range(10)],
        pubs=[("code", True, "t1")],
        ratchet=[("pass", "t1")],
    )
    v = ImprovementLedger(eng).summarize(recent=5)
    assert v.recent_cycle_ms > v.prior_cycle_ms * 1.5
    assert v.verdict == "stalling"
    assert any("research-gets-harder" in s for s in v.signals)


def test_record_persists_velocity_node_and_module_fn():
    eng = LedgerEngine(
        cycles=[("a", 100.0)], pubs=[("code", True, "t1")], ratchet=[("pass", "t1")]
    )
    v = ImprovementLedger(eng).record()
    nodes = [n for n in eng.nodes.values() if n["type"] == "ImprovementVelocity"]
    assert nodes and json.loads(nodes[0]["metrics_json"])["verdict"] == v.verdict
    # module-level read returns the same shape
    d = improvement_velocity(eng)
    assert d["cycles"] == 1 and "verdict" in d
