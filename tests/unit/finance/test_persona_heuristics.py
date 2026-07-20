"""Persona Decision-Heuristic Enrichment tests — CONCEPT:AU-KG.domains.persona-decision-heuristic-enrichment.

Deterministic evaluator, verdict logic (incl. the forensic-short inversion),
unknown-metric handling, and KG seeding.
"""

from __future__ import annotations

import pytest

from agent_utilities.domains.finance.persona_heuristics import (
    PERSONA_HEURISTICS,
    evaluate_all,
    evaluate_persona,
    list_personas,
    persona_heuristics_batch,
    seed_persona_heuristics,
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


def test_all_personas_have_weighted_frameworks():
    for persona, rules in PERSONA_HEURISTICS.items():
        assert rules, f"{persona} has no rules"
        assert abs(sum(r.weight for r in rules) - 1.0) < 1e-6, persona


def test_graham_passes_deep_value():
    metrics = {"pe": 11.0, "pb": 1.1, "margin_of_safety": 0.40, "current_ratio": 2.5}
    ev = evaluate_persona("graham_investor", metrics)
    assert ev.verdict == "bullish"
    assert ev.score == pytest.approx(1.0)
    assert all(r.status == "pass" for r in ev.results)


def test_graham_fails_expensive():
    metrics = {"pe": 40.0, "pb": 6.0, "margin_of_safety": -0.10, "current_ratio": 1.0}
    ev = evaluate_persona("graham_investor", metrics)
    assert ev.verdict == "bearish"
    assert ev.score == pytest.approx(0.0)


def test_buffett_quality_business():
    metrics = {
        "roe": 0.22,
        "roic": 0.18,
        "de_ratio": 0.3,
        "owner_earnings_yield": 0.06,
        "earnings_positive_years": 10,
    }
    ev = evaluate_persona("buffett_investor", metrics)
    assert ev.verdict == "bullish"


def test_burry_short_inversion():
    # A broken company SATISFIES Burry's forensic triggers -> bearish, not bullish.
    metrics = {"accruals_ratio": 0.20, "m_score": -1.0, "z_score": 1.0, "pe": 60.0}
    ev = evaluate_persona("burry_investor", metrics)
    assert ev.score == pytest.approx(1.0)
    assert ev.verdict == "bearish"  # inverted because satisfying = broken


def test_burry_clean_company_is_bullish_for_short_lens():
    metrics = {"accruals_ratio": 0.01, "m_score": -3.0, "z_score": 4.0, "pe": 15.0}
    ev = evaluate_persona("burry_investor", metrics)
    assert ev.score == pytest.approx(0.0)
    assert ev.verdict == "bullish"  # nothing broken -> no short


def test_unknown_metric_is_not_silent_pass():
    ev = evaluate_persona("graham_investor", {})  # no metrics at all
    assert ev.verdict == "insufficient_data"
    assert all(r.status == "unknown" for r in ev.results)


def test_lynch_peg_lens():
    ev = evaluate_persona(
        "lynch_investor", {"peg": 0.7, "earnings_growth": 0.20, "de_ratio": 0.4}
    )
    assert ev.verdict == "bullish"


def test_evaluate_all_covers_every_persona():
    out = evaluate_all({"pe": 12.0, "roe": 0.2})
    assert set(out) == set(list_personas())


def test_explain_and_citation_are_grounded():
    ev = evaluate_persona("graham_investor", {"pe": 11.0, "pb": 1.1})
    cite = ev.citation()
    assert "GrahamInvestor" in cite
    assert "AU-KG.domains.persona-decision-heuristic-enrichment" in cite
    # each explain line names the numeric threshold
    assert any("pe < 15" in r.explain() for r in ev.results)


def test_batch_and_seed_to_kg():
    batch = persona_heuristics_batch()
    total_rules = sum(len(r) for r in PERSONA_HEURISTICS.values())
    assert len(batch.nodes) == total_rules
    assert all(n.type == "DecisionHeuristic" for n in batch.nodes)
    assert all(e.rel_type == "HEURISTIC_OF" for e in batch.edges)

    backend = _FakeBackend()
    n, e = seed_persona_heuristics(backend)
    assert n == total_rules
    assert e == total_rules


def test_seed_none_backend_noop():
    assert seed_persona_heuristics(None) == (0, 0)


def test_debate_engine_wires_heuristic_evidence():
    """LIVE-PATH: the DebateEngine folds a bound persona's heuristic verdict into
    its prompt block (KG-2.28 wired into the existing debate machinery)."""
    from agent_utilities.domains.finance.debate_engine import (
        DebateContext,
        DebateEngine,
    )

    eng = DebateEngine.with_personas(bull="graham_investor", bear="burry_investor")
    ctx = DebateContext(
        ticker="ACME",
        asset_class="equity",
        metrics={"pe": 11.0, "pb": 1.1, "margin_of_safety": 0.4, "current_ratio": 2.5},
    )
    # Graham's bull-side heuristic verdict appears in the prompt fragment.
    block = eng._heuristic_block(eng.bull_persona, ctx)
    assert "GrahamInvestor" in block
    assert "AU-KG.domains.persona-decision-heuristic-enrichment" in block
    # No metrics -> no heuristic block (generic path unaffected).
    assert eng._heuristic_block(eng.bull_persona, DebateContext("X", "equity")) == ""
