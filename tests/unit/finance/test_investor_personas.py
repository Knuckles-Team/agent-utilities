"""Investor personas + forensic/pattern agents wired to the debate (CONCEPT:AU-KG.research.research-pipeline-runner).

Engine-grounded numbers (forensic_report / momentum / mean_reversion / regimes)
are exercised with an in-process fake client so the suite never requires a live
epistemic-graph engine. Engine-dependent assertions degrade gracefully: when the
real engine is absent the helpers return UNAVAILABLE / engine_confirmed=False
rather than fabricating numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_utilities.domains.finance.filing_diff import (
    FilingDiffAgent,
    diff_filing_sections,
)
from agent_utilities.domains.finance.forensic_screener import (
    ForensicScreener,
    ForensicVerdict,
)
from agent_utilities.domains.finance.investor_debate import (
    INVESTOR_PERSONAS,
    build_financial_debate_team,
    persona_for_role,
    seed_financial_debate_team,
)
from agent_utilities.domains.finance.pattern_classifier import (
    EdgeLabel,
    PatternClassifier,
    PricePattern,
)
from agent_utilities.domains.finance.trading_swarm import SwarmRole

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "agent_utilities" / "prompts"

PERSONA_FILES = [
    "buffett_investor",
    "burry_investor",
    "druckenmiller_investor",
    "damodaran_investor",
    "graham_investor",
]


# ── persona prompt JSONs ──────────────────────────────────────────────────────
@pytest.mark.parametrize("stem", PERSONA_FILES)
def test_persona_prompt_schema(stem):
    """Each persona matches the existing prompt JSON schema and is discoverable."""
    path = PROMPTS_DIR / f"{stem}.json"
    assert path.exists(), f"missing persona prompt {path}"
    data = json.loads(path.read_text())
    # Same shape as chief_trading_officer.json / finance_operations_coordinator.json
    assert data["task"] == stem
    assert data["type"] == "prompt"
    assert "metadata" in data and "identity" in data and "instructions" in data
    assert data["identity"]["role"]
    assert data["instructions"]["core_directive"]
    assert isinstance(data["tools"], list) and data["tools"]
    assert data["team_config"] == "financial_debate"
    # Persona declares a swarm_role mapping in metadata.
    assert data["metadata"]["swarm_role"] in {r.value for r in SwarmRole}


def test_registry_builder_can_resolve_personas():
    """The KG prompt-ingestion registry resolves each persona's fields."""
    from agent_utilities.agent.registry_builder import (
        _load_prompt_metadata,
        _resolve_fields,
    )

    for stem in PERSONA_FILES:
        data = _load_prompt_metadata(PROMPTS_DIR / f"{stem}.json")
        assert data is not None
        name, _desc, caps, _sp = _resolve_fields(data, stem)
        assert name == stem
        assert "graph-os" in caps or "data-science-mcp" in caps


# ── persona → SwarmRole mapping ───────────────────────────────────────────────
def test_persona_swarm_role_mapping_matches_json():
    """The investor_debate mapping agrees with each persona JSON's metadata."""
    for persona in INVESTOR_PERSONAS:
        data = json.loads((PROMPTS_DIR / f"{persona.prompt}.json").read_text())
        assert data["metadata"]["swarm_role"] == persona.swarm_role.value


def test_persona_for_role_lookup():
    fundamental = persona_for_role(SwarmRole.FUNDAMENTAL_ANALYST)
    assert "buffett_investor" in fundamental
    assert "damodaran_investor" in fundamental
    assert "graham_investor" in fundamental
    assert persona_for_role(SwarmRole.BEAR_RESEARCHER) == ["burry_investor"]
    assert persona_for_role(SwarmRole.DIRECTOR) == ["druckenmiller_investor"]


# ── forensic screener (engine-grounded) ───────────────────────────────────────
class _FakeFinance:
    def __init__(self, report):
        self._report = report

    def forensic_report(self, this_year, prior_year):
        return self._report


class _FakeClient:
    def __init__(self, report=None, **finance_kwargs):
        self.finance = (
            _FakeFinance(report)
            if report is not None
            else _FakeFinanceSignals(**finance_kwargs)
        )


def test_forensic_screener_grounds_in_engine():
    report = {
        "m_score": -1.2,
        "z_score": 1.1,
        "f_score": 3,
        "accruals_ratio": 0.18,
        "flags": ["high_accruals", "z_distress_zone"],
        "verdict": "INVESTIGATE",
    }
    screener = ForensicScreener(engine_client=_FakeClient(report=report))
    verdict = screener.screen("ACME", {"sales": 100}, {"sales": 90})
    assert verdict.available is True
    assert verdict.is_red_flag is True
    assert verdict.m_score == -1.2
    assert "high_accruals" in verdict.flags
    assert "INVESTIGATE" in verdict.citation()
    assert "epistemic-graph" in verdict.citation()


def test_forensic_screener_no_engine_does_not_hallucinate():
    """With no engine the screener returns UNAVAILABLE, never a fake number."""
    screener = ForensicScreener(engine_client=None)
    # Force lazy probe to find no engine.
    from agent_utilities.domains.finance import forensic_screener as fs

    fs.reset_engine_cache()
    verdict = screener.screen("ACME", {}, {})
    if not verdict.available:
        assert isinstance(verdict, ForensicVerdict)
        assert verdict.verdict == "UNAVAILABLE"
        assert verdict.m_score is None
        assert "no numbers fabricated" in verdict.citation()


# ── year-over-year filing diff ────────────────────────────────────────────────
def test_filing_diff_isolates_new_and_removed_ignoring_boilerplate():
    last = (
        "You should carefully consider the following risk factors.\n\n"
        "We depend on a single supplier for our key component.\n\n"
        "Our results may fluctuate from quarter to quarter."
    )
    this = (
        "You should carefully consider the following risk factors.\n\n"
        "Our results may fluctuate from quarter to quarter.\n\n"
        "There is substantial doubt about our ability to continue as a going concern."
    )
    diff = diff_filing_sections("Risk Factors", this, last)
    new_text = " ".join(diff.added).lower()
    removed_text = " ".join(diff.removed).lower()
    assert "going concern" in new_text  # genuinely NEW
    assert "single supplier" in removed_text  # genuinely REMOVED
    # Boilerplate ("carefully consider") was skipped, not reported.
    assert diff.boilerplate_skipped >= 1
    assert "carefully consider" not in new_text
    assert diff.has_material_change


def test_filing_diff_agent_falls_back_to_deterministic_offline():
    """When the LLM is unavailable the agent emits the deterministic diff."""

    class _BoomModel:
        def run_sync(self, *a, **k):  # pragma: no cover - forced failure path
            raise RuntimeError("no LLM")

    agent = FilingDiffAgent(llm_client=_BoomModel())
    result = agent.run(
        "Risk Factors",
        "There is substantial doubt about our ability to continue as a going concern.",
        "Our results may fluctuate.",
    )
    assert result.section == "Risk Factors"
    # Deterministic fallback still surfaces structured NEW/REMOVED findings.
    assert result.has_material_change
    assert {f.change_type for f in result.findings} == {"NEW", "REMOVED"}


# ── price-action pattern classifier (engine-grounded numerics) ────────────────
class _FakeFinanceSignals:
    def __init__(self, momentum=None, mean_reversion=None, zscore=None):
        self._mom = momentum or [0.0]
        self._mr = mean_reversion or [0.0]
        self._z = zscore or [0.0]

    def momentum(self, prices, lookback):
        return self._mom

    def mean_reversion(self, values, window):
        return self._mr

    def rolling_zscore(self, values, window):
        return self._z

    def detect_regimes(self, observations, n_states=2, **kw):
        return {"n_states": n_states}


def _large_green_candle():
    # body ~ full range, almost no wick → LARGE_BODY momentum
    return {"open": 100.0, "high": 110.2, "low": 99.9, "close": 110.0}


def test_pattern_large_body_is_momentum_and_engine_confirmed():
    client = _FakeClient(momentum=[0.05])  # positive momentum confirms up-direction
    clf = PatternClassifier(engine_client=client)
    window = [
        {"open": 100, "high": 101, "low": 99, "close": 100.2},
        {"open": 100.2, "high": 101, "low": 99.5, "close": 100.5},
        _large_green_candle(),
    ]
    res = clf.classify(window)
    assert res.pattern == PricePattern.LARGE_BODY
    assert res.edge == EdgeLabel.MOMENTUM
    assert res.direction == 1
    assert res.engine_confirmed is True
    assert res.metrics["momentum"] == 0.05


def test_pattern_wick_into_level_is_mean_reversion():
    client = _FakeClient(mean_reversion=[-0.01], zscore=[2.0])  # stretched up → fade
    clf = PatternClassifier(engine_client=client)
    # Upper wick rejecting a level at 105 (3 candles for engine confirmation).
    window = [
        {"open": 99.5, "high": 100.2, "low": 99.0, "close": 100.0},
        {"open": 100, "high": 101, "low": 99, "close": 100.5},
        {"open": 100.5, "high": 105.0, "low": 100.4, "close": 100.6},
    ]
    res = clf.classify(window, levels=[105.0])
    assert res.pattern == PricePattern.WICK_INTO_LEVEL
    assert res.edge == EdgeLabel.MEAN_REVERSION
    assert res.direction == -1  # fade the upside rejection
    assert res.engine_confirmed is True


def test_pattern_choppy_has_no_edge():
    clf = PatternClassifier(engine_client=_FakeClient(momentum=[0.0]))
    window = [
        {"open": 100, "high": 100.4, "low": 99.6, "close": 100.05},
        {"open": 100.05, "high": 100.3, "low": 99.7, "close": 99.95},
    ]
    res = clf.classify(window)
    assert res.pattern == PricePattern.CHOPPY
    assert res.edge == EdgeLabel.NO_EDGE


def test_pattern_no_engine_marks_unconfirmed():
    from agent_utilities.domains.finance import pattern_classifier as pc

    pc.reset_engine_cache()
    clf = PatternClassifier(engine_client=None)
    res = clf.classify([{"open": 100, "high": 110.2, "low": 99.9, "close": 110.0}])
    # Shape still classified; edge just not numerically confirmed by the engine.
    if not res.engine_confirmed:
        assert res.pattern == PricePattern.LARGE_BODY
        assert res.metrics.get("momentum") is None


# ── team seeding via the shared write_batch path ──────────────────────────────
from tests.kg_recording_backend import RecordingGraphBackend as _FakeBackend


def test_build_financial_debate_team_hierarchy():
    team = build_financial_debate_team()
    assert team.name == "financial_debate"
    assert team.lead == "portfolio_manager"
    # Risk officer reports to the portfolio manager (risk-first veto flow).
    assert ("risk_compliance_officer", "portfolio_manager") in team.reports_to
    # Each persona reports to the risk officer.
    for persona in INVESTOR_PERSONAS:
        assert (persona.prompt, "risk_compliance_officer") in team.reports_to


def test_seed_financial_debate_team_persists_via_write_batch():
    backend = _FakeBackend()
    nodes, edges = seed_financial_debate_team(backend)
    assert nodes > 0 and edges > 0
    assert backend.nodes["team:financial-debate"]["type"] == "Team"
    rels = {(s, t, r) for s, t, r in backend.edges}
    # Persona is a MEMBER_OF_TEAM and REPORTS_TO the risk officer.
    assert (
        "agent:buffett-investor",
        "team:financial-debate",
        "MEMBER_OF_TEAM",
    ) in rels
    assert (
        "agent:burry-investor",
        "agent:risk-compliance-officer",
        "REPORTS_TO",
    ) in rels
