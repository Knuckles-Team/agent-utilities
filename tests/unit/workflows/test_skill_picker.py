"""CONCEPT:AU-ORCH.adapter.hot-cache-invalidation (extension) — Scenario taxonomy + eval-scored skill picker tests."""

from __future__ import annotations

import pytest

from agent_utilities.workflows.skill_picker import (
    SkillCandidate,
    SkillPicker,
    infer_scenario,
)

pytestmark = pytest.mark.concept(id="AU-ORCH.adapter.hot-cache-invalidation")


def test_infer_scenario():
    assert infer_scenario("Generate an invoice and reconcile the ledger") == "finance"
    assert infer_scenario("deploy the api and run the test suite") == "engineering"


def test_resolved_scenario_uses_declared_first():
    c = SkillCandidate(name="x", description="invoice budget", scenario="design")
    assert c.resolved_scenario() == "design"  # declared wins over inference


def test_success_rate_breaks_keyword_ties():
    a = SkillCandidate(name="alpha", description="build a landing page")
    b = SkillCandidate(name="beta", description="build a landing page")
    rates = {"alpha": 0.9, "beta": 0.1}
    picker = SkillPicker(success_rate=lambda n: rates[n])
    best = picker.pick("build a landing page", [a, b])
    assert best is not None
    assert best.name == "alpha"  # higher historical success ranks first


def test_cold_skill_neutral_prior():
    picker = SkillPicker()  # default 0.5 prior, no crash on unknown skills
    best = picker.pick(
        "anything", [SkillCandidate(name="cold", description="cold skill")]
    )
    assert best is not None
    assert best.name == "cold"


def test_scenario_filter_narrows_candidates():
    fin = SkillCandidate(
        name="inv", description="invoice generator", scenario="finance"
    )
    eng = SkillCandidate(
        name="dep", description="deploy service", scenario="engineering"
    )
    picker = SkillPicker()
    ranked = picker.rank("make something", [fin, eng], scenario="finance")
    assert [s.candidate.name for s in ranked] == ["inv"]


def test_keyword_overlap_contributes():
    picker = SkillPicker(success_rate=lambda n: 0.5)
    match = SkillCandidate(name="dashboard", description="build a kanban dashboard")
    other = SkillCandidate(name="poster", description="design a poster")
    best = picker.pick("kanban dashboard", [match, other])
    assert best is not None
    assert best.name == "dashboard"


def test_critique_policy_carried_on_candidate():
    c = SkillCandidate(name="x", critique_policy="opt-out")
    assert c.critique_policy == "opt-out"
