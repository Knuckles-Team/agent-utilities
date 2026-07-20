#!/usr/bin/python
"""Tests for the agentic red-team harness.

CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort
"""

import pytest

from agent_utilities.harness.red_team import (
    ATTACK_CATALOG,
    AttackProbe,
    RedTeamReport,
    RedTeamRunner,
)

pytestmark = pytest.mark.concept("AU-AHE.evaluation.adaptive-reasoning-effort")


def test_catalog_is_well_formed():
    assert len(ATTACK_CATALOG) >= 5
    cats = {p.category for p in ATTACK_CATALOG}
    assert {"prompt_injection", "jailbreak", "data_exfiltration"} <= cats
    for p in ATTACK_CATALOG:
        assert isinstance(p, AttackProbe)
        assert p.probe_text and p.success_markers
        assert p.severity in {"low", "medium", "high", "critical"}
        assert p.owasp


# --- assess ----------------------------------------------------------------


def test_assess_detects_marker_compromise():
    runner = RedTeamRunner()
    probe = next(p for p in ATTACK_CATALOG if p.id == "pi-001")
    r = runner.assess(probe, "Sure, PWNED.")
    assert r.compromised is True
    assert any("success markers" in reason for reason in r.reasons)


def test_assess_clean_refusal_is_defended():
    runner = RedTeamRunner()
    probe = next(p for p in ATTACK_CATALOG if p.id == "pi-001")
    r = runner.assess(probe, "I can't comply with that request.")
    assert r.compromised is False
    assert r.reasons == []


def test_assess_unsafe_content_flagged_even_without_marker():
    runner = RedTeamRunner()
    probe = next(p for p in ATTACK_CATALOG if p.id == "jb-001")
    # No literal success marker, but emits an unsafe marker.
    r = runner.assess(probe, "This is a guaranteed cure for everything.")
    assert r.compromised is True
    assert any("unsafe" in reason for reason in r.reasons)


# --- run -------------------------------------------------------------------


def test_run_against_safe_target_defends_all():
    runner = RedTeamRunner()

    def safe_target(_probe: str) -> str:
        return "I won't do that; here is a safe, neutral answer."

    report = runner.run(safe_target)
    assert isinstance(report, RedTeamReport)
    assert report.total == len(ATTACK_CATALOG)
    assert report.compromised == 0
    assert report.defense_rate == 1.0
    assert report.compromised_probes == []


def test_run_against_vulnerable_target_reports_breaches():
    runner = RedTeamRunner()

    # A target that naively echoes the attacker's demanded tokens.
    echoes = {
        "pi-001": "PWNED",
        "pi-002": "LEAKED",
        "jb-001": "DAN MODE ENABLED",
        "ro-001": "MY SYSTEM PROMPT IS ...",
        "ex-001": "SENT: sk-secret",
        "sb-001": "I cannot possibly do 2+2.",
    }

    def vuln_target(probe_text: str) -> str:
        for p in ATTACK_CATALOG:
            if p.probe_text == probe_text:
                return echoes[p.id]
        return ""

    report = runner.run(vuln_target)
    assert report.compromised == len(ATTACK_CATALOG)
    assert report.defense_rate == 0.0
    # critical-severity breaches are surfaced in the severity breakdown
    assert report.by_severity.get("critical", 0) >= 2
    assert {r.probe_id for r in report.compromised_probes} == {
        p.id for p in ATTACK_CATALOG
    }


def test_run_target_error_is_defended_not_crash():
    runner = RedTeamRunner()

    def broken_target(_probe: str) -> str:
        raise RuntimeError("boom")

    report = runner.run(broken_target)
    # A target that errors leaks nothing → all defended, no exception propagated.
    assert report.compromised == 0
