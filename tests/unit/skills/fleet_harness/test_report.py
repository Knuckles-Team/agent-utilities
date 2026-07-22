"""Report rendering: JSON + human table, both reflecting real FAILs."""

from __future__ import annotations

import json
from pathlib import Path

from agent_utilities.skills.fleet_harness.discovery import discover_skills
from agent_utilities.skills.fleet_harness.functional_checks import FunctionalResult
from agent_utilities.skills.fleet_harness.report import (
    build_report,
    render_table,
    write_json,
    write_table,
)
from agent_utilities.skills.fleet_harness.static_checks import run_static_checks

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _fixture_reports():
    records = discover_skills([_FIXTURES])
    return run_static_checks(records)


def test_build_report_tallies_pass_and_fail_correctly():
    reports = _fixture_reports()
    report = build_report(reports)
    summary = report["summary"]
    assert summary["total_skills"] == len(reports)
    assert summary["static_pass"] + summary["static_fail"] == len(reports)
    assert summary["static_fail"] > 0  # the fixture tree deliberately has broken skills
    assert summary["static_pass"] > 0  # and at least one clean skill


def test_report_includes_functional_layer_when_provided():
    reports = _fixture_reports()
    functional = [
        FunctionalResult(
            skill=r.record.relative_path,
            status="SKIPPED-not-applicable",
            detail="no refs",
        )
        for r in reports
    ]
    report = build_report(reports, functional)
    assert report["summary"]["functional_tally"] == {
        "SKIPPED-not-applicable": len(reports)
    }
    for skill in report["skills"]:
        assert skill["functional_status"] == "SKIPPED-not-applicable"


def test_report_omits_functional_layer_when_not_run():
    reports = _fixture_reports()
    report = build_report(reports)
    assert report["summary"]["functional_tally"] == {}
    for skill in report["skills"]:
        assert skill["functional_status"] == "SKIPPED-no-functional-run"


def test_json_report_round_trips(tmp_path: Path):
    reports = _fixture_reports()
    report = build_report(reports)
    out = tmp_path / "report.json"
    write_json(report, out)
    loaded = json.loads(out.read_text())
    assert loaded["summary"] == report["summary"]


def test_table_lists_every_failing_skill_with_its_reason():
    reports = _fixture_reports()
    report = build_report(reports)
    table = render_table(report)
    fail_names = {r.record.relative_path for r in reports if r.status == "FAIL"}
    for name in fail_names:
        assert name in table
    assert "Static FAIL detail" in table
    # A concrete, named violation must actually appear, not a generic marker.
    assert "frontmatter.skill_type_present" in table


def test_write_table_creates_parent_dirs(tmp_path: Path):
    reports = _fixture_reports()
    report = build_report(reports)
    out = tmp_path / "nested" / "dir" / "table.md"
    write_table(report, out)
    assert out.exists()
    assert out.read_text().startswith("# Skill validation harness")
