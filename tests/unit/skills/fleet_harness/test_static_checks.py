"""Static layer: each test proves the harness actually DETECTS a broken skill.

Per-repo hard requirement (no masking): a malformed fixture must produce a
FAIL with the specific rule + message — never a silently-passing or
generically-swallowed result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_utilities.skills.fleet_harness.discovery import SkillRecord, discover_skills
from agent_utilities.skills.fleet_harness.static_checks import (
    parse_frontmatter,
    run_static_checks,
    validate_skill,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _record_for(name: str, *, repo_name: str = "fixtures") -> SkillRecord:
    skill_md = _FIXTURES / name / "SKILL.md"
    return SkillRecord(
        skill_md=skill_md,
        skill_dir=skill_md.parent,
        repo_root=_FIXTURES,
        repo_name=repo_name,
    )


class TestParseFrontmatter:
    def test_no_frontmatter_block(self):
        data, _body, error = parse_frontmatter("# just a heading\n")
        assert data is None
        assert "no `---` frontmatter block" in error

    def test_invalid_yaml(self):
        data, _body, error = parse_frontmatter("---\n[unterminated\n---\nbody")
        assert data is None
        assert "not valid YAML" in error

    def test_yaml_not_a_mapping(self):
        data, _body, error = parse_frontmatter("---\n- a\n- b\n---\nbody")
        assert data is None
        assert "did not parse to a mapping" in error

    def test_valid(self):
        data, body, error = parse_frontmatter(
            "---\nname: x\ndescription: y\n---\nbody text"
        )
        assert error is None
        assert data == {"name": "x", "description": "y"}
        assert body.strip() == "body text"


class TestGoodSkillPasses:
    def test_good_skill_is_all_pass(self):
        report = validate_skill(_record_for("good-skill"))
        assert report.status == "PASS", [
            f"{c.rule}: {c.message}" for c in report.failures
        ]
        assert report.name == "good-skill"


class TestDetectsMissingFrontmatter:
    def test_fails_with_specific_reason(self):
        report = validate_skill(_record_for("bad_missing_frontmatter"))
        assert report.status == "FAIL"
        rules = {c.rule for c in report.failures}
        assert "frontmatter.parses" in rules


class TestDetectsMissingSkillType:
    def test_fails_on_missing_skill_type(self):
        report = validate_skill(_record_for("bad_missing_skill_type"))
        assert report.status == "FAIL"
        rules = {c.rule: c.message for c in report.failures}
        assert "frontmatter.skill_type_present" in rules
        assert "skill_type" in rules["frontmatter.skill_type_present"]


class TestDetectsNameMismatch:
    def test_fails_when_name_does_not_match_directory(self):
        report = validate_skill(_record_for("bad_name_mismatch"))
        assert report.status == "FAIL"
        rules = {c.rule for c in report.failures}
        assert "frontmatter.name_matches_directory" in rules

    def test_graph_type_is_exempt_from_name_match(self):
        record = SkillRecord(
            skill_md=_FIXTURES / "graph_node" / "qualified-name-ok" / "SKILL.md",
            skill_dir=_FIXTURES / "graph_node" / "qualified-name-ok",
            repo_root=_FIXTURES,
            repo_name="fixtures",
        )
        report = validate_skill(record)
        assert report.status == "PASS", [
            f"{c.rule}: {c.message}" for c in report.failures
        ]


class TestDetectsBrokenFileReference:
    def test_fails_on_missing_referenced_script(self):
        report = validate_skill(_record_for("bad_broken_reference"))
        assert report.status == "FAIL"
        rules = {c.rule: c.message for c in report.failures}
        assert "structure.referenced_files_exist" in rules
        assert "does_not_exist.py" in rules["structure.referenced_files_exist"]

    def test_passes_when_reference_exists(self):
        report = validate_skill(_record_for("good-skill"))
        integrity = [
            c for c in report.checks if c.rule == "structure.referenced_files_exist"
        ]
        assert integrity and integrity[0].status == "PASS"


class TestDetectsUnresolvableGraphosTool:
    def test_fails_for_au_repo_skill_referencing_unknown_tool(self):
        record = _record_for(
            "bad_unresolvable_graphos_tool", repo_name="agent-utilities"
        )
        report = validate_skill(record)
        assert report.status == "FAIL"
        rules = {c.rule: c.message for c in report.failures}
        assert "tools.graphos_references_resolve" in rules
        assert (
            "graph_this_tool_does_not_exist"
            in rules["tools.graphos_references_resolve"]
        )

    def test_not_checked_for_non_graphos_repos(self):
        # Same body, but reported under a fleet repo name — the canonical
        # graph-os surface is not this skill's contract, so the rule does not
        # even run (no false FAIL against an unrelated tool namespace).
        record = _record_for(
            "bad_unresolvable_graphos_tool", repo_name="some-other-agent-repo"
        )
        report = validate_skill(record)
        rules = {c.rule for c in report.checks}
        assert "tools.graphos_references_resolve" not in rules


class TestDetectsUnresolvablePackageTool:
    def test_fails_when_declared_tool_not_in_package_source(self):
        record = SkillRecord(
            skill_md=_FIXTURES
            / "fake_package"
            / "skills"
            / "bad-tool-skill"
            / "SKILL.md",
            skill_dir=_FIXTURES / "fake_package" / "skills" / "bad-tool-skill",
            repo_root=_FIXTURES,
            repo_name="fixtures",
        )
        report = validate_skill(record)
        assert report.status == "FAIL"
        rules = {c.rule: c.message for c in report.failures}
        assert "tools.package_references_resolve" in rules
        assert "nonexistent_tool" in rules["tools.package_references_resolve"]

    def test_passes_when_declared_tool_exists_in_package(self):
        record = SkillRecord(
            skill_md=_FIXTURES
            / "fake_package"
            / "skills"
            / "good-tool-skill"
            / "SKILL.md",
            skill_dir=_FIXTURES / "fake_package" / "skills" / "good-tool-skill",
            repo_root=_FIXTURES,
            repo_name="fixtures",
        )
        report = validate_skill(record)
        assert report.status == "PASS", [
            f"{c.rule}: {c.message}" for c in report.failures
        ]


class TestNameUniqueness:
    def test_duplicate_names_both_fail(self):
        records = [_record_for("dup_skills/dup-a"), _record_for("dup_skills/dup-b")]
        reports = run_static_checks(records)
        assert all(r.status == "FAIL" for r in reports)
        for report in reports:
            rules = {c.rule: c.message for c in report.failures}
            assert "frontmatter.name_unique" in rules
            assert "2 skills" in rules["frontmatter.name_unique"]

    def test_unique_names_pass_the_uniqueness_rule(self):
        records = [_record_for("good-skill")]
        reports = run_static_checks(records)
        (report,) = reports
        unique_checks = [
            c for c in report.checks if c.rule == "frontmatter.name_unique"
        ]
        assert unique_checks and unique_checks[0].status == "PASS"

    def test_graph_type_node_is_exempt_from_uniqueness_even_when_name_collides(
        self, tmp_path: Path
    ):
        # A skill-graph page legitimately mirrors its parent topic's exact
        # name (e.g. a "deployment" page inside the "agent-utilities"
        # skill-graph is itself named `agent-utilities-deployment`, matching
        # the real atomic skill it documents) — this must NOT be flagged as
        # a fleet-wide collision the way two atomic skills sharing a name
        # would be.
        atomic_dir = tmp_path / "shared-topic"
        atomic_dir.mkdir()
        (atomic_dir / "SKILL.md").write_text(
            "---\nname: shared-topic\nskill_type: skill\ndescription: atomic skill.\n---\nbody\n"
        )
        graph_dir = tmp_path / "graph-page"
        graph_dir.mkdir()
        (graph_dir / "SKILL.md").write_text(
            "---\nname: shared-topic\nskill_type: graph\ndescription: reference page.\n---\nbody\n"
        )
        atomic = SkillRecord(
            skill_md=atomic_dir / "SKILL.md",
            skill_dir=atomic_dir,
            repo_root=tmp_path,
            repo_name="tmp",
        )
        graph_node = SkillRecord(
            skill_md=graph_dir / "SKILL.md",
            skill_dir=graph_dir,
            repo_root=tmp_path,
            repo_name="tmp",
        )
        reports = run_static_checks([atomic, graph_node])
        atomic_report = next(r for r in reports if r.record is atomic)
        graph_report = next(r for r in reports if r.record is graph_node)

        graph_unique = [
            c for c in graph_report.checks if c.rule == "frontmatter.name_unique"
        ]
        assert graph_unique and graph_unique[0].status == "PASS"
        assert "exempt" in graph_unique[0].message

        # The atomic skill's own uniqueness check is unaffected — the graph
        # node was never in its collision pool, so it PASSes cleanly too.
        atomic_unique = [
            c for c in atomic_report.checks if c.rule == "frontmatter.name_unique"
        ]
        assert atomic_unique and atomic_unique[0].status == "PASS"


class TestDescriptionPortabilityWarnings:
    def test_long_description_warns_but_does_not_fail(self, tmp_path: Path):
        skill_dir = tmp_path / "long-desc-skill"
        skill_dir.mkdir()
        long_desc = "x" * 1200
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: long-desc-skill\nskill_type: skill\ndescription: {long_desc}\n---\nbody\n"
        )
        record = SkillRecord(
            skill_md=skill_dir / "SKILL.md",
            skill_dir=skill_dir,
            repo_root=tmp_path,
            repo_name="tmp",
        )
        report = validate_skill(record)
        assert report.status == "PASS"
        warnings = {c.rule for c in report.warnings}
        assert "frontmatter.description_portable_length" in warnings


@pytest.mark.parametrize(
    "fixture_name",
    [
        "bad_missing_frontmatter",
        "bad_missing_skill_type",
        "bad_name_mismatch",
        "bad_broken_reference",
    ],
)
def test_every_bad_fixture_fails(fixture_name: str):
    """Meta-test: every fixture deliberately named `bad_*` must FAIL — the
    harness's whole purpose is catching these, so a false PASS here is a
    harness bug, not a skill bug."""
    report = validate_skill(_record_for(fixture_name))
    assert report.status == "FAIL"


def test_full_fixture_tree_run_is_deterministic():
    records = discover_skills([_FIXTURES])
    first = run_static_checks(records)
    second = run_static_checks(records)
    assert [(r.record.relative_path, r.status) for r in first] == [
        (r.record.relative_path, r.status) for r in second
    ]
