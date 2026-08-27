"""Regression gate for the consolidated pre-bundled skill suite."""

import tomllib
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from agent_utilities.mcp.skill_coverage import parse_graph_os_sidecar
from agent_utilities.mcp.tool_specs import (
    INTENT_VERBS,
    TOOL_SPECS,
    TOOL_SPECS_BY_NAME,
    canonical_tool_names,
)
from agent_utilities.skills.validation import (
    EXPECTED_SKILLS,
    FORWARD_MATRIX,
    SKILLS_ROOT,
    _validate_skill,
    validate,
)


def test_prebundled_skill_suite_is_valid():
    assert len(EXPECTED_SKILLS) == 13
    assert validate() == []


def test_validation_contract_is_distribution_owned() -> None:
    assert FORWARD_MATRIX.parent == SKILLS_ROOT
    assert FORWARD_MATRIX.is_file()
    project = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    scripts = project["project"]["scripts"]
    assert scripts["agent-utilities-validate-skills"] == (
        "agent_utilities.skills.runtime_validation:main"
    )


def test_tool_spec_universe_is_immutable_and_profile_aware() -> None:
    core = canonical_tool_names()
    granular = canonical_tool_names(include_intent=False)
    finance = canonical_tool_names(features=frozenset({"finance"}))

    assert set(INTENT_VERBS) <= core
    assert core - granular == set(INTENT_VERBS)
    assert "quant" not in core
    assert finance - core == {"quant"}
    assert len(TOOL_SPECS) == len(TOOL_SPECS_BY_NAME)
    with pytest.raises(TypeError):
        TOOL_SPECS_BY_NAME["new_tool"] = TOOL_SPECS[0]  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        TOOL_SPECS[0].name = "changed"  # type: ignore[misc]


def test_validate_skill_flags_every_broken_facet_of_a_malformed_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WB1-AU-02 characterization: the only pre-existing coverage of
    `_validate_skill` is `test_prebundled_skill_suite_is_valid` above, which
    only proves the real 13-skill suite is 100% clean (0 errors) -- it never
    exercises a single one of `_validate_skill`'s ~20 error-append branches
    (now split across `_validate_skill_frontmatter`/`_body`/
    `_workflow_terms`/`_openai_sidecar`/`_openai_interface`/
    `_graph_os_sidecar`/`_files`). This drives a deliberately-broken
    synthetic skill directory through every one of those branches so the
    extract-method split is proven, not just assumed, to preserve behavior.
    """
    import agent_utilities.skills.validation as validation_mod

    monkeypatch.setattr(validation_mod, "PACKAGE_ROOT", tmp_path)
    skill_dir = tmp_path / "example-skill"
    (skill_dir / "agents").mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\n"
        "name: wrong-name\n"
        "skill_type: not-a-skill\n"
        "---\n"
        "TODO: finish this skill.\n"
        "No workflow section here, no imperative steps, no cost guidance.\n",
        encoding="utf-8",
    )
    # WD1-GATE-01A: this literal used to use a home-directory root, which is
    # exactly the shape `check_tracked_privacy.py`'s own
    # `classify_runtime_source_line` flags as a machine-specific home path
    # in runtime source (it scans tracked .py source text, not just what
    # this write_text call produces at runtime) -- a false-positive collision
    # between two independent scanners, not a real leak. `_validate_skill`'s
    # "absolute filesystem path" pattern
    # (`agent_utilities/skills/validation.py::_PRIVATE_PATTERNS`) matches a
    # much wider set of path roots than the privacy gate's narrow allowlist
    # of home-directory conventions, so a root outside that narrower
    # allowlist still trips the validator under test while never matching
    # the privacy gate's pattern.
    skill_dir.joinpath("README.md").write_text(
        "See /srv/someone/notes for details.\n", encoding="utf-8"
    )

    errors = _validate_skill(skill_dir)

    assert any("frontmatter must contain only name" in e for e in errors)
    assert any("frontmatter name must match directory" in e for e in errors)
    assert any("frontmatter skill_type must be 'skill'" in e for e in errors)
    assert any("description is empty" in e for e in errors)
    assert any("unresolved TODO in SKILL.md" in e for e in errors)
    assert any("must contain a Workflow section" in e for e in errors)
    assert any("at least three imperative steps" in e for e in errors)
    assert any("missing economy-model guidance" in e for e in errors)
    assert any("must explain direct and delegated execution" in e for e in errors)
    assert any("missing agents/openai.yaml" in e for e in errors)
    assert any("missing agents/graph-os.yaml" in e for e in errors)
    assert any("auxiliary skill documentation is forbidden" in e for e in errors)
    assert any("contains absolute filesystem path" in e for e in errors)


def test_graph_os_sidecar_schema_v1_is_rejected(tmp_path: Path) -> None:
    sidecar = tmp_path / "graph-os.yaml"
    sidecar.write_text(
        "schema_version: 1\ntier: domain\nwraps:\n  - graph_query\n",
        encoding="utf-8",
    )
    meta = parse_graph_os_sidecar(sidecar, skill_name="example")
    assert meta.errors
    assert any("schema_version must be 2" in error for error in meta.errors)
    assert any("unsupported keys: ['wraps']" in error for error in meta.errors)
