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
    validate,
)


def test_prebundled_skill_suite_is_valid():
    assert len(EXPECTED_SKILLS) == 10
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
