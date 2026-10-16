"""Behavioral contracts for the cross-project version cascade planner."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.release import version_cascade as vc

_PYPROJECT_TEMPLATE = """\
[project]
name = "agent-utilities"
dynamic = ["version"]
dependencies = [
    "epistemic-graph[full]>={engine_version},<{engine_ceiling_major}.0.0",
    "pydantic>=2.0.0",
]
"""

_VERSION_TEMPLATE = '''"""Authoritative package version."""

__version__ = "{au_version}"
'''

_MATRIX_TEMPLATE = """\
apiVersion: graphos.io/v1
kind: CompatibilityMatrix
components:
  epistemic-graph:
    version: "=={engine_version}"
    artifactKind: oci
  agent-utilities:
    version: "=={au_version}"
    artifactKind: oci
    dependsOn:
      epistemic-graph: "=={engine_version}"
  connector-bundles:
    version: "==1"
    artifactKind: catalog
    dependsOn:
      agent-utilities: "=={au_version}"
      epistemic-graph: "=={engine_version}"
  prebundled-skills:
    version: "==1"
    artifactKind: catalog
    dependsOn:
      agent-utilities: "=={au_version}"
releaseTrain:
  assemblyOrder:
    - epistemic-graph
    - agent-utilities
    - connector-bundles
    - prebundled-skills
"""

_PLAIN_PROVIDER = """\
[project]
name = "plain-provider"
dependencies = [ "agent-utilities>=2.0.0,<3.0.0", "requests>=2.0.0",]

[project.optional-dependencies]
mcp = [ "agent-utilities[mcp]>=2.0.0,<3.0.0",]
"""

_EXTRAS_PROVIDER = """\
[project]
name = "extras-provider"
dependencies = [ "agent-utilities[mcp,agent-runtime,logfire]>=2.0.0,<3.0.0",]

[project.optional-dependencies]
agent = [ "agent-utilities[agent-runtime,logfire]>=2.0.0,<3.0.0",]
"""

_MISSING_CEILING_PROVIDER = """\
[project]
name = "missing-ceiling-provider"
dependencies = [ "agent-utilities>=2.0.0", "python-dotenv>=1.0.0",]

[project.optional-dependencies]
mcp = [ "agent-utilities[mcp]>=2.0.0",]
"""


def _write_root(
    tmp_path: Path, *, au_version: str = "2.1.1", engine_version: str = "2.23.2"
) -> Path:
    root = tmp_path / "agent-utilities"
    (root / "agent_utilities").mkdir(parents=True)
    (root / "deploy" / "release").mkdir(parents=True)
    engine_ceiling_major = int(engine_version.split(".")[0]) + 1
    (root / "pyproject.toml").write_text(
        _PYPROJECT_TEMPLATE.format(
            engine_version=engine_version, engine_ceiling_major=engine_ceiling_major
        ),
        encoding="utf-8",
    )
    (root / "agent_utilities" / "_version.py").write_text(
        _VERSION_TEMPLATE.format(au_version=au_version), encoding="utf-8"
    )
    (root / "deploy" / "release" / "compatibility-matrix.yml").write_text(
        _MATRIX_TEMPLATE.format(au_version=au_version, engine_version=engine_version),
        encoding="utf-8",
    )
    return root


def _write_providers(tmp_path: Path, providers: dict[str, str]) -> Path:
    providers_root = tmp_path / "agents"
    providers_root.mkdir(parents=True)
    for name, content in providers.items():
        provider_dir = providers_root / name
        provider_dir.mkdir()
        (provider_dir / "pyproject.toml").write_text(content, encoding="utf-8")
    return providers_root


_CLEAN_PROVIDERS = {
    "plain-provider": _PLAIN_PROVIDER,
    "extras-provider": _EXTRAS_PROVIDER,
}
_ALL_PROVIDERS = {
    **_CLEAN_PROVIDERS,
    "missing-ceiling-provider": _MISSING_CEILING_PROVIDER,
}


def _provider_edits(plan: vc.CascadePlan) -> list[vc.FileEdit]:
    return [edit for edit in plan.edits if edit.tier == 3]


def test_patch_au_bump_produces_zero_provider_edits(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    providers_root = _write_providers(tmp_path, _CLEAN_PROVIDERS)

    plan = vc.plan_cascade(root=root, providers_root=providers_root, au_bump="patch")

    assert plan.severity == "patch"
    assert plan.au_current == "2.1.1"
    assert plan.au_proposed == "2.1.2"
    assert _provider_edits(plan) == []
    assert plan.breaking_notes == ()


def test_minor_au_bump_produces_zero_provider_edits(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    providers_root = _write_providers(tmp_path, _CLEAN_PROVIDERS)

    plan = vc.plan_cascade(root=root, providers_root=providers_root, au_bump="minor")

    assert plan.severity == "minor"
    assert plan.au_proposed == "2.2.0"
    assert _provider_edits(plan) == []


def test_major_au_bump_rewrites_every_provider(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    providers_root = _write_providers(tmp_path, _ALL_PROVIDERS)

    plan = vc.plan_cascade(root=root, providers_root=providers_root, au_version="3.0.0")

    assert plan.severity == "major"
    assert plan.au_current == "2.1.1"
    assert plan.au_proposed == "3.0.0"
    assert plan.breaking_notes != ()
    assert any("provider" in note for note in plan.breaking_notes)

    provider_edits = _provider_edits(plan)
    plain_edits = [
        edit
        for edit in provider_edits
        if edit.path == providers_root / "plain-provider" / "pyproject.toml"
    ]
    plain_edit = next(
        edit for edit in plain_edits if edit.current.startswith("agent-utilities>=")
    )
    assert plain_edit.current == "agent-utilities>=2.0.0,<3.0.0"
    assert plain_edit.proposed == "agent-utilities>=3.0.0,<4.0.0"

    extras_edits = [
        edit
        for edit in provider_edits
        if edit.path == providers_root / "extras-provider" / "pyproject.toml"
    ]
    proposed_texts = {edit.proposed for edit in extras_edits}
    assert "agent-utilities[mcp,agent-runtime,logfire]>=3.0.0,<4.0.0" in proposed_texts
    assert "agent-utilities[agent-runtime,logfire]>=3.0.0,<4.0.0" in proposed_texts
    # extras spelling/order must be preserved verbatim, nothing else in the
    # bracket may move
    for edit in extras_edits:
        current_extras = edit.current.split(">=", 1)[0]
        proposed_extras = edit.proposed.split(">=", 1)[0]
        assert current_extras == proposed_extras

    missing_ceiling_edits = [
        edit
        for edit in provider_edits
        if edit.path == providers_root / "missing-ceiling-provider" / "pyproject.toml"
    ]
    assert {edit.kind for edit in missing_ceiling_edits} == {"missing-ceiling"}
    assert {edit.proposed for edit in missing_ceiling_edits} == {
        "agent-utilities>=3.0.0,<4.0.0",
        "agent-utilities[mcp]>=3.0.0,<4.0.0",
    }


def test_engine_major_bump_rewrites_au_band_and_matrix(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    providers_root = _write_providers(tmp_path, _CLEAN_PROVIDERS)

    plan = vc.plan_cascade(
        root=root, providers_root=providers_root, engine_version="3.0.0"
    )

    assert plan.severity == "major"
    assert plan.engine_current == "2.23.2"
    assert plan.engine_proposed == "3.0.0"
    # a pure engine bump never touches provider files
    assert _provider_edits(plan) == []
    assert any("engine" in note for note in plan.breaking_notes)

    tier2_edits = {edit.kind: edit for edit in plan.edits if edit.tier == 2}
    engine_dependency = tier2_edits["engine-dependency"]
    assert engine_dependency.current == "epistemic-graph[full]>=2.23.2,<3.0.0"
    assert engine_dependency.proposed == "epistemic-graph[full]>=3.0.0,<4.0.0"

    matrix_edit = tier2_edits["compatibility-matrix:epistemic-graph.version"]
    assert '"==2.23.2"' in matrix_edit.current
    assert '"==3.0.0"' in matrix_edit.proposed

    dependency_kinds = {edit.kind for edit in plan.edits if edit.tier == 2}
    assert "compatibility-matrix:agent-utilities.dependsOn.epistemic-graph" in (
        dependency_kinds
    )
    assert "compatibility-matrix:connector-bundles.dependsOn.epistemic-graph" in (
        dependency_kinds
    )


def test_engine_minor_bump_moves_floor_only(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    providers_root = _write_providers(tmp_path, _CLEAN_PROVIDERS)

    plan = vc.plan_cascade(
        root=root, providers_root=providers_root, engine_bump="minor"
    )

    assert plan.severity == "minor"
    assert plan.engine_proposed == "2.24.0"
    engine_dependency = next(
        edit for edit in plan.edits if edit.kind == "engine-dependency"
    )
    # floor moves to the new exact version, ceiling major is unchanged
    assert engine_dependency.proposed == "epistemic-graph[full]>=2.24.0,<3.0.0"


def test_missing_ceiling_detected_and_repaired_under_patch_bump(
    tmp_path: Path,
) -> None:
    root = _write_root(tmp_path)
    providers_root = _write_providers(tmp_path, _ALL_PROVIDERS)

    plan = vc.plan_cascade(root=root, providers_root=providers_root, au_bump="patch")

    assert plan.severity == "patch"
    provider_edits = _provider_edits(plan)
    # only the drifted provider is touched; the well-formed ones are untouched
    assert {edit.path.parent.name for edit in provider_edits} == {
        "missing-ceiling-provider"
    }
    assert {edit.kind for edit in provider_edits} == {"missing-ceiling"}
    proposed = {edit.proposed for edit in provider_edits}
    assert proposed == {
        "agent-utilities>=2.0.0,<3.0.0",
        "agent-utilities[mcp]>=2.0.0,<3.0.0",
    }


def test_apply_plan_writes_planned_bytes_and_rejects_stale_edits(
    tmp_path: Path,
) -> None:
    root = _write_root(tmp_path)
    providers_root = _write_providers(tmp_path, _ALL_PROVIDERS)

    plan = vc.plan_cascade(root=root, providers_root=providers_root, au_version="3.0.0")
    written = vc.apply_plan(plan, root=root, providers_root=providers_root)

    assert set(written) == {edit.path for edit in plan.edits}
    for edit in plan.edits:
        text = edit.path.read_text(encoding="utf-8")
        assert edit.proposed in text
        assert edit.current not in text

    version_text = (root / "agent_utilities" / "_version.py").read_text(
        encoding="utf-8"
    )
    assert '__version__ = "3.0.0"' in version_text

    # re-applying the same plan against the now-mutated files must fail closed
    with pytest.raises(vc.CascadeError):
        vc.apply_plan(plan, root=root, providers_root=providers_root)


def test_apply_plan_writes_nothing_on_a_dry_run(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    providers_root = _write_providers(tmp_path, _ALL_PROVIDERS)
    before = (root / "agent_utilities" / "_version.py").read_text(encoding="utf-8")

    vc.plan_cascade(root=root, providers_root=providers_root, au_version="3.0.0")

    after = (root / "agent_utilities" / "_version.py").read_text(encoding="utf-8")
    assert before == after


def test_render_plan_is_deterministic(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    providers_root = _write_providers(tmp_path, _ALL_PROVIDERS)

    plan = vc.plan_cascade(root=root, providers_root=providers_root, au_version="3.0.0")

    first = vc.render_plan(plan)
    second = vc.render_plan(plan)
    assert first == second
    assert "severity=major" in first
    assert first.startswith("version cascade: severity=major\n")

    # rebuilding the identical plan from scratch renders identically too
    replan = vc.plan_cascade(
        root=root, providers_root=providers_root, au_version="3.0.0"
    )
    assert vc.render_plan(replan) == first


def test_render_plan_reports_zero_edits_explicitly(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    providers_root = _write_providers(tmp_path, _CLEAN_PROVIDERS)

    plan = vc.plan_cascade(root=root, providers_root=providers_root, au_bump="patch")

    rendered = vc.render_plan(plan)
    assert "tier 3 edits" not in rendered


def test_plan_cascade_is_pure_and_reusable(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    providers_root = _write_providers(tmp_path, _ALL_PROVIDERS)

    before_pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    before_matrix = (
        root / "deploy" / "release" / "compatibility-matrix.yml"
    ).read_text(encoding="utf-8")

    plan_one = vc.plan_cascade(
        root=root, providers_root=providers_root, au_version="3.0.0"
    )
    plan_two = vc.plan_cascade(
        root=root, providers_root=providers_root, au_version="3.0.0"
    )

    assert plan_one == plan_two
    assert (root / "pyproject.toml").read_text(encoding="utf-8") == before_pyproject
    assert (root / "deploy" / "release" / "compatibility-matrix.yml").read_text(
        encoding="utf-8"
    ) == before_matrix


def test_conflicting_explicit_and_bump_arguments_is_a_usage_error(
    tmp_path: Path,
) -> None:
    root = _write_root(tmp_path)
    providers_root = _write_providers(tmp_path, _CLEAN_PROVIDERS)

    with pytest.raises(vc.CascadeError):
        vc.plan_cascade(
            root=root,
            providers_root=providers_root,
            au_version="3.0.0",
            au_bump="major",
        )


def test_no_op_target_version_yields_an_empty_plan(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    providers_root = _write_providers(tmp_path, _CLEAN_PROVIDERS)

    plan = vc.plan_cascade(root=root, providers_root=providers_root, au_version="2.1.1")

    assert plan.is_empty
    assert plan.severity == "none"


def test_cli_dry_run_writes_nothing_and_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_root(tmp_path)
    providers_root = _write_providers(tmp_path, _ALL_PROVIDERS)
    before = (root / "agent_utilities" / "_version.py").read_text(encoding="utf-8")

    exit_code = vc.main(
        [
            "--agent-utilities-version",
            "3.0.0",
            "--root",
            str(root),
            "--providers-root",
            str(providers_root),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "DRY RUN" in captured.out
    after = (root / "agent_utilities" / "_version.py").read_text(encoding="utf-8")
    assert before == after


def test_cli_apply_writes_files_and_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_root(tmp_path)
    providers_root = _write_providers(tmp_path, _ALL_PROVIDERS)

    exit_code = vc.main(
        [
            "--agent-utilities-version",
            "3.0.0",
            "--root",
            str(root),
            "--providers-root",
            str(providers_root),
            "--apply",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "applied" in captured.out
    after = (root / "agent_utilities" / "_version.py").read_text(encoding="utf-8")
    assert '__version__ = "3.0.0"' in after


def test_cli_usage_error_with_no_targets(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    providers_root = _write_providers(tmp_path, _CLEAN_PROVIDERS)

    exit_code = vc.main(["--root", str(root), "--providers-root", str(providers_root)])

    assert exit_code == 2


def test_cli_empty_plan_exits_one(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    providers_root = _write_providers(tmp_path, _CLEAN_PROVIDERS)

    exit_code = vc.main(
        [
            "--agent-utilities-version",
            "2.1.1",
            "--root",
            str(root),
            "--providers-root",
            str(providers_root),
        ]
    )

    assert exit_code == 1
