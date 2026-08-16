"""Integration tests for parallel workflows compilation and topological scheduling.

CONCEPT:AU-ORCH.execution.parallel-engine-visualizer — Parallel Engine
"""

from __future__ import annotations

import os
from pathlib import Path

from agent_utilities.models.execution_manifest import (
    AgentSpec,
    ExecutionManifest,
    SynthesisSpec,
)
from agent_utilities.orchestration import ParallelEngine
from agent_utilities.workflows.skill_compiler import SkillCompiler

# Environment settings for clean integration tests
os.environ["OTEL_SDK_DISABLED"] = "true"


def get_skills_root() -> Path:
    """Find the universal-skills folder via the installed ``universal_skills`` package.

    A hardcoded ``/home/agent-user/workspace/...`` path only worked on one
    specific historical checkout layout; every worktree (this repo runs from
    git worktrees under ``/home/local/worktrees/...``) resolved it to a path
    that never existed. ``universal_skills`` is a uv workspace member — its
    installed (editable) package path is the portable way to find it,
    consistent with how production code locates it (see
    ``agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest.discover_workflow_skill_files``).
    """
    import universal_skills

    root = Path(next(iter(universal_skills.__path__)))
    if not root.exists():
        raise FileNotFoundError(f"Skills directory not found at {root}")
    return root


def _installed_universal_skills_version() -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version("universal-skills")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _require_installed_skill(skills_root: Path, category: str, skill_name: str) -> Path:
    """Resolve a category/skill dir, failing LOUDLY and BY NAME if it is absent.

    CONCEPT B2 (REMAINING-ISSUES-DESIGNS.md): the latest published
    ``universal-skills`` PyPI release (currently 1.2.1) does not yet ship
    every category this test suite expects — ``infrastructure-workflows``,
    ``finance-workflows``, and most of ``development-workflows`` are missing
    from the installed package. That is an upstream PUBLISHING gap, not a bug
    in this repo, and not a test bug either — the fix is a real
    ``universal-skills`` release containing those categories. Until that
    ships, this must keep failing (never silently skip/xfail/assert-away) so
    the gap stays visible, but the failure names EXACTLY what is missing from
    which installed version instead of surfacing as an opaque
    ``assert None is not None`` deep inside ``SkillCompiler``.
    """
    skill_dir = skills_root / category / skill_name
    if not skill_dir.is_dir():
        raise AssertionError(
            f"universal-skills=={_installed_universal_skills_version()} (installed) "
            f"does not publish {category}/{skill_name} — upstream publishing gap "
            "(CONCEPT B2), not a defect in agent-utilities. Regenerate/publish a "
            "universal-skills release containing this category, then this test "
            "will resolve the real path and exercise it."
        )
    return skill_dir


def test_deploy_observability_stack_compilation():
    """Verify that deploy_observability_stack workflow compiles into correct parallel DAG."""
    skills_root = get_skills_root()
    skill_dir = _require_installed_skill(
        skills_root, "infrastructure-workflows", "deploy-observability-stack"
    )

    plan = SkillCompiler.compile(skill_dir)
    assert plan is not None
    assert len(plan.steps) == 5

    # Assert node IDs
    node_ids = [step.id for step in plan.steps]
    assert "prometheus-setup" in node_ids
    assert "grafana-setup" in node_ids
    assert "loki-setup" in node_ids
    assert "observability-synth" in node_ids
    assert "kg-persistence" in node_ids

    # Assert step dependencies
    step_dict = {step.id: step for step in plan.steps}
    assert step_dict["prometheus-setup"].depends_on == []
    assert step_dict["grafana-setup"].depends_on == []
    assert step_dict["loki-setup"].depends_on == []
    assert set(step_dict["observability-synth"].depends_on) == {
        "prometheus-setup",
        "grafana-setup",
        "loki-setup",
    }
    assert step_dict["kg-persistence"].depends_on == ["observability-synth"]

    # Verify team.yaml loading
    team = SkillCompiler.load_team_config(skill_dir)
    assert team is not None
    assert team["name"] == "Observability Engineering Swarm"
    assert "prometheus-setup" in team["specialist_ids"]
    assert "grafana-setup" in team["specialist_ids"]
    assert "loki-setup" in team["specialist_ids"]
    assert "observability-synth" in team["specialist_ids"]
    assert "checkpointing" in team["capability_overrides"]["prometheus-setup"]
    assert (
        "adversarial_verification"
        in team["capability_overrides"]["observability-synth"]
    )


def test_alpha_factor_mining_compilation():
    """Verify that alpha_factor_mining workflow compiles into correct parallel DAG."""
    skills_root = get_skills_root()
    skill_dir = _require_installed_skill(
        skills_root, "finance-workflows", "alpha-factor-mining"
    )

    plan = SkillCompiler.compile(skill_dir)
    assert plan is not None
    assert len(plan.steps) == 5

    # Assert node IDs
    node_ids = [step.id for step in plan.steps]
    assert "technical-alpha" in node_ids
    assert "fundamental-alpha" in node_ids
    assert "sentiment-alpha" in node_ids
    assert "factor-fusion" in node_ids
    assert "kg-persistence" in node_ids

    # Assert step dependencies
    step_dict = {step.id: step for step in plan.steps}
    assert step_dict["technical-alpha"].depends_on == []
    assert step_dict["fundamental-alpha"].depends_on == []
    assert step_dict["sentiment-alpha"].depends_on == []
    assert set(step_dict["factor-fusion"].depends_on) == {
        "technical-alpha",
        "fundamental-alpha",
        "sentiment-alpha",
    }
    assert step_dict["kg-persistence"].depends_on == ["factor-fusion"]

    # Verify team.yaml loading
    team = SkillCompiler.load_team_config(skill_dir)
    assert team is not None
    assert team["name"] == "Quant Research Swarm"
    assert "technical-alpha" in team["specialist_ids"]
    assert "factor-fusion" in team["specialist_ids"]
    assert "checkpointing" in team["capability_overrides"]["technical-alpha"]


def test_sdd_full_lifecycle_compilation():
    """Verify that sdd-full-lifecycle workflow compiles into the correct sequential DAG.

    The skill was redesigned (now v1.2.1) since this test was written: it's a
    strictly sequential intake->spec->verify->plan->implement->test chain, not
    the old fan-out/fan-in shape (spec-generator -> 3 parallel builders ->
    verification-gate -> kg-persistence). Step ids and depends_on below match
    universal_skills/development-workflows/sdd-full-lifecycle/SKILL.md.
    """
    skills_root = get_skills_root()
    skill_dir = _require_installed_skill(
        skills_root, "development-workflows", "sdd-full-lifecycle"
    )

    plan = SkillCompiler.compile(skill_dir)
    assert plan is not None
    assert len(plan.steps) == 6

    # Assert node IDs
    node_ids = [step.id for step in plan.steps]
    assert "spec-intake-wizard" in node_ids
    assert "spec-generator" in node_ids
    assert "spec-verifier" in node_ids
    assert "task-planner" in node_ids
    assert "sdd-implementer" in node_ids
    assert "automated-test-runner" in node_ids

    # Assert step dependencies (strictly sequential)
    step_dict = {step.id: step for step in plan.steps}
    assert step_dict["spec-intake-wizard"].depends_on == []
    assert step_dict["spec-generator"].depends_on == ["spec-intake-wizard"]
    assert step_dict["spec-verifier"].depends_on == ["spec-generator"]
    assert step_dict["task-planner"].depends_on == ["spec-verifier"]
    assert step_dict["sdd-implementer"].depends_on == ["task-planner"]
    assert step_dict["automated-test-runner"].depends_on == ["sdd-implementer"]

    # Verify team.yaml loading
    team = SkillCompiler.load_team_config(skill_dir)
    assert team is not None
    assert team["name"] == "sdd-full-lifecycle-team"
    assert "spec-intake-wizard" in team["specialist_ids"]
    assert "automated-test-runner" in team["specialist_ids"]


def test_parallel_engine_wave_scheduling_for_workflows():
    """Verify that ParallelEngine wave scheduling matches the compiled step DAGs topological layers."""
    skills_root = get_skills_root()
    engine = ParallelEngine()

    # 1. Test deploy_observability_stack scheduling
    observability_plan = SkillCompiler.compile(
        _require_installed_skill(
            skills_root, "infrastructure-workflows", "deploy-observability-stack"
        )
    )
    assert observability_plan is not None

    agents = [
        AgentSpec(
            agent_id=step.id,
            role=step.id,
            task_template=step.refined_subtask or "",
            depends_on=step.depends_on,
        )
        for step in observability_plan.steps
    ]
    manifest = ExecutionManifest(
        name="Observability Test",
        agents=agents,
        execution_mode="parallel",
        query="Verify observability deployment layers",
        synthesis=SynthesisSpec(strategy="flat"),
    )

    waves = engine._schedule_waves(manifest)
    assert len(waves) == 3
    # Wave 0 has 3 agents
    assert set(a.agent_id for a in waves[0]) == {
        "prometheus-setup",
        "grafana-setup",
        "loki-setup",
    }
    # Wave 1 has 1 agent
    assert [a.agent_id for a in waves[1]] == ["observability-synth"]
    # Wave 2 has 1 agent
    assert [a.agent_id for a in waves[2]] == ["kg-persistence"]

    # 2. Test sdd_full_lifecycle scheduling
    sdd_plan = SkillCompiler.compile(
        _require_installed_skill(
            skills_root, "development-workflows", "sdd-full-lifecycle"
        )
    )
    assert sdd_plan is not None

    sdd_agents = [
        AgentSpec(
            agent_id=step.id,
            role=step.id,
            task_template=step.refined_subtask or "",
            depends_on=step.depends_on,
        )
        for step in sdd_plan.steps
    ]
    sdd_manifest = ExecutionManifest(
        name="SDD Test",
        agents=sdd_agents,
        execution_mode="parallel",
        query="Verify SDD lifecycle layers",
        synthesis=SynthesisSpec(strategy="flat"),
    )

    # The current sdd-full-lifecycle skill is a strictly sequential 6-step
    # chain (see test_sdd_full_lifecycle_compilation), so each wave has
    # exactly one agent.
    sdd_waves = engine._schedule_waves(sdd_manifest)
    assert len(sdd_waves) == 6
    assert [a.agent_id for a in sdd_waves[0]] == ["spec-intake-wizard"]
    assert [a.agent_id for a in sdd_waves[1]] == ["spec-generator"]
    assert [a.agent_id for a in sdd_waves[2]] == ["spec-verifier"]
    assert [a.agent_id for a in sdd_waves[3]] == ["task-planner"]
    assert [a.agent_id for a in sdd_waves[4]] == ["sdd-implementer"]
    assert [a.agent_id for a in sdd_waves[5]] == ["automated-test-runner"]


def test_all_library_workflows_compilation():
    """Verify that all multi-agent workflows in the library compile and schedule correctly.

    The library's category taxonomy has been reorganized since this test's
    original 8-folder allowlist (``infra``, ``health``, ``system``, ``finance``,
    ``dev-workflows``, ``research``, ``social``, ``ops``) was written: multi-agent
    workflows (a ``references/team.yaml`` alongside ``SKILL.md``) now live
    exclusively under ``<domain>-workflows/`` category directories, distinct
    from the single-skill ``<domain>/`` directories that share the same root.
    Discover those categories by suffix instead of hardcoding a stale list, so
    this test tracks the catalog's actual shape rather than one snapshot of it.
    """
    skills_root = get_skills_root()
    engine = ParallelEngine()

    workflow_paths = []
    for dir_path in sorted(skills_root.iterdir()):
        if not dir_path.is_dir() or not dir_path.name.endswith("-workflows"):
            continue
        for p in dir_path.iterdir():
            if p.is_dir() and (p / "SKILL.md").exists():
                workflow_paths.append(p)

    # Verified current catalog size: 166 team.yaml-backed multi-agent workflows
    # across the 10 "*-workflows" categories (see the D-RG2-1-era investigation
    # that replaced the stale 8-folder/~240 expectation above). Keep a small
    # buffer below that so a handful of workflows moving around doesn't flake
    # this test, while still catching a real catalog regression.
    assert len(workflow_paths) >= 160, (
        f"Expected at least 160 multi-agent workflows, found {len(workflow_paths)} in "
        f"universal-skills=={_installed_universal_skills_version()} (installed). If "
        "this is below 160 because whole *-workflows categories are missing "
        "(infrastructure-workflows, finance-workflows, most of "
        "development-workflows) that is the known CONCEPT B2 upstream "
        "publishing gap, not a defect here — see "
        "_require_installed_skill's docstring above."
    )

    for skill_dir in workflow_paths:
        # 1. Compile workflow
        plan = SkillCompiler.compile(skill_dir)
        assert plan is not None, f"Failed to compile {skill_dir.name}"
        assert len(plan.steps) >= 1, f"{skill_dir.name} has no steps"

        # 2. Verify team configuration
        team = SkillCompiler.load_team_config(skill_dir)
        assert team is not None, f"Failed to load team config for {skill_dir.name}"
        assert "specialist_ids" in team

        # 3. Schedule waves via ParallelEngine
        agents = [
            AgentSpec(
                agent_id=step.id,
                role=step.id,
                task_template=step.refined_subtask or "",
                depends_on=step.depends_on,
            )
            for step in plan.steps
        ]
        manifest = ExecutionManifest(
            name=f"Test Run for {skill_dir.name}",
            agents=agents,
            execution_mode="parallel",
            query=f"Run parallel execution for {skill_dir.name}",
            synthesis=SynthesisSpec(strategy="flat"),
        )

        waves = engine._schedule_waves(manifest)
        assert len(waves) >= 1, f"Failed to schedule waves for {skill_dir.name}"
