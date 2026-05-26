"""Integration tests for parallel workflows compilation and topological scheduling.

CONCEPT:ORCH-1.25 — Parallel Engine
"""

from __future__ import annotations

import os
from pathlib import Path

from agent_utilities.graph.parallel_engine import ParallelEngine
from agent_utilities.models.execution_manifest import (
    AgentSpec,
    ExecutionManifest,
    SynthesisSpec,
)
from agent_utilities.workflows.skill_compiler import SkillCompiler

# Environment settings for clean integration tests
os.environ["OTEL_SDK_DISABLED"] = "true"


def get_skills_root() -> Path:
    """Find the universal-skills folder from the workspace."""
    # The path structure is agent-packages/skills/universal-skills/universal_skills/workflows/
    root = Path(
        "/home/apps/workspace/agent-packages/skills/universal-skills/universal_skills/workflows"
    )
    if not root.exists():
        raise FileNotFoundError(f"Skills directory not found at {root}")
    return root


def test_deploy_observability_stack_compilation():
    """Verify that deploy_observability_stack workflow compiles into correct parallel DAG."""
    skills_root = get_skills_root()
    skill_dir = skills_root / "infra" / "deploy_observability_stack"

    plan = SkillCompiler.compile(skill_dir)
    assert plan is not None
    assert len(plan.steps) == 4

    # Assert node IDs
    node_ids = [step.node_id for step in plan.steps]
    assert "prometheus-setup" in node_ids
    assert "grafana-setup" in node_ids
    assert "loki-setup" in node_ids
    assert "observability-synth" in node_ids

    # Assert step dependencies
    step_dict = {step.node_id: step for step in plan.steps}
    assert step_dict["prometheus-setup"].depends_on == []
    assert step_dict["grafana-setup"].depends_on == []
    assert step_dict["loki-setup"].depends_on == []
    assert set(step_dict["observability-synth"].depends_on) == {
        "prometheus-setup",
        "grafana-setup",
        "loki-setup",
    }

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
    skill_dir = skills_root / "finance" / "alpha_factor_mining"

    plan = SkillCompiler.compile(skill_dir)
    assert plan is not None
    assert len(plan.steps) == 4

    # Assert node IDs
    node_ids = [step.node_id for step in plan.steps]
    assert "technical-alpha" in node_ids
    assert "fundamental-alpha" in node_ids
    assert "sentiment-alpha" in node_ids
    assert "factor-fusion" in node_ids

    # Assert step dependencies
    step_dict = {step.node_id: step for step in plan.steps}
    assert step_dict["technical-alpha"].depends_on == []
    assert step_dict["fundamental-alpha"].depends_on == []
    assert step_dict["sentiment-alpha"].depends_on == []
    assert set(step_dict["factor-fusion"].depends_on) == {
        "technical-alpha",
        "fundamental-alpha",
        "sentiment-alpha",
    }

    # Verify team.yaml loading
    team = SkillCompiler.load_team_config(skill_dir)
    assert team is not None
    assert team["name"] == "Quant Research Swarm"
    assert "technical-alpha" in team["specialist_ids"]
    assert "factor-fusion" in team["specialist_ids"]
    assert "checkpointing" in team["capability_overrides"]["technical-alpha"]


def test_sdd_full_lifecycle_compilation():
    """Verify that sdd_full_lifecycle workflow compiles into correct multi-wave DAG."""
    skills_root = get_skills_root()
    skill_dir = skills_root / "dev-workflows" / "sdd_full_lifecycle"

    plan = SkillCompiler.compile(skill_dir)
    assert plan is not None
    assert len(plan.steps) == 5

    # Assert node IDs
    node_ids = [step.node_id for step in plan.steps]
    assert "spec-generator" in node_ids
    assert "python-backend-engineer" in node_ids
    assert "typescript-frontend-developer" in node_ids
    assert "qa-test-engineer" in node_ids
    assert "verification-gate" in node_ids

    # Assert step dependencies
    step_dict = {step.node_id: step for step in plan.steps}
    assert step_dict["spec-generator"].depends_on == []
    assert step_dict["python-backend-engineer"].depends_on == ["spec-generator"]
    assert step_dict["typescript-frontend-developer"].depends_on == ["spec-generator"]
    assert step_dict["qa-test-engineer"].depends_on == ["spec-generator"]
    assert set(step_dict["verification-gate"].depends_on) == {
        "python-backend-engineer",
        "typescript-frontend-developer",
        "qa-test-engineer",
    }

    # Verify team.yaml loading
    team = SkillCompiler.load_team_config(skill_dir)
    assert team is not None
    assert team["name"] == "Software Engineering Swarm"
    assert "spec-generator" in team["specialist_ids"]
    assert "verification-gate" in team["specialist_ids"]


def test_parallel_engine_wave_scheduling_for_workflows():
    """Verify that ParallelEngine wave scheduling matches the compiled step DAGs topological layers."""
    skills_root = get_skills_root()
    engine = ParallelEngine()

    # 1. Test deploy_observability_stack scheduling
    observability_plan = SkillCompiler.compile(
        skills_root / "infra" / "deploy_observability_stack"
    )
    assert observability_plan is not None

    agents = [
        AgentSpec(
            agent_id=step.node_id,
            role=step.node_id,
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
    assert len(waves) == 2
    # Wave 0 has 3 agents
    assert set(a.agent_id for a in waves[0]) == {
        "prometheus-setup",
        "grafana-setup",
        "loki-setup",
    }
    # Wave 1 has 1 agent
    assert [a.agent_id for a in waves[1]] == ["observability-synth"]

    # 2. Test sdd_full_lifecycle scheduling
    sdd_plan = SkillCompiler.compile(
        skills_root / "dev-workflows" / "sdd_full_lifecycle"
    )
    assert sdd_plan is not None

    sdd_agents = [
        AgentSpec(
            agent_id=step.node_id,
            role=step.node_id,
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

    sdd_waves = engine._schedule_waves(sdd_manifest)
    assert len(sdd_waves) == 3
    # Wave 0
    assert [a.agent_id for a in sdd_waves[0]] == ["spec-generator"]
    # Wave 1
    assert set(a.agent_id for a in sdd_waves[1]) == {
        "python-backend-engineer",
        "typescript-frontend-developer",
        "qa-test-engineer",
    }
    # Wave 2
    assert [a.agent_id for a in sdd_waves[2]] == ["verification-gate"]


def test_all_library_workflows_compilation():
    """Verify that all 240 workflows in the library compile and schedule correctly."""
    skills_root = get_skills_root()
    engine = ParallelEngine()

    folders = [
        "infra",
        "health",
        "system",
        "finance",
        "dev-workflows",
        "research",
        "social",
        "ops",
    ]

    workflow_paths = []
    for f in folders:
        dir_path = skills_root / f
        if not dir_path.exists():
            continue
        for p in dir_path.iterdir():
            if p.is_dir() and (p / "SKILL.md").exists():
                workflow_paths.append(p)

    # We expect exactly 240 workflows (or very close depending on initial setup)
    assert len(workflow_paths) >= 235, (
        f"Expected around 240 workflows, found {len(workflow_paths)}"
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
                agent_id=step.node_id,
                role=step.node_id,
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
