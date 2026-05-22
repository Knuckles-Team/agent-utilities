"""Workflow Distillation Hook — Automated Execution→Template Promotion.

CONCEPT:ORCH-1.25 × CONCEPT:AHE-3.2 — Workflow Distillation Pipeline

Closes the evolution feedback loop by automatically distilling successful
workflow executions into reusable Workflow+TeamConfig pairs in the KG.

Configuration (from ``config.json``)::

    {
      "distillation": {
        "promotion_threshold": 3,
        "quality_score_minimum": 0.6
      }
    }
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.models.graph import GraphPlan

logger = logging.getLogger(__name__)

DEFAULT_PROMOTION_THRESHOLD = 3
DEFAULT_QUALITY_MINIMUM = 0.6


class WorkflowDistillationHook:
    """Automated promotion of execution patterns to reusable templates.

    CONCEPT:ORCH-1.25 — Distillation Hook

    Args:
        engine: The IntelligenceGraphEngine instance.
        promotion_threshold: Successes before auto-promotion (default: 3).
        quality_minimum: Minimum quality score (default: 0.6).
    """

    def __init__(
        self,
        engine: IntelligenceGraphEngine | None = None,
        promotion_threshold: int | None = None,
        quality_minimum: float | None = None,
    ) -> None:
        self.engine = engine
        resolved_threshold, resolved_quality = _load_distillation_config()
        self.promotion_threshold = (
            promotion_threshold
            if promotion_threshold is not None
            else resolved_threshold
        )
        self.quality_minimum = (
            quality_minimum if quality_minimum is not None else resolved_quality
        )

    async def on_execution_complete(
        self,
        run_id: str,
        plan: GraphPlan,
        result: str | dict[str, Any],
        team_config_id: str | None = None,
        quality_score: float = 0.0,
    ) -> dict[str, Any]:
        """Process a completed execution for potential distillation.

        CONCEPT:ORCH-1.25 — Distillation Entry Point
        """
        outcome: dict[str, Any] = {
            "promoted": False,
            "workflow_id": None,
            "team_config_id": None,
            "reason": "",
        }

        if not self.engine:
            outcome["reason"] = "no_engine"
            return outcome

        if quality_score < self.quality_minimum:
            outcome["reason"] = f"quality_below_minimum ({quality_score:.2f})"
            return outcome

        if not plan or len(plan.steps) < 2:
            outcome["reason"] = "trivial_workflow"
            return outcome

        pattern_key = _compute_pattern_key(plan)
        success_count = self._record_success(pattern_key)

        if success_count < self.promotion_threshold:
            outcome[
                "reason"
            ] = f"below_threshold ({success_count}/{self.promotion_threshold})"
            return outcome

        logger.info(
            "[ORCH-1.25] Pattern '%s' reached threshold — promoting!", pattern_key[:40]
        )
        outcome = await self._distill_and_promote(
            run_id=run_id,
            plan=plan,
            result=result,
            team_config_id=team_config_id,
            quality_score=quality_score,
            pattern_key=pattern_key,
        )
        return outcome

    def _record_success(self, pattern_key: str) -> int:
        """Increment success counter for a workflow pattern in the KG."""
        if (
            not self.engine
            or not hasattr(self.engine, "backend")
            or not self.engine.backend
        ):
            return 1
        import time

        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            self.engine.backend.execute(
                "MERGE (d:DistillationTracker {pattern_key: $key}) "
                "ON CREATE SET d.success_count = 1, d.first_seen = $ts, d.last_seen = $ts "
                "ON MATCH SET d.success_count = d.success_count + 1, d.last_seen = $ts",
                {"key": pattern_key, "ts": ts},
            )
            rows = self.engine.backend.execute(
                "MATCH (d:DistillationTracker {pattern_key: $key}) RETURN d.success_count AS count",
                {"key": pattern_key},
            )
            if rows:
                return int(rows[0].get("count", 1))
        except Exception as e:
            logger.debug("[ORCH-1.25] DistillationTracker update failed: %s", e)
        return 1

    async def _distill_and_promote(
        self,
        run_id: str,
        plan: GraphPlan,
        result: str | dict[str, Any],
        team_config_id: str | None,
        quality_score: float,
        pattern_key: str,
    ) -> dict[str, Any]:
        """Perform paired Workflow + TeamConfig promotion.

        CONCEPT:ORCH-1.25 × CONCEPT:AHE-3.3
        """
        outcome: dict[str, Any] = {
            "promoted": False,
            "workflow_id": None,
            "team_config_id": team_config_id,
            "reason": "",
        }
        try:
            skill_dir = await self._scaffold_skill(pattern_key, plan, team_config_id)
            if skill_dir:
                outcome["workflow_id"] = f"wf_{skill_dir.name}"

            # Optional: We could call SkillCompiler.register_in_kg here if we wanted immediate KG registration
            # but standard flow usually means it's available for next ingestion sweep
        except Exception as e:
            logger.warning("[ORCH-1.25] Workflow skill scaffolding failed: %s", e)
            outcome["reason"] = f"scaffold_failed: {e}"
            return outcome

        if team_config_id:
            try:
                from ..knowledge_graph.core.engine_registry import RegistryMixin

                if isinstance(self.engine, RegistryMixin):
                    self.engine.record_team_outcome(
                        team_config_id, reward=quality_score
                    )
            except Exception as e:
                logger.debug("[ORCH-1.25] TeamConfig reward update failed: %s", e)

        outcome["promoted"] = True
        outcome["reason"] = "threshold_met"
        logger.info(
            "[ORCH-1.25] Distillation complete: skill scaffolded at %s", skill_dir
        )
        return outcome

    async def _scaffold_skill(
        self, pattern_key: str, plan: GraphPlan, team_config_id: str | None
    ) -> Path:
        """Scaffold a new skill directory from a proven execution pattern."""
        import hashlib
        import time
        from pathlib import Path

        import yaml

        # Determine the universal-skills path
        # Assuming typical project structure: agent-packages/agent-utilities/... and agent-packages/skills/...
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        skills_dir = (
            base_dir
            / "skills"
            / "universal-skills"
            / "universal_skills"
            / "workflows"
            / "distilled"
        )
        skills_dir.mkdir(parents=True, exist_ok=True)

        pattern_hash = hashlib.md5(
            pattern_key.encode(), usedforsecurity=False
        ).hexdigest()[:8]
        skill_name = f"distilled-{pattern_hash}"
        skill_dir = skills_dir / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate SKILL.md
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        steps_md = []
        for i, step in enumerate(plan.steps):
            steps_md.append(f"### Step {i + 1}: {step.node_id}")
            steps_md.append(
                step.refined_subtask or step.input_data or "No task description."
            )
            steps_md.append("")

        skill_content = f"""---
name: {skill_name}
description: >-
  Auto-generated workflow skill from successful execution pattern.
  Promoted after reaching success threshold. Review and refine before distribution.
tags: [distilled, auto-generated, workflow]
metadata:
  author: distillation-hook
  version: '0.1.0'
  promoted_at: '{ts}'
---
# Distilled Workflow

> [!NOTE]
> This skill was auto-generated by the distillation pipeline (ORCH-1.25).
> Review and refine the steps below before distributing.

## Workflow Execution Steps

{"".join(steps_md)}
"""
        with open(skill_dir / "SKILL.md", "w", encoding="utf-8") as f:
            f.write(skill_content)

        # 2. Extract and write TeamConfig if available
        if (
            team_config_id
            and self.engine
            and hasattr(self.engine, "backend")
            and self.engine.backend
        ):
            refs_dir = skill_dir / "references"
            refs_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Attempt to retrieve team config from KG
                rows = self.engine.backend.execute(
                    "MATCH (t:TeamConfigNode {id: $id}) RETURN t",
                    {"id": team_config_id},
                )
                if rows and "t" in rows[0]:
                    t_props = rows[0]["t"]
                    team_data = {
                        "name": t_props.get("name", skill_name + "_team"),
                        "task_pattern": t_props.get(
                            "task_pattern", "distilled workflow"
                        ),
                        "execution_mode": t_props.get("execution_mode", "sequential"),
                    }
                    if "specialist_ids" in t_props:
                        import json

                        try:
                            team_data["specialist_ids"] = json.loads(
                                t_props["specialist_ids"]
                            )
                        except Exception:
                            team_data["specialist_ids"] = []

                    with open(refs_dir / "team.yaml", "w", encoding="utf-8") as f:
                        yaml.dump(team_data, f, sort_keys=False)
            except Exception as e:
                logger.debug("Failed to extract TeamConfig for scaffolded skill: %s", e)

        return skill_dir


def _compute_pattern_key(plan: GraphPlan) -> str:
    """Compute a canonical key from a GraphPlan for deduplication."""
    parts = []
    for step in plan.steps:
        dep_str = ",".join(sorted(step.depends_on)) if step.depends_on else "_"
        parts.append(f"{step.node_id}:{dep_str}")
    return "|".join(parts)


def _load_distillation_config() -> tuple[int, float]:
    """Load distillation thresholds from config.json."""
    try:
        from agent_utilities.config import AgentConfig

        config = AgentConfig.load()
        distillation = config.raw.get("distillation", {})
        return (
            int(distillation.get("promotion_threshold", DEFAULT_PROMOTION_THRESHOLD)),
            float(distillation.get("quality_score_minimum", DEFAULT_QUALITY_MINIMUM)),
        )
    except Exception:
        return DEFAULT_PROMOTION_THRESHOLD, DEFAULT_QUALITY_MINIMUM
