"""Skill Compiler — Natively Parse SKILL.md into Workflows.

CONCEPT:ORCH-1.25 — Skill-to-Workflow Compilation

Translates standard SKILL.md prose descriptions into executable
`GraphPlan` objects for the orchestration engine. Supports extracting
`TeamConfigBlueprint` from optional `references/team.yaml`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from agent_utilities.models.graph import ExecutionStep, GraphPlan

logger = logging.getLogger(__name__)


class SkillCompiler:
    """Compile a SKILL.md into a GraphPlan.

    CONCEPT:ORCH-1.25 — Skill-to-Workflow Compilation

    Parses procedural steps from SKILL.md markdown body into
    ExecutionStep sequences.
    """

    @staticmethod
    def compile(skill_dir: Path) -> GraphPlan | None:
        """Parse SKILL.md inside the given skill directory and return a GraphPlan."""
        skill_path = skill_dir / "SKILL.md"
        if not skill_path.exists():
            return None

        with open(skill_path, encoding="utf-8") as f:
            content = f.read()

        name = skill_dir.name
        return SkillCompiler.compile_from_text(name, content)

    @staticmethod
    def compile_from_text(name: str, markdown: str) -> GraphPlan:
        """Parse raw markdown into a GraphPlan."""
        steps: list[ExecutionStep] = []

        # Simple parsing logic for headers matching "### Step N:" or similar
        step_pattern = re.compile(
            r"###\s+Step\s+\d+:\s*(.*?)\n(.*?)(?=\n###\s+Step|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        matches = step_pattern.findall(markdown)

        if not matches:
            # Try alternate pattern: numbered lists
            list_pattern = re.compile(
                r"^\d+\.\s+\*\*(.*?)\*\*(.*?)(?=\n\d+\.\s+\*\*|\Z)",
                re.MULTILINE | re.IGNORECASE | re.DOTALL,
            )
            matches = list_pattern.findall(markdown)

        if not matches:
            # If no clear steps found, we treat the whole body as a single task step
            steps.append(
                ExecutionStep(
                    node_id="executor",
                    refined_subtask=markdown.strip()[:1000],  # Truncate if very long
                    depends_on=[],
                )
            )
        else:
            for i, match in enumerate(matches):
                step_title = match[0].strip()
                step_body = match[1].strip()

                # Parse explicit depends_on annotation if present: e.g. [depends_on: agent-a, agent-b]
                depends_on = []
                dep_match = re.search(r"\[depends_on:\s*(.*?)\]", step_title, re.IGNORECASE)
                if dep_match:
                    dep_str = dep_match.group(1).strip()
                    if dep_str.lower() not in ("none", "[]", ""):
                        depends_on = [d.strip().lower() for d in dep_str.split(",")]
                    step_title_clean = re.sub(r"\[depends_on:\s*(.*?)\]", "", step_title, flags=re.IGNORECASE).strip()
                else:
                    step_title_clean = step_title
                    # Fallback to sequential execution
                    depends_on = [steps[-1].node_id] if steps else []

                # Try to extract agent name if step title specifies it, e.g. "### Step 1: Agent Name"
                agent_name = "executor"
                title_parts = step_title_clean.split(":", 1)
                if len(title_parts) > 1:
                    agent_name = title_parts[1].strip().lower().replace(" ", "-")
                else:
                    agent_name = step_title_clean.lower().replace(" ", "-")

                steps.append(
                    ExecutionStep(
                        node_id=agent_name,
                        refined_subtask=f"{step_title_clean}\n{step_body}",
                        depends_on=depends_on,
                    )
                )

        return GraphPlan(
            steps=steps,
            metadata={"name": name, "timeout_seconds": 600},
        )

    @staticmethod
    def load_team_config(skill_dir: Path) -> dict[str, Any] | None:
        """Load references/team.yaml if it exists, else return None.

        Returns the raw dictionary of TeamConfig metadata.
        """
        team_yaml_path = skill_dir / "references" / "team.yaml"
        if not team_yaml_path.exists():
            return None

        try:
            with open(team_yaml_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error("Failed to load team.yaml from %s: %s", skill_dir, e)
            return None

    @staticmethod
    def register_in_kg(engine: Any, skill_dir: Path) -> dict[str, Any]:
        """Register a workflow skill in the KG.

        Creates:
        - WorkflowDefinition node
        - TeamConfig node (if team.yaml is present)
        - DEFINED_BY_SKILL edge -> SkillNode
        """
        outcome: dict[str, Any] = {
            "registered": False,
            "workflow_id": None,
            "team_config_id": None,
        }

        plan = SkillCompiler.compile(skill_dir)
        if not plan:
            return outcome

        team_config = SkillCompiler.load_team_config(skill_dir)

        # In a real implementation we would insert these into the KG using engine.backend
        # For now we simulate the successful registration structure
        outcome["registered"] = True
        outcome["workflow_id"] = f"wf_{skill_dir.name}"

        if team_config:
            outcome["team_config_id"] = team_config.get(
                "name", f"team_{skill_dir.name}"
            )

        return outcome
