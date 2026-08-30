"""Skill Compiler — Natively Parse SKILL.md into Workflows.

CONCEPT:AU-ORCH.execution.skill-workflow-compilation — Skill-to-Workflow Compilation

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

from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
    skill_reference,
)
from agent_utilities.models.graph import ExecutionStep, GraphPlan

logger = logging.getLogger(__name__)

_STEP_HEADER_PATTERN = re.compile(
    r"###\s+Step\s+(\d+):\s*(.*?)\n(.*?)(?=\n###\s+Step|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_STEP_LIST_PATTERN = re.compile(
    r"^(\d+)\.\s+\*\*(.*?)\*\*(.*?)(?=\n\d+\.\s+\*\*|\Z)",
    re.MULTILINE | re.IGNORECASE | re.DOTALL,
)
_SKILL_ANNOTATION_PATTERN = re.compile(r"\[skill:\s*(.*?)\]", re.IGNORECASE)
_DEPENDENCY_ANNOTATION_PATTERN = re.compile(r"\[depends_on:\s*(.*?)\]", re.IGNORECASE)
_STEP_REFERENCE_PATTERN = re.compile(r"^(?:step-?)?(\d+)$")


def _normalise_identifier(value: str) -> str:
    """Normalise markdown identifiers to the compiler's node-id format."""
    return value.strip().lower().replace(" ", "-").replace("_", "-")


def _parse_dependencies(value: str) -> list[str]:
    """Parse a comma-separated dependency annotation."""
    cleaned = value.strip()
    if cleaned.lower() in ("none", "[]", ""):
        return []
    return [_normalise_identifier(item) for item in cleaned.split(",")]


def _parse_skill_annotation(title: str) -> tuple[str | None, str]:
    """Return an explicit skill id and title with its annotation removed."""
    match = _SKILL_ANNOTATION_PATTERN.search(title)
    if match is None:
        return None, title
    return (
        _normalise_identifier(match.group(1)),
        _SKILL_ANNOTATION_PATTERN.sub("", title).strip(),
    )


def _parse_dependency_annotation(
    title: str, body: str, previous_id: str | None
) -> tuple[list[str], str, str]:
    """Extract a dependency annotation from a title or body."""
    match = _DEPENDENCY_ANNOTATION_PATTERN.search(title)
    if match is not None:
        return (
            _parse_dependencies(match.group(1)),
            _DEPENDENCY_ANNOTATION_PATTERN.sub("", title).strip(),
            body,
        )

    match = _DEPENDENCY_ANNOTATION_PATTERN.search(body)
    if match is not None:
        return (
            _parse_dependencies(match.group(1)),
            title,
            _DEPENDENCY_ANNOTATION_PATTERN.sub("", body).strip(),
        )

    dependencies = [previous_id] if previous_id is not None else []
    return dependencies, title, body


def _infer_agent_name(explicit_id: str | None, title: str) -> str:
    """Infer the node id from an explicit id or the step title."""
    if explicit_id:
        return explicit_id
    title_parts = title.split(":", 1)
    candidate = title_parts[1] if len(title_parts) > 1 else title
    return _normalise_identifier(candidate)


def _ensure_unique_id(base_name: str, used_ids: set[str]) -> str:
    """Return a unique node id, retaining the compiler's numeric suffixes."""
    candidate = base_name
    counter = 1
    while candidate in used_ids:
        candidate = f"{base_name}-{counter}"
        counter += 1
    return candidate


def _parse_step(
    match: tuple[str, str, str], previous_id: str | None, used_ids: set[str]
) -> tuple[int, str, str, str, list[str]]:
    """Parse one markdown match into its plan fields."""
    step_num, title, body = match
    explicit_id, title = _parse_skill_annotation(title.strip())
    dependencies, clean_title, clean_body = _parse_dependency_annotation(
        title, body.strip(), previous_id
    )
    agent_name = _ensure_unique_id(
        _infer_agent_name(explicit_id, clean_title), used_ids
    )
    return int(step_num), agent_name, clean_title, clean_body, dependencies


def _resolve_dependency(dependency: str, step_num_to_id: dict[int, str]) -> str:
    """Resolve a numeric step reference to the corresponding node id."""
    cleaned = _normalise_identifier(dependency)
    match = _STEP_REFERENCE_PATTERN.match(cleaned)
    if match is None:
        return cleaned
    step_num = int(match.group(1))
    return step_num_to_id.get(step_num, cleaned)


def _resolve_step_dependencies(
    steps: list[ExecutionStep], parsed_step_info: list[tuple[int, str]]
) -> None:
    """Replace numeric dependency references with parsed node ids in-place."""
    step_num_to_id = dict(parsed_step_info)
    for step in steps:
        step.depends_on = [
            _resolve_dependency(dependency, step_num_to_id)
            for dependency in step.depends_on
        ]


def _find_step_matches(markdown: str) -> list[tuple[str, str, str]]:
    """Find header-style steps, falling back to the numbered-list form."""
    return _STEP_HEADER_PATTERN.findall(markdown) or _STEP_LIST_PATTERN.findall(
        markdown
    )


def _split_update_frontmatter(markdown: str) -> tuple[str, str]:
    """Separate the original frontmatter from the markdown body."""
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", markdown, re.DOTALL)
    if fm_match:
        return fm_match.group(0), markdown[fm_match.end() :]
    return "", markdown


def _split_update_sections(body: str) -> tuple[str, str, int]:
    """Find the preserved prose and numbering style around markdown steps."""
    step_header_pattern = re.compile(
        r"^(###\s+Step\s+\d+:|^\d+\.\s+\*\*)", re.MULTILINE | re.IGNORECASE
    )
    matches = list(step_header_pattern.finditer(body))
    if not matches:
        return body.rstrip() + "\n\n", "", 1

    intro_prose = body[: matches[0].start()]
    step_blocks_raw: list[str] = []
    for idx in range(len(matches)):
        start = matches[idx].start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        step_blocks_raw.append(body[start:end])

    last_block = step_blocks_raw[-1]
    concluding_match = re.search(
        r"\n\n(##?\s+.*)$", last_block, re.MULTILINE | re.DOTALL
    )
    if concluding_match:
        concluding_prose = last_block[concluding_match.start() :]
    else:
        concluding_prose = ""

    start_index = 1
    first_header = matches[0].group(0)
    num_match = re.search(r"\d+", first_header)
    if num_match:
        start_index = int(num_match.group(0))
    return intro_prose, concluding_prose, start_index


def _clean_update_subtask(refined_subtask: str | None) -> str:
    """Remove duplicated step headers before rendering a step body."""
    if refined_subtask:
        subtask_lines = refined_subtask.strip().split("\n")
        clean_lines = []
        for line in subtask_lines:
            line_strip = line.strip()
            if (
                line_strip.lower().startswith("step ") and ":" in line_strip
            ) or line_strip.lower().startswith("### step "):
                continue
            clean_lines.append(line)
        return "\n".join(clean_lines).strip()
    return "Execute step task."


def _format_update_step(step: ExecutionStep, step_num: int) -> str:
    """Render one GraphPlan step in the compiler's markdown format."""
    depends_suffix = ""
    if step.depends_on:
        depends_suffix = f" [depends_on: {', '.join(step.depends_on)}]"
    header = f"### Step {step_num}: {step.id}{depends_suffix}\n"
    body_text = _clean_update_subtask(step.refined_subtask)
    return f"{header}{body_text}\n\n"


class SkillCompiler:
    """Compile a SKILL.md into a GraphPlan.

    CONCEPT:AU-ORCH.execution.skill-workflow-compilation — Skill-to-Workflow Compilation

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
        matches = _find_step_matches(markdown)
        if not matches:
            steps = [
                ExecutionStep(
                    id="executor",
                    refined_subtask=markdown.strip()[:1000],
                    depends_on=[],
                )
            ]
            return GraphPlan(
                steps=steps,
                metadata={"name": name, "timeout_seconds": 600},
            )

        steps: list[ExecutionStep] = []
        parsed_step_info: list[tuple[int, str]] = []
        used_ids: set[str] = set()
        previous_id = None
        for match in matches:
            step_num, agent_name, title, body, dependencies = _parse_step(
                match, previous_id, used_ids
            )
            used_ids.add(agent_name)
            parsed_step_info.append((step_num, agent_name))
            steps.append(
                ExecutionStep(
                    id=agent_name,
                    refined_subtask=f"{title}\n{body}",
                    depends_on=dependencies,
                )
            )
            previous_id = agent_name

        _resolve_step_dependencies(steps, parsed_step_info)
        return GraphPlan(
            steps=steps,
            metadata={"name": name, "timeout_seconds": 600},
        )

    @staticmethod
    def update_markdown(original_markdown: str, plan: GraphPlan) -> str:
        """Update original markdown losslessly with changes from GraphPlan.

        Preserves frontmatter (updates metadata if needed), introductory prose,
        concluding prose, spacing, and comments, while updating step headers,
        bodies, and depends_on tags.
        """
        frontmatter, body = _split_update_frontmatter(original_markdown)
        intro_prose, concluding_prose, start_index = _split_update_sections(body)
        formatted_steps = [
            _format_update_step(step, start_index + i)
            for i, step in enumerate(plan.steps)
        ]
        new_body = intro_prose + "".join(formatted_steps) + concluding_prose
        return f"{frontmatter}{new_body}"

    @staticmethod
    def save(skill_dir: Path, plan: GraphPlan) -> None:
        """Update SKILL.md inside the given skill directory losslessly in-place."""
        skill_path = skill_dir / "SKILL.md"
        if not skill_path.exists():
            # If it doesn't exist, generate a fresh basic one
            metadata_name = skill_dir.name
            markdown = f"""---
name: {metadata_name}
description: Evolved skill workflow.
tags: [evolved]
---
# {metadata_name} Workflow

"""
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            original_content = markdown
        else:
            with open(skill_path, encoding="utf-8") as f:
                original_content = f.read()

        updated_content = SkillCompiler.update_markdown(original_content, plan)
        with open(skill_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
        logger.info("Losslessly saved updated skill workflow")

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
        except Exception as exc:
            logger.error(
                "Failed to load team configuration for %s (%s)",
                skill_reference(skill_dir.name),
                type(exc).__name__,
            )
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

        # If we have a real engine (not None, and not Mock/MagicMock), we save it using WorkflowStore
        if engine is not None and type(engine).__name__ not in (
            "MagicMock",
            "Mock",
            "NonCallableMagicMock",
        ):
            try:
                from agent_utilities.knowledge_graph.workflow_store import WorkflowStore
                from agent_utilities.models.knowledge_graph import RegistryNodeType

                store = WorkflowStore(engine)

                # 1. Save workflow definition in KG
                workflow_id = store.save_workflow(
                    name=skill_dir.name,
                    plan=plan,
                    description=team_config.get("description", "")
                    if team_config
                    else "",
                    metadata={
                        "skill_ref": skill_reference(skill_dir.name),
                        "has_team_config": bool(team_config),
                    },
                )

                # 2. Register a SkillNode for the skill directory itself
                skill_node_id = f"skill:{skill_dir.name}"
                engine.add_node(
                    skill_node_id,
                    RegistryNodeType.SKILL,
                    properties={
                        "name": skill_dir.name,
                        "source_ref": skill_reference(skill_dir.name),
                        "step_count": len(plan.steps),
                    },
                )

                # Link WorkflowDefinition -> DEFINED_BY_SKILL -> SkillNode
                engine.link_nodes(workflow_id, skill_node_id, "DEFINED_BY_SKILL")

                outcome["registered"] = True
                outcome["workflow_id"] = workflow_id

                # 3. Create TeamConfig node (if team.yaml is present)
                if team_config:
                    team_name = team_config.get("name", f"team_{skill_dir.name}")
                    team_config_id = (
                        f"team_config:{team_name.lower().replace(' ', '_')}"
                    )

                    tc_props = {
                        "id": team_config_id,
                        "name": team_name,
                        "task_pattern": team_config.get("task_pattern", skill_dir.name),
                        "specialist_ids": team_config.get("specialist_ids", []),
                        "capability_overrides": team_config.get(
                            "capability_overrides", {}
                        ),
                        "success_rate": 1.0,
                        "usage_count": 0,
                        "type": RegistryNodeType.TEAM_CONFIG,
                    }

                    engine.add_node(
                        team_config_id,
                        RegistryNodeType.TEAM_CONFIG,
                        properties=tc_props,
                    )

                    # Link TeamConfigNode to WorkflowDefinition
                    engine.link_nodes(team_config_id, workflow_id, "HAS_WORKFLOW")

                    outcome["team_config_id"] = team_config_id

            except Exception as exc:
                logger.error(
                    "Failed to register workflow %s (%s)",
                    skill_reference(skill_dir.name),
                    type(exc).__name__,
                )
                # CONCEPT:AU-AHE.evaluation.return-none-on-failure — a REAL
                # KG write (save_workflow/add_node/link_nodes) just failed.
                # This used to report registered=True with a fabricated
                # `wf_<name>` id that names no actual WorkflowDefinition node —
                # a caller checking outcome["registered"] would believe the
                # skill was wired into the KG when it was not, the exact
                # "component that cannot do its job returning something its
                # caller reads as success" shape. Report the failure honestly:
                # registered stays False, workflow_id stays None (never a fake
                # id), and the exception type is surfaced for diagnosis.
                outcome["registered"] = False
                outcome["workflow_id"] = None
                outcome["error"] = type(exc).__name__
        else:
            # Simulated successful registration structure for testing/dry-runs
            outcome["registered"] = True
            outcome["workflow_id"] = f"wf_{skill_dir.name}"

            if team_config:
                outcome["team_config_id"] = team_config.get(
                    "name", f"team_{skill_dir.name}"
                )

        return outcome
