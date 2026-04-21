#!/usr/bin/python
"""SDD (Spec-Driven Development) Utility Module.

This module provides high-level utilities for managing structured SDD artifacts
(Specifications, Plans, Constitutions) and their relationship to tasks. It handles
disk persistence in the agent's 'agent_data' directory and provides logic for
dependency analysis.
"""

import re
from pathlib import Path
from typing import Any, TypeVar

from .models import (
    ImplementationPlan,
    ProjectConstitution,
    Spec,
    Task,
    Tasks,
    TaskStatus,
)

T = TypeVar("T", ProjectConstitution, Spec, ImplementationPlan, Tasks)


class SDDManager:
    """Manages structured SDD data within an agent's workspace."""

    def __init__(self, workspace_path: str | Path):
        self.workspace_root = Path(workspace_path)
        self.specify_dir = self.workspace_root / ".specify"

    def _get_path(self, model_type: type[T], feature_id: str | None = None) -> Path:
        """Resolve the standard path for an SDD model (Markdown-first)."""
        if model_type == ProjectConstitution:
            return self.specify_dir / "constitution.md"

        if feature_id is None:
            raise ValueError(f"feature_id is required for {model_type.__name__}")

        if model_type == Spec:
            return self.specify_dir / "specs" / feature_id / "spec.md"

        if model_type == ImplementationPlan:
            return self.specify_dir / "specs" / feature_id / "plan.md"

        if model_type == Tasks:
            return self.specify_dir / "specs" / feature_id / "tasks.md"

        raise ValueError(f"Unsupported SDD model type: {model_type}")

    def save(self, model: T, feature_id: str | None = None) -> Path:
        """Persist an SDD model to the .specify directory as Markdown."""
        path = self._get_path(type(model), feature_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(model, ProjectConstitution):
            content = self._render_constitution_md(model)
        elif isinstance(model, Spec):
            content = self._render_spec_md(model)
        elif isinstance(model, ImplementationPlan):
            content = self._render_plan_md(model)
        elif isinstance(model, Tasks):
            content = self._render_tasks_md(model)
        else:
            raise ValueError(f"Unsupported model for saving: {type(model)}")

        path.write_text(content, encoding="utf-8")

        # Sync to Knowledge Graph
        self.record_sdd_outcome(model, feature_id)

        return path

    def load(self, model_type: type[T], feature_id: str | None = None) -> T | None:
        """Load an SDD model from the .specify directory by parsing Markdown."""
        path = self._get_path(model_type, feature_id)
        if not path.exists():
            return None

        content = path.read_text(encoding="utf-8")

        from typing import cast

        if model_type == ProjectConstitution:
            return cast(T, self._parse_constitution_md(content))
        if model_type == Spec:
            return cast(T, self._parse_spec_md(content, feature_id))
        if model_type == ImplementationPlan:
            return cast(T, self._parse_plan_md(content, feature_id))
        if model_type == Tasks:
            return cast(T, self.import_from_markdown(path, feature_id or "default"))

        return None

    def record_sdd_outcome(self, model: T, feature_id: str | None = None):
        """Record the creation or update of an SDD artifact in the Knowledge Graph."""
        from .knowledge_graph.engine import IntelligenceGraphEngine

        engine = IntelligenceGraphEngine.get_active()
        if not engine or not engine.backend:
            return

        artifact_type = type(model).__name__
        name = feature_id if feature_id else "Global"

        query = (
            "MERGE (a:SDDArtifact {id: $id}) "
            "SET a.type = $type, a.name = $name, a.last_updated = timestamp() "
            "RETURN a.id"
        )
        props = {
            "id": f"sdd:{artifact_type}:{name}",
            "type": artifact_type,
            "name": name,
        }
        try:
            engine.backend.execute(query, props)
            # Link to workspace/project node if it exists
            engine.backend.execute(
                "MATCH (p:Project) WHERE p.name = 'current' "
                "MATCH (a:SDDArtifact {id: $id}) "
                "MERGE (p)-[:HAS_ARTIFACT]->(a)",
                {"id": props["id"]},
            )
        except Exception:
            pass

    def get_parallel_opportunities(self, task_list: Tasks) -> list[list[str]]:
        """Identify groups of tasks that can be safely run in parallel.

        Calculates parallel groups based on:
        1. Explicit dependency graph (tasks without dependencies or with completed deps).
        2. File collision detection (tasks affecting same files cannot be parallel).
        """
        completed = set()
        all_tasks = {}
        for task in task_list.tasks:
            all_tasks[task.id] = task
            if task.status == TaskStatus.COMPLETED:
                completed.add(task.id)

        pending = [t for t in all_tasks.values() if t.status != TaskStatus.COMPLETED]

        groups = []
        current_batch = []
        occupied_files = set()

        for task in pending:
            # Check if all dependencies are met
            deps_met = all(d in completed for d in task.depends_on)

            if deps_met:
                # Check for file collisions with the current batch
                has_collision = any(f in occupied_files for f in task.file_paths)

                if not has_collision:
                    current_batch.append(task.id)
                    occupied_files.update(task.file_paths)
                else:
                    # If collision, this task must wait for next batch
                    if current_batch:
                        groups.append(current_batch)
                        current_batch = [task.id]
                        occupied_files = set(task.file_paths)

        if current_batch:
            groups.append(current_batch)

        return groups

    def export_to_markdown(self, model: Spec | Tasks, feature_id: str) -> Path:
        """Export an SDD model to a human-readable Markdown file.

        This provides spec-kit parity by maintaining mirrored .md files in the workspace.
        """
        if isinstance(model, Spec):
            path = self.workspace_root / f"spec-{feature_id}.md"
            content = self._render_spec_md(model)
        elif isinstance(model, Tasks):
            path = self.workspace_root / f"tasks-{feature_id}.md"
            content = self._render_tasks_md(model)
        else:
            raise ValueError(f"Unsupported model for markdown export: {type(model)}")

        path.write_text(content, encoding="utf-8")
        return path

    def _render_spec_md(self, spec: Spec) -> str:
        md = [f"# Spec: {spec.title}\n"]
        md.append(f"**Feature ID**: {spec.feature_id}\n")
        md.append("## User Stories")
        for us in spec.user_stories:
            md.append(f"- **{us.title}**: {us.description}")
            for ac in us.acceptance_criteria:
                md.append(f"  - [ ] {ac}")

        if spec.non_functional_requirements:
            md.append("\n## Non-Functional Requirements")
            for req in spec.non_functional_requirements:
                md.append(f"- {req}")

        return "\n".join(md)

    def _render_constitution_md(self, constitution: ProjectConstitution) -> str:
        md = ["# Project Constitution\n"]
        md.append(f"**Vision**: {constitution.vision}")
        md.append(f"**Mission**: {constitution.mission}\n")
        md.append("## Core Principles")
        for principle in constitution.core_principles:
            md.append(f"- {principle}")
        md.append("\n## Tech Stack")
        for k, v in constitution.tech_stack.items():
            md.append(f"- **{k}**: {v}")
        return "\n".join(md)

    def _render_plan_md(self, plan: ImplementationPlan) -> str:
        md = ["# Implementation Plan\n"]
        md.append(f"## Approach\n{plan.approach}\n")
        md.append("## Risks")
        for risk in plan.risks:
            md.append(f"- {risk}")
        return "\n".join(md)

    def _parse_constitution_md(self, content: str) -> ProjectConstitution:
        vision = ""
        mission = ""
        principles = []
        tech_stack = {}

        vision_match = re.search(r"\*\*Vision\*\*:\s*(.*)", content)
        if vision_match:
            vision = vision_match.group(1).strip()

        mission_match = re.search(r"\*\*Mission\*\*:\s*(.*)", content)
        if mission_match:
            mission = mission_match.group(1).strip()

        # Simple bullet list parsing
        lines = content.splitlines()
        current_section = None
        for line in lines:
            if "## Core Principles" in line:
                current_section = "principles"
            elif "## Tech Stack" in line:
                current_section = "tech"
            elif line.startswith("- ") and current_section == "principles":
                principles.append(line[2:].strip())
            elif line.startswith("- **") and current_section == "tech":
                match = re.match(r"- \*\*(.*?)\*\*:\s*(.*)", line)
                if match:
                    tech_stack[match.group(1).strip()] = match.group(2).strip()

        return ProjectConstitution(
            vision=vision,
            mission=mission,
            core_principles=principles,
            tech_stack=tech_stack,
        )

    def _parse_spec_md(self, content: str, feature_id: str | None = None) -> Spec:
        title = ""
        title_match = re.search(r"# Spec:\s*(.*)", content)
        if title_match:
            title = title_match.group(1).strip()

        # Simplified parsing
        from .models import UserStory

        stories = []
        current_story = None
        for line in content.splitlines():
            if line.startswith("- **"):
                match = re.match(r"- \*\*(.*?)\*\*:\s*(.*)", line)
                if match:
                    if current_story:
                        stories.append(current_story)
                    current_story = UserStory(
                        id=match.group(1).strip(),
                        title=match.group(1).strip(),
                        description=match.group(2).strip(),
                        acceptance_criteria=[],
                    )
            elif line.startswith("  - [ ]") and current_story:
                current_story.acceptance_criteria.append(line[7:].strip())
        if current_story:
            stories.append(current_story)

        return Spec(
            feature_id=feature_id or "unknown", title=title, user_stories=stories
        )

    def _parse_plan_md(
        self, content: str, feature_id: str | None = None
    ) -> ImplementationPlan:
        title = "Implementation Plan"
        title_match = re.search(r"^# (.*)", content)
        if title_match:
            title = title_match.group(1).strip()

        approach = ""
        risks = []
        # Simplified parsing
        approach_match = re.search(r"## Approach\n(.*?)\n##", content, re.DOTALL)
        if approach_match:
            approach = approach_match.group(1).strip()

        for line in content.splitlines():
            if line.startswith("- ") and "Risks" in content:
                # This is a bit fragile, but works for basic cases
                risks.append(line[2:].strip())

        return ImplementationPlan(
            feature_id=feature_id or "unknown",
            title=title,
            approach=approach,
            risks=risks,
        )

    def _render_tasks_md(self, tasks: Tasks) -> str:
        md = [f"# Tasks: {tasks.feature_id}\n"]
        for task in tasks.tasks:
            status_marker = "[x]" if task.status == TaskStatus.COMPLETED else "[ ]"
            parallel_marker = " [P]" if task.parallel else ""
            md.append(f"### {status_marker} {task.id}: {task.title}{parallel_marker}")
            md.append(f"{task.description}")
            if task.depends_on:
                md.append(f"\n**Depends on**: {', '.join(task.depends_on)}")
            if task.file_paths:
                md.append(f"\n**Files**: {', '.join(task.file_paths)}")
            md.append("")

        return "\n".join(md)

    def import_from_markdown(self, markdown_path: str | Path, feature_id: str) -> Tasks:
        """Parse a tasks.md file back into a structured Tasks model.

        Supports spec-kit [P] markers for parallel execution detection.
        """
        path = Path(markdown_path)
        content = path.read_text(encoding="utf-8")

        tasks: list[Task] = []
        # Pattern for "### [ ] T1: Title [P]"
        task_pattern = re.compile(
            r"### \[(?P<status>[ xX])\] (?P<id>[A-Za-z0-9_-]+): (?P<title>.*?)(?P<parallel> \[P\])?$"
        )

        current_task: dict[str, Any] | None = None
        for line in content.splitlines():
            match = task_pattern.match(line)
            if match:
                if current_task:
                    tasks.append(Task(**current_task))

                status_char = match.group("status").lower()
                status = (
                    TaskStatus.COMPLETED if status_char == "x" else TaskStatus.PENDING
                )

                current_task = {
                    "id": match.group("id"),
                    "title": match.group("title").strip(),
                    "description": "",
                    "status": status,
                    "parallel": bool(match.group("parallel")),
                    "depends_on": [],
                    "file_paths": [],
                }
            elif current_task:
                if line.startswith("**Depends on**:"):
                    deps = line.replace("**Depends on**:", "").strip()
                    current_task["depends_on"] = [
                        d.strip() for d in deps.split(",") if d.strip()
                    ]
                elif line.startswith("**Files**:"):
                    files = line.replace("**Files**:", "").strip()
                    current_task["file_paths"] = [
                        f.strip() for f in files.split(",") if f.strip()
                    ]
                elif (
                    line.strip()
                    and not line.startswith("#")
                    and not line.startswith("##")
                ):
                    current_task["description"] += line.strip() + "\n"

        if current_task:
            tasks.append(Task(**current_task))

        return Tasks(feature_id=feature_id, tasks=tasks)
