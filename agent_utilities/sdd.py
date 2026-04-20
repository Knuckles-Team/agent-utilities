#!/usr/bin/python
# coding: utf-8
"""SDD (Spec-Driven Development) Utility Module.

This module provides high-level utilities for managing structured SDD artifacts
(Specifications, Plans, Constitutions) and their relationship to tasks. It handles
disk persistence in the agent's 'agent_data' directory and provides logic for
dependency analysis.
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Type, TypeVar, Union

from .models import (
    ProjectConstitution,
    Spec,
    ImplementationPlan,
    Tasks,
    TaskStatus,
)

T = TypeVar("T", ProjectConstitution, Spec, ImplementationPlan, Tasks)


class SDDManager:
    """Manages structured SDD data within an agent's workspace."""

    def __init__(self, workspace_path: Union[str, Path]):
        self.workspace_root = Path(workspace_path)
        self.agent_data = self.workspace_root / "agent_data"

    def _get_path(self, model_type: Type[T], feature_id: Optional[str] = None) -> Path:
        """Resolve the standard path for an SDD model."""
        if model_type == ProjectConstitution:
            return self.agent_data / "constitution.json"

        if feature_id is None:
            raise ValueError(f"feature_id is required for {model_type.__name__}")

        if model_type == Spec:
            return self.agent_data / "specs" / f"{feature_id}.json"

        if model_type == ImplementationPlan:
            return self.agent_data / "plans" / f"{feature_id}.json"

        if model_type == Tasks:
            return self.agent_data / "tasks" / f"{feature_id}.json"

        raise ValueError(f"Unsupported SDD model type: {model_type}")

    def save(self, model: T, feature_id: Optional[str] = None) -> Path:
        """Persist an SDD model to the agent_data directory."""
        path = self._get_path(type(model), feature_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(model.model_dump_json(indent=2))

        return path

    def load(
        self, model_type: Type[T], feature_id: Optional[str] = None
    ) -> Optional[T]:
        """Load an SDD model from the agent_data directory."""
        path = self._get_path(model_type, feature_id)
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return model_type.model_validate(data)

    def get_parallel_opportunities(self, task_list: Tasks) -> List[List[str]]:
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

    def export_to_markdown(self, model: Union[Spec, Tasks], feature_id: str) -> Path:
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
        md = [f"# Spec: {spec.title} ({spec.feature_id})\n"]
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

    def import_from_markdown(
        self, markdown_path: Union[str, Path], feature_id: str
    ) -> Tasks:
        """Parse a tasks.md file back into a structured Tasks model.

        Supports spec-kit [P] markers for parallel execution detection.
        """
        path = Path(markdown_path)
        content = path.read_text(encoding="utf-8")

        tasks = []
        # Pattern for "### [ ] T1: Title [P]"
        task_pattern = re.compile(
            r"### \[(?P<status>[ xX])\] (?P<id>[A-Za-z0-9_-]+): (?P<title>.*?)(?P<parallel> \[P\])?$"
        )

        current_task = None
        for line in content.splitlines():
            match = task_pattern.match(line)
            if match:
                if current_task:
                    tasks.append(current_task)

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
            tasks.append(current_task)

        return Tasks(feature_id=feature_id, tasks=tasks)
