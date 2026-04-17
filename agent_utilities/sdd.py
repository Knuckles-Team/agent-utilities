#!/usr/bin/python
# coding: utf-8
"""SDD (Spec-Driven Development) Utility Module.

This module provides high-level utilities for managing structured SDD artifacts
(Specifications, Plans, Constitutions) and their relationship to tasks. It handles
disk persistence in the agent's 'agent_data' directory and provides logic for
dependency analysis.
"""

import json
from pathlib import Path
from typing import List, Optional, Type, TypeVar, Union

from .models import (
    ProjectConstitution,
    FeatureSpec,
    ImplementationPlan,
    TaskList,
)

T = TypeVar("T", ProjectConstitution, FeatureSpec, ImplementationPlan, TaskList)


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

        if model_type == FeatureSpec:
            return self.agent_data / "specs" / f"{feature_id}.json"

        if model_type == ImplementationPlan:
            return self.agent_data / "plans" / f"{feature_id}.json"

        if model_type == TaskList:
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

    def get_parallel_opportunities(self, task_list: TaskList) -> List[List[str]]:
        """Identify groups of tasks that can be safely run in parallel.

        Calculates parallel groups based on:
        1. Explicit dependency graph (tasks without dependencies or with completed deps).
        2. File collision detection (tasks affecting same files cannot be parallel).
        """
        completed = set()
        all_tasks = {}
        for phase in task_list.phases:
            for task in phase.tasks:
                all_tasks[task.id] = task
                if task.status == "completed":
                    completed.add(task.id)

        pending = [t for t in all_tasks.values() if t.status != "completed"]

        groups = []
        current_batch = []
        occupied_files = set()

        for task in pending:
            # Check if all dependencies are met
            deps_met = all(d in completed for d in task.dependencies)

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
