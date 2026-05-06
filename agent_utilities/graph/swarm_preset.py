"""Swarm Preset Template Engine — YAML-driven multi-agent workflows.

CONCEPT:ORCH-1.4 — Swarm Preset Template Engine
"""

from __future__ import annotations

import logging
from collections import deque

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SwarmAgentDef(BaseModel):
    """Agent role definition within a swarm preset. CONCEPT:ORCH-1.4"""

    id: str
    role: str = ""
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    model_name: str | None = None
    max_iterations: int = 25
    timeout_seconds: int = 300


class SwarmTaskDef(BaseModel):
    """Task definition in the swarm DAG. CONCEPT:ORCH-1.4"""

    id: str
    agent_id: str
    prompt: str = ""
    depends_on: list[str] = Field(default_factory=list)
    input_from: dict[str, str] = Field(default_factory=dict)


class SwarmPresetDef(BaseModel):
    """Complete swarm preset definition. CONCEPT:ORCH-1.4"""

    name: str
    description: str = ""
    agents: list[SwarmAgentDef] = Field(default_factory=list)
    tasks: list[SwarmTaskDef] = Field(default_factory=list)
    variables: dict[str, str] = Field(default_factory=dict)


class SwarmPresetEngine:
    """Engine for loading, validating, and executing swarm presets.

    CONCEPT:ORCH-1.4 — Swarm Preset Template Engine

    Provides YAML loading, DAG resolution via topological sort, parallel
    dispatch identification, and variable substitution in prompts.
    """

    def load_from_yaml(self, yaml_content: str) -> SwarmPresetDef:
        """Load a swarm preset from YAML content.

        Args:
            yaml_content: Raw YAML string defining the preset.

        Returns:
            A validated ``SwarmPresetDef``.

        Raises:
            ValueError: If the YAML is invalid or missing required fields.
        """
        try:
            raw = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}") from e

        if not isinstance(raw, dict):
            raise ValueError("YAML root must be a mapping")

        agents = [SwarmAgentDef(**a) for a in raw.get("agents", [])]
        tasks = [SwarmTaskDef(**t) for t in raw.get("tasks", [])]

        return SwarmPresetDef(
            name=raw.get("name", "unnamed"),
            description=raw.get("description", ""),
            agents=agents,
            tasks=tasks,
            variables=raw.get("variables", {}),
        )

    def validate_dag(self, preset: SwarmPresetDef) -> list[str]:
        """Validate the task DAG for cycles and missing references.

        Args:
            preset: The preset to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []
        task_ids = {t.id for t in preset.tasks}
        agent_ids = {a.id for a in preset.agents}

        for task in preset.tasks:
            if task.agent_id not in agent_ids:
                errors.append(
                    f"Task '{task.id}' references unknown agent '{task.agent_id}'"
                )
            for dep in task.depends_on:
                if dep not in task_ids:
                    errors.append(f"Task '{task.id}' depends on unknown task '{dep}'")

        # Cycle detection via Kahn's algorithm
        in_degree: dict[str, int] = {t.id: 0 for t in preset.tasks}
        adjacency: dict[str, list[str]] = {t.id: [] for t in preset.tasks}
        for task in preset.tasks:
            for dep in task.depends_on:
                if dep in adjacency:
                    adjacency[dep].append(task.id)
                    in_degree[task.id] = in_degree.get(task.id, 0) + 1

        queue: deque[str] = deque(tid for tid, deg in in_degree.items() if deg == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for neighbor in adjacency.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited < len(preset.tasks):
            errors.append("Task DAG contains a cycle")

        return errors

    def resolve_dag(self, preset: SwarmPresetDef) -> list[list[str]]:
        """Resolve the task DAG into parallel execution layers.

        Each layer contains tasks that can execute in parallel.

        Args:
            preset: The preset to resolve.

        Returns:
            List of layers, each a list of task IDs.

        Raises:
            ValueError: If the DAG is invalid.
        """
        errors = self.validate_dag(preset)
        if errors:
            raise ValueError(f"DAG validation failed: {'; '.join(errors)}")

        in_degree: dict[str, int] = {t.id: 0 for t in preset.tasks}
        adjacency: dict[str, list[str]] = {t.id: [] for t in preset.tasks}

        for task in preset.tasks:
            for dep in task.depends_on:
                adjacency[dep].append(task.id)
                in_degree[task.id] += 1

        current_layer = [tid for tid, deg in in_degree.items() if deg == 0]
        layers: list[list[str]] = []

        while current_layer:
            layers.append(sorted(current_layer))
            next_layer: list[str] = []
            for node in current_layer:
                for neighbor in adjacency[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_layer.append(neighbor)
            current_layer = next_layer

        return layers

    def substitute_variables(
        self,
        preset: SwarmPresetDef,
        user_vars: dict[str, str] | None = None,
    ) -> SwarmPresetDef:
        """Apply variable substitution to all task prompts.

        Variables are ``{variable_name}`` in prompts. User vars override defaults.

        Args:
            preset: The preset with templates.
            user_vars: User-provided overrides.

        Returns:
            A new ``SwarmPresetDef`` with substituted prompts.
        """
        merged_vars = {**preset.variables, **(user_vars or {})}

        substituted_tasks: list[SwarmTaskDef] = []
        for task in preset.tasks:
            prompt = task.prompt
            for key, value in merged_vars.items():
                prompt = prompt.replace(f"{{{key}}}", value)
            substituted_tasks.append(task.model_copy(update={"prompt": prompt}))

        return preset.model_copy(
            update={"tasks": substituted_tasks, "variables": merged_vars}
        )

    def get_parallel_tasks(self, preset: SwarmPresetDef) -> list[str]:
        """Identify root tasks with no dependencies.

        Returns:
            List of task IDs that can run immediately.
        """
        return [t.id for t in preset.tasks if not t.depends_on]

    def to_yaml(self, preset: SwarmPresetDef) -> str:
        """Serialize a preset definition to YAML."""
        return yaml.dump(
            preset.model_dump(exclude_defaults=True),
            default_flow_style=False,
            sort_keys=False,
        )
