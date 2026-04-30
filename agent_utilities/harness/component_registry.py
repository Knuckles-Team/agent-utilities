"""Harness Component Registry.

CONCEPT:AU-012 — Agentic Harness Engineering (Component Observability)

Tracks which filesystem paths correspond to which AHE component types,
enabling file-level version tracking and granular git rollbacks.

The registry serves as the bridge between the **normative state**
(filesystem files) and the **causal boundary** (change manifests):
    - Register a file as a harness component
    - Record edits with git commit SHAs
    - Roll back specific components without affecting others
    - Query edit history for a component or file
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from typing import Any

from pydantic import BaseModel, Field

from .manifest import ChangeManifest, ComponentEdit, ComponentType

logger = logging.getLogger(__name__)


class ComponentRegistration(BaseModel):
    """A registered harness component with its file path and metadata.

    Attributes:
        file_path: Relative path to the component file.
        component_type: AHE component category.
        description: Human-readable description.
        registered_at: ISO 8601 timestamp of registration.
        edit_history: List of ComponentEdit IDs applied to this file.
    """

    file_path: str
    component_type: ComponentType
    description: str = ""
    registered_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    edit_history: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HarnessComponentRegistry:
    """Maps file paths to AHE component types with version tracking.

    This registry is the foundation for **component observability** —
    it ensures every harness file is independently trackable and
    rollbackable.

    The registry persists to ``.specify/harness_registry.json`` and
    can be loaded/saved alongside SDD artifacts.

    Args:
        workspace_path: Root path of the workspace for resolving
            relative file paths and git operations.
    """

    def __init__(self, workspace_path: str) -> None:
        self.workspace_path = workspace_path
        self._components: dict[str, ComponentRegistration] = {}
        self._registry_path = os.path.join(
            workspace_path, ".specify", "harness_registry.json"
        )
        self._load()

    def _load(self) -> None:
        """Load the registry from disk if it exists."""
        if os.path.exists(self._registry_path):
            try:
                with open(self._registry_path) as f:
                    data = json.load(f)
                for path, entry in data.items():
                    self._components[path] = ComponentRegistration.model_validate(entry)
                logger.info(
                    f"HarnessComponentRegistry: Loaded {len(self._components)} components."
                )
            except Exception as e:
                logger.warning(
                    f"HarnessComponentRegistry: Failed to load registry: {e}"
                )

    def _save(self) -> None:
        """Persist the registry to disk."""
        os.makedirs(os.path.dirname(self._registry_path), exist_ok=True)
        data = {path: reg.model_dump() for path, reg in self._components.items()}
        with open(self._registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def register_component(
        self,
        file_path: str,
        component_type: ComponentType,
        description: str = "",
    ) -> None:
        """Register a file as a harness component.

        Args:
            file_path: Relative path to the component file.
            component_type: The AHE component category.
            description: Optional human-readable description.
        """
        self._components[file_path] = ComponentRegistration(
            file_path=file_path,
            component_type=component_type,
            description=description,
        )
        self._save()
        logger.info(f"Registered component: {file_path} as {component_type.value}")

    def register_defaults(self) -> None:
        """Register the standard agent-utilities components.

        This pre-populates the registry with known component-to-file
        mappings for the agent-utilities harness.
        """
        defaults = [
            # System Prompts
            (
                "agent_utilities/prompts/main_agent.json",
                ComponentType.SYSTEM_PROMPT,
                "Primary agent system prompt",
            ),
            (
                "agent_utilities/prompt_builder.py",
                ComponentType.SYSTEM_PROMPT,
                "Prompt construction and resolution",
            ),
            (
                "agent_utilities/structured_prompts.py",
                ComponentType.SYSTEM_PROMPT,
                "JSON-as-Code prompt models",
            ),
            # Tool Descriptions
            (
                "agent_utilities/tool_filtering.py",
                ComponentType.TOOL_DESCRIPTION,
                "Tool/skill filtering and metadata",
            ),
            # Tool Implementations
            (
                "agent_utilities/tools/developer_tools.py",
                ComponentType.TOOL_IMPLEMENTATION,
                "Developer tool implementations",
            ),
            (
                "agent_utilities/tools/workspace_tools.py",
                ComponentType.TOOL_IMPLEMENTATION,
                "Workspace tool implementations",
            ),
            (
                "agent_utilities/tools/knowledge_tools.py",
                ComponentType.TOOL_IMPLEMENTATION,
                "Knowledge graph tools",
            ),
            (
                "agent_utilities/tools/self_improvement_tools.py",
                ComponentType.TOOL_IMPLEMENTATION,
                "Self-improvement tools",
            ),
            # Middleware
            (
                "agent_utilities/middlewares.py",
                ComponentType.MIDDLEWARE,
                "JWT extraction and logging middleware",
            ),
            (
                "agent_utilities/guardrails.py",
                ComponentType.MIDDLEWARE,
                "Policy enforcement engine",
            ),
            (
                "agent_utilities/tool_guard.py",
                ComponentType.MIDDLEWARE,
                "Tool security and approval middleware",
            ),
            # Sub-Agents
            (
                "agent_utilities/graph/steps.py",
                ComponentType.SUB_AGENT,
                "HSM graph step definitions",
            ),
            # Long-Term Memory
            (
                "agent_utilities/knowledge_graph/engine.py",
                ComponentType.LONG_TERM_MEMORY,
                "Intelligence graph engine",
            ),
        ]
        for path, ctype, desc in defaults:
            if path not in self._components:
                self.register_component(path, ctype, desc)

    def get_component(self, file_path: str) -> ComponentRegistration | None:
        """Look up a component registration by file path."""
        return self._components.get(file_path)

    def get_components_by_type(
        self, component_type: ComponentType
    ) -> list[ComponentRegistration]:
        """Get all components of a specific type."""
        return [
            reg
            for reg in self._components.values()
            if reg.component_type == component_type
        ]

    def get_all_components(self) -> dict[str, ComponentRegistration]:
        """Return all registered components."""
        return dict(self._components)

    def record_edit(self, file_path: str, edit_id: str) -> None:
        """Record that an edit was applied to a component.

        Args:
            file_path: The edited file's relative path.
            edit_id: The ComponentEdit ID to record.
        """
        reg = self._components.get(file_path)
        if reg:
            reg.edit_history.append(edit_id)
            self._save()

    def get_component_history(self, file_path: str) -> list[str]:
        """Get the edit history for a component.

        Args:
            file_path: The component file path.

        Returns:
            List of ComponentEdit IDs in chronological order.
        """
        reg = self._components.get(file_path)
        return reg.edit_history if reg else []

    def get_git_log(
        self, file_path: str, max_entries: int = 10
    ) -> list[dict[str, str]]:
        """Get the git log for a specific component file.

        Args:
            file_path: Relative path to the file.
            max_entries: Maximum number of log entries.

        Returns:
            List of dicts with 'sha', 'date', 'message' keys.
        """
        abs_path = os.path.join(self.workspace_path, file_path)
        if not os.path.exists(abs_path):
            return []

        try:
            result = subprocess.run(  # nosec B603 B607
                [
                    "git",
                    "log",
                    f"-{max_entries}",
                    "--pretty=format:%H|%aI|%s",
                    "--",
                    file_path,
                ],
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            entries = []
            for line in result.stdout.strip().split("\n"):
                if "|" in line:
                    parts = line.split("|", 2)
                    entries.append(
                        {
                            "sha": parts[0],
                            "date": parts[1],
                            "message": parts[2] if len(parts) > 2 else "",
                        }
                    )
            return entries
        except Exception as e:
            logger.warning(f"Failed to get git log for {file_path}: {e}")
            return []

    def rollback_component(self, file_path: str, to_commit: str) -> bool:
        """Roll back a single component file to a specific git commit.

        This performs a surgical git checkout of a single file, preserving
        all other files at their current state. This is the key enabler
        for AHE's granular component rollback.

        Args:
            file_path: Relative path to the component file.
            to_commit: Git commit SHA to restore from.

        Returns:
            True if the rollback succeeded.
        """
        try:
            result = subprocess.run(  # nosec B603 B607
                ["git", "checkout", to_commit, "--", file_path],
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                logger.info(f"Rolled back {file_path} to commit {to_commit[:8]}")
                return True
            else:
                logger.error(f"Rollback failed for {file_path}: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Rollback exception for {file_path}: {e}")
            return False

    def get_current_manifest(self) -> ChangeManifest:
        """Create a snapshot ChangeManifest from the current registry state.

        Returns:
            A ChangeManifest representing all currently registered components.
        """
        manifest = ChangeManifest()
        for path, reg in self._components.items():
            if reg.edit_history:
                manifest.add_edit(
                    ComponentEdit(
                        component_type=reg.component_type,
                        file_path=path,
                        edit_summary=f"Current state of {path}",
                    )
                )
        return manifest
