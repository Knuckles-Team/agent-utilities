"""Workflow Catalog — Externally Consumable Workflow Definitions.

CONCEPT:ORCH-1.24 — Workflow Catalog

Defines reusable orchestration scenarios as structured ``WorkflowScenario``
objects, loadable from YAML and persistable into the Knowledge Graph.

The catalog serves as the **single source of truth** for all predefined
workflows. Each scenario specifies:

- Which agents/MCP servers to invoke
- The task for each step
- Dependencies between steps (sequential vs parallel)
- Expected outcomes for validation
- Required environment variables and MCP servers

Usage::

    from agent_utilities.workflows.catalog import WorkflowCatalog

    # Load from YAML
    catalog = WorkflowCatalog.load()

    # Convert to GraphPlans
    plans = catalog.to_graph_plans()

    # Persist all into KG
    catalog.register_in_kg(engine)

    # Export as JSON for external consumers
    catalog.export_json("/tmp/workflows.json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_utilities.models.graph import ExecutionStep, GraphPlan

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """A single step in a workflow scenario.

    CONCEPT:ORCH-1.24 — Workflow Step Definition

    Attributes:
        agent: The MCP server or agent name to route to.
        task: The natural language task description.
        expected: Keywords expected in the output for validation.
        depends_on: Indices of steps that must complete first.
            Empty list means this step can run in parallel with others.
        timeout: Per-step timeout in seconds.
        is_parallel: Whether this step should run concurrently with
            siblings that share no dependencies.
    """

    agent: str
    task: str
    expected: list[str] = field(default_factory=list)
    depends_on: list[int] = field(default_factory=list)
    timeout: float = 3600.0
    is_parallel: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "agent": self.agent,
            "task": self.task,
            "expected": self.expected,
            "depends_on": self.depends_on,
            "timeout": self.timeout,
            "is_parallel": self.is_parallel,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowStep:
        """Deserialize from a dict."""
        return cls(
            agent=data["agent"],
            task=data["task"],
            expected=data.get("expected", []),
            depends_on=data.get("depends_on", []),
            timeout=data.get("timeout", 3600.0),
            is_parallel=data.get("is_parallel", False),
        )


@dataclass
class WorkflowScenario:
    """A complete workflow scenario with ordered steps.

    CONCEPT:ORCH-1.24 — Workflow Scenario Definition

    Attributes:
        name: Unique identifier for the workflow.
        description: Human-readable purpose description.
        domain: Functional domain (infrastructure, research, etc).
        steps: Ordered list of workflow steps.
        tags: Tags for discovery and filtering.
        requires: Environment variables or MCP servers needed.
        timeout_seconds: Total workflow timeout.
        version: Scenario version (auto-incremented in KG).
    """

    name: str
    description: str
    domain: str = "general"
    steps: list[WorkflowStep] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    timeout_seconds: int = 300
    version: int = 1

    def to_graph_plan(self) -> GraphPlan:
        """Convert this scenario to an executable GraphPlan.

        CONCEPT:ORCH-1.24 — Scenario→Plan Conversion

        Returns:
            A ``GraphPlan`` with properly wired dependencies.
        """
        execution_steps: list[ExecutionStep] = []
        for i, step in enumerate(self.steps):
            # Resolve depends_on indices to node_id references
            depends_on_ids = []
            for dep_idx in step.depends_on:
                if 0 <= dep_idx < len(self.steps):
                    depends_on_ids.append(self.steps[dep_idx].agent)

            # Determine parallelism: parallel if no dependencies AND
            # not the first step
            is_parallel = step.is_parallel or (i > 0 and not step.depends_on)

            execution_steps.append(
                ExecutionStep(
                    node_id=step.agent,
                    refined_subtask=step.task,
                    is_parallel=is_parallel,
                    depends_on=depends_on_ids,
                    access_list=depends_on_ids,
                    timeout=step.timeout,
                )
            )

        return GraphPlan(
            steps=execution_steps,
            metadata={
                "source": "workflow_catalog",
                "scenario_name": self.name,
                "domain": self.domain,
                "tags": self.tags,
                "version": self.version,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "steps": [s.to_dict() for s in self.steps],
            "tags": self.tags,
            "requires": self.requires,
            "timeout_seconds": self.timeout_seconds,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowScenario:
        """Deserialize from a dict."""
        steps = [WorkflowStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            domain=data.get("domain", "general"),
            steps=steps,
            tags=data.get("tags", []),
            requires=data.get("requires", []),
            timeout_seconds=data.get("timeout_seconds", 300),
            version=data.get("version", 1),
        )


class WorkflowCatalog:
    """Registry of all available workflow scenarios.

    CONCEPT:ORCH-1.24 — Workflow Catalog

    Provides loading from YAML, conversion to ``GraphPlan``,
    persistence to the KG, and export for external consumers.

    Usage::

        catalog = WorkflowCatalog.load()
        catalog.register_in_kg(engine)

        # Or load from a custom path
        catalog = WorkflowCatalog.load_from_yaml("/path/to/catalog.yaml")
    """

    def __init__(self, scenarios: list[WorkflowScenario] | None = None) -> None:
        self.scenarios: list[WorkflowScenario] = scenarios or []

    @classmethod
    def load(cls) -> WorkflowCatalog:
        """Load the built-in catalog from the package resources.

        CONCEPT:ORCH-1.24 — Package-Embedded Catalog

        Returns:
            A WorkflowCatalog with all built-in scenarios.
        """
        # Try importlib.resources first (package-embedded)
        try:
            catalog_path = Path(__file__).parent / "catalog.yaml"
            if catalog_path.exists():
                return cls.load_from_yaml(catalog_path)
        except Exception:
            pass  # nosec B110 — fallback to empty catalog is expected

        logger.warning("[ORCH-1.24] No built-in catalog found, returning empty catalog")
        return cls()

    @classmethod
    def load_from_yaml(cls, path: str | Path) -> WorkflowCatalog:
        """Load a workflow catalog from a YAML file.

        CONCEPT:ORCH-1.24 — YAML Catalog Loading

        Args:
            path: Path to the YAML catalog file.

        Returns:
            A WorkflowCatalog with parsed scenarios.
        """
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Catalog file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        scenarios = []
        for wf_data in data.get("workflows", []):
            scenario = WorkflowScenario.from_dict(wf_data)
            scenarios.append(scenario)

        logger.info(
            "[ORCH-1.24] Loaded %d workflow scenarios from %s",
            len(scenarios),
            path,
        )
        return cls(scenarios=scenarios)

    @classmethod
    def load_from_json(cls, path: str | Path) -> WorkflowCatalog:
        """Load a workflow catalog from a JSON file.

        Args:
            path: Path to the JSON catalog file.

        Returns:
            A WorkflowCatalog with parsed scenarios.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        scenarios = [WorkflowScenario.from_dict(wf) for wf in data.get("workflows", [])]
        return cls(scenarios=scenarios)

    def get(self, name: str) -> WorkflowScenario | None:
        """Get a scenario by name."""
        for s in self.scenarios:
            if s.name == name:
                return s
        return None

    def filter_by_tag(self, tag: str) -> list[WorkflowScenario]:
        """Filter scenarios by tag."""
        return [s for s in self.scenarios if tag in s.tags]

    def filter_by_domain(self, domain: str) -> list[WorkflowScenario]:
        """Filter scenarios by domain."""
        return [s for s in self.scenarios if s.domain == domain]

    def to_graph_plans(self) -> dict[str, GraphPlan]:
        """Convert all scenarios to GraphPlan objects.

        Returns:
            Dict mapping scenario name to GraphPlan.
        """
        return {s.name: s.to_graph_plan() for s in self.scenarios}

    def register_in_kg(
        self,
        engine: IntelligenceGraphEngine,
    ) -> list[str]:
        """Persist all scenarios into the Knowledge Graph.

        CONCEPT:ORCH-1.24 — KG Catalog Registration

        Uses ``WorkflowStore`` to persist each scenario as a
        ``WorkflowDefinition`` with linked ``WorkflowStep`` nodes.
        Auto-increments version if a workflow with the same name exists.

        Args:
            engine: The IntelligenceGraphEngine for persistence.

        Returns:
            List of workflow IDs created/updated.
        """
        from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

        store = WorkflowStore(engine)
        workflow_ids: list[str] = []

        for scenario in self.scenarios:
            plan = scenario.to_graph_plan()

            # Check for existing version
            existing = store.load_workflow(scenario.name)
            version = scenario.version
            if existing:
                # Auto-increment version
                existing_meta = existing.metadata or {}
                version = existing_meta.get("version", 0) + 1
                logger.info(
                    "[ORCH-1.24] Auto-incrementing '%s' to version %d",
                    scenario.name,
                    version,
                )

            plan.metadata["version"] = version

            wid = store.save_workflow(
                name=scenario.name,
                plan=plan,
                description=scenario.description,
                nl_spec=scenario.description,
                metadata={
                    "domain": scenario.domain,
                    "tags": scenario.tags,
                    "requires": scenario.requires,
                    "timeout_seconds": scenario.timeout_seconds,
                    "version": version,
                    "source": "workflow_catalog",
                },
            )
            workflow_ids.append(wid)
            logger.info(
                "[ORCH-1.24] Registered workflow '%s' → %s (v%d)",
                scenario.name,
                wid,
                version,
            )

        return workflow_ids

    def export_json(self, path: str | Path) -> Path:
        """Export the catalog as a JSON file for external consumers.

        Args:
            path: Output file path.

        Returns:
            Path to the written JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "source": "agent_utilities.workflows.catalog",
            "workflow_count": len(self.scenarios),
            "workflows": [s.to_dict() for s in self.scenarios],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(
            "[ORCH-1.24] Exported %d workflows to %s", len(self.scenarios), path
        )
        return path

    def export_yaml(self, path: str | Path) -> Path:
        """Export the catalog as a YAML file.

        Args:
            path: Output file path.

        Returns:
            Path to the written YAML file.
        """
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {"workflows": [s.to_dict() for s in self.scenarios]}

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info(
            "[ORCH-1.24] Exported %d workflows to %s", len(self.scenarios), path
        )
        return path

    def summary(self) -> str:
        """Generate a human-readable summary of the catalog.

        Returns:
            Formatted multi-line summary string.
        """
        lines = [f"Workflow Catalog: {len(self.scenarios)} scenarios\n"]
        for s in self.scenarios:
            lines.append(f"  [{s.domain}] {s.name} (v{s.version})")
            lines.append(f"    {s.description}")
            lines.append(f"    Steps: {len(s.steps)} | Tags: {', '.join(s.tags)}")
            lines.append(f"    Requires: {', '.join(s.requires)}")
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # KG-Resolved Agent Export
    # ------------------------------------------------------------------

    def resolve_agent_configs(
        self,
        engine: IntelligenceGraphEngine,
        scenario_name: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Resolve full agent configurations from the Knowledge Graph.

        CONCEPT:ORCH-1.24 — Agent Config Resolution

        For each unique agent referenced across workflow steps, queries the
        KG for the full configuration including:
        - Agent name and description
        - MCP server command/args/env
        - Tools provided (name + description)
        - System prompt
        - Skills and capabilities

        Args:
            engine: The IntelligenceGraphEngine to query.
            scenario_name: Optional — resolve only agents for this scenario.
                If None, resolves agents across all scenarios.

        Returns:
            Dict mapping agent name to its resolved configuration.
        """
        from agent_utilities.orchestration.agent_runner import _resolve_agent_from_kg

        # Resolve model config once for all agents
        try:
            from agent_utilities.core.config import (
                DEFAULT_LITE_LLM_MODEL_ID,
                DEFAULT_LLM_MODEL_ID,
            )
            from agent_utilities.core.config import (
                config as _cfg,
            )

            model_id = DEFAULT_LITE_LLM_MODEL_ID or DEFAULT_LLM_MODEL_ID or ""
            # Derive intelligence level from chat_models if available
            intelligence_level = "normal"
            if _cfg.chat_models:
                intelligence_level = _cfg.chat_models[0].intelligence_level
        except Exception:
            model_id = ""
            intelligence_level = "normal"

        # Collect unique agent names
        agent_names: set[str] = set()
        if scenario_name:
            scenario = self.get(scenario_name)
            if scenario:
                for step in scenario.steps:
                    agent_names.add(step.agent)
        else:
            for s in self.scenarios:
                for step in s.steps:
                    agent_names.add(step.agent)

        # Resolve each agent from the KG
        resolved: dict[str, dict[str, Any]] = {}
        for agent_name in sorted(agent_names):
            try:
                meta = _resolve_agent_from_kg(engine, agent_name)
                resolved[agent_name] = {
                    "name": agent_name,
                    "type": meta.get("type", "unknown"),
                    "description": meta.get("system_prompt", ""),
                    "server_id": meta.get("server_id", ""),
                    "mcp_command": meta.get("mcp_command", ""),
                    "mcp_url": meta.get("url", ""),
                    "env": meta.get("env", ""),
                    "tools": meta.get("tools", []),
                    "capabilities": meta.get("capabilities", []),
                    "system_prompt": meta.get("system_prompt", ""),
                    "model_id": model_id,
                    "intelligence_level": intelligence_level,
                }
                logger.info(
                    "[ORCH-1.24] Resolved agent '%s': type=%s, %d tools",
                    agent_name,
                    meta.get("type", "unknown"),
                    len(meta.get("tools", [])),
                )
            except Exception as e:
                logger.warning(
                    "[ORCH-1.24] Failed to resolve agent '%s': %s",
                    agent_name,
                    e,
                )
                resolved[agent_name] = {
                    "name": agent_name,
                    "type": "unresolved",
                    "description": "",
                    "tools": [],
                    "error": str(e),
                }

        return resolved

    def export_with_agents(
        self,
        engine: IntelligenceGraphEngine,
        path: str | Path | None = None,
        scenario_name: str | None = None,
    ) -> dict[str, Any]:
        """Export workflows with fully resolved agent configurations.

        CONCEPT:ORCH-1.24 — Rich Workflow Export

        Produces a comprehensive JSON payload containing:
        - All workflow scenarios (or a single one)
        - Full agent configurations (tools, system prompts, MCP details)
        - Enough information to reconstruct the workflow on another system

        Args:
            engine: The IntelligenceGraphEngine for agent resolution.
            path: Optional file path to write the JSON output.
            scenario_name: Optional — export only this workflow.

        Returns:
            The complete export dict with ``workflows`` and ``agents``.
        """
        # Select scenarios
        if scenario_name:
            scenario = self.get(scenario_name)
            scenarios = [scenario] if scenario else []
        else:
            scenarios = self.scenarios

        # Resolve agent configs
        agent_configs = self.resolve_agent_configs(engine, scenario_name)

        export_data = {
            "version": "1.0",
            "source": "agent_utilities.workflows.catalog",
            "workflow_count": len(scenarios),
            "workflows": [s.to_dict() for s in scenarios],
            "agents": agent_configs,
        }

        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
            logger.info(
                "[ORCH-1.24] Exported %d workflows + %d agents to %s",
                len(scenarios),
                len(agent_configs),
                path,
            )

        return export_data
