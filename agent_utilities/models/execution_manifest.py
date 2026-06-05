"""CONCEPT:ORCH-1.8 — Parallel Engine Execution Manifest.

The ``ExecutionManifest`` is the single universal input to the
``ParallelEngine``. Every execution — from a trivial 1-agent LLM call
to a 300-agent enterprise swarm — is expressed as a manifest.

Models:
    - ``AgentSpec``: Specification for a single agent invocation.
    - ``SynthesisSpec``: How to merge outputs from parallel agents.
    - ``ExecutionManifest``: The complete execution specification.
    - ``AgentExecutionResult``: Result from a single agent execution.
    - ``WaveResult``: Results from one execution wave.
    - ``ExecutionResult``: Complete execution outcome.

See docs/pillars/1_graph_orchestration/ORCH-1.8-Parallel_Engine.md
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

# ── Agent Specification ─────────────────────────────────────────────


class AgentSpec(BaseModel):
    """Specification for a single agent in the execution manifest.

    CONCEPT:ORCH-1.8 — Parallel Engine

    Each ``AgentSpec`` describes one logical agent invocation. Fan-out
    is expressed via ``partitions``: if set, the agent is invoked once
    per partition with ``{{partition}}`` replaced in ``task_template``.

    Attributes:
        agent_id: Which agent prompt/type to invoke.
        role: Functional role (researcher, auditor, etc.).
        department: Hierarchical department for company-scale topology.
        tools: MCP server tools this agent has access to.
        model_id: Override model selection (empty = use default).
        system_prompt: Override system prompt (empty = use default).
        task_template: Task description with optional ``{{partition}}``.
        partitions: What this agent works on (if fan-out).
        depends_on: Agent IDs this depends on (DAG edges).
        timeout: Override per-agent timeout in seconds.
        memory_channels: KG memory channels for this agent.
    """

    agent_id: str
    role: str = ""
    department: str = "general"
    tools: list[str] = Field(default_factory=list)
    model_id: str = ""
    system_prompt: str = ""
    task_template: str = ""
    partitions: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    timeout: float | None = None
    memory_channels: list[str] = Field(default_factory=lambda: ["episodic"])
    # CONCEPT:ORCH-1.32 — KG-Governed Agent Swarm
    # (per-agent swarm fields below; all default to current behaviour):
    success_criteria: str = ""  # SWARM-2: what "done correctly" means (verify gate)
    output_schema: str | None = None  # SWARM-4: JSON schema/shape the agent must return
    model_role: str = ""  # SWARM-6: ModelRole to route to when model_id is unset
    max_retries: int = 0  # SWARM-5: retries-with-backoff on failure (0 = off)


# ── Synthesis Specification ─────────────────────────────────────────


class SynthesisSpec(BaseModel):
    """How to merge outputs from parallel agents.

    CONCEPT:ORCH-1.26 — RLM-Native Hierarchical Synthesis

    Strategies:
        - ``auto``: Engine selects based on agent count and output size.
        - ``flat``: Simple concatenation (for ≤10 agents).
        - ``hierarchical``: Group → sub-summaries → final summary.
        - ``progressive``: Stream synthesis as agents complete.
        - ``rlm``: Full RLM environment for massive-scale programmatic synthesis.

    Attributes:
        strategy: Synthesis strategy to use.
        ratio: For hierarchical: how many outputs per synthesis sub-node.
        model_id: Model for synthesis LLM calls (empty = use default).
        output_schema: Optional Pydantic model name for typed output.
    """

    strategy: Literal["auto", "flat", "hierarchical", "progressive", "rlm"] = "auto"
    ratio: int = Field(default=10, ge=2, le=100)
    model_id: str = ""
    output_schema: str | None = None


# ── Execution Manifest ──────────────────────────────────────────────


class ExecutionManifest(BaseModel):
    """Universal execution specification for the Parallel Engine.

    CONCEPT:ORCH-1.8 — Parallel Engine

    This is the **only** input to the engine. Everything — from a single
    agent query to a 300-agent enterprise swarm — is expressed as a
    manifest. Manifest generators convert from various sources
    (planners, workflows, TeamConfigs, skill workflows, KG presets)
    into this unified format.

    Attributes:
        manifest_id: Unique identifier for this execution.
        name: Human-readable name for logging and KG persistence.
        agents: The agents to execute.
        synthesis: How to merge outputs.
        coordination_protocol: Protocol from CoordinationLayer.
        execution_mode: Execution strategy.
        max_concurrency: Override global ``MAX_PARALLEL_AGENTS`` config.
        batch_size: Override global ``PARALLEL_BATCH_SIZE`` config.
        query: Original user query.
        context: Shared context injected into all agents.
        metadata: Workflow/TeamConfig/Skill provenance metadata.
        source: How this manifest was generated.
        kg_template_id: If materialized from a KG SwarmTemplate.
    """

    manifest_id: str = Field(default_factory=lambda: f"manifest:{uuid.uuid4().hex[:8]}")
    name: str = ""
    agents: list[AgentSpec]
    synthesis: SynthesisSpec = Field(default_factory=SynthesisSpec)
    coordination_protocol: str = "auto"
    execution_mode: Literal["auto", "sequential", "parallel", "mixed", "wave"] = "auto"
    max_concurrency: int | None = None
    batch_size: int | None = None
    query: str = ""
    context: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    kg_template_id: str | None = None

    @property
    def agent_count(self) -> int:
        """Total agent invocations including fan-out partitions."""
        return sum(max(1, len(a.partitions)) for a in self.agents)

    @property
    def is_trivial(self) -> bool:
        """Single agent, no fan-out — fast-path to inline execution."""
        return len(self.agents) == 1 and not self.agents[0].partitions

    @property
    def has_dependencies(self) -> bool:
        """Whether any agent has explicit DAG dependencies."""
        return any(a.depends_on for a in self.agents)

    @classmethod
    def from_graph_plan(
        cls, plan: Any, name: str = "", query: str = ""
    ) -> ExecutionManifest:
        """Convert a ``GraphPlan`` into an ``ExecutionManifest``.

        CONCEPT:ORCH-1.8 — GraphPlan → Manifest Bridge

        Enables ``WorkflowRunner`` and ``SkillCompiler`` outputs to flow
        directly into ``ParallelEngine`` without re-planning.

        Args:
            plan: A ``GraphPlan`` instance (from models.graph).
            name: Human-readable workflow name.
            query: Original user query, if any.

        Returns:
            An ``ExecutionManifest`` ready for ``ParallelEngine.execute()``.
        """
        agents = []
        for step in plan.steps:
            agents.append(
                AgentSpec(
                    agent_id=step.id,
                    task_template=step.refined_subtask or f"Execute: {step.id}",
                    depends_on=step.depends_on or [],
                    timeout=getattr(step, "timeout", None),
                )
            )
        return cls(
            agents=agents,
            name=name,
            query=query,
            source="graph_plan",
            metadata=plan.metadata
            if hasattr(plan, "metadata") and plan.metadata
            else {},
        )


# ── Execution Results ───────────────────────────────────────────────


class AgentExecutionResult(BaseModel):
    """Result from a single agent execution within the Parallel Engine.

    CONCEPT:ORCH-1.8 — Parallel Engine

    Attributes:
        agent_id: Which agent produced this result.
        role: Role of the agent.
        partition: Which partition this result covers (if fan-out).
        output: The agent's output text.
        success: Whether execution succeeded.
        error: Error message if failed.
        duration_ms: Wall-clock execution time in milliseconds.
        model_id: Which model was actually used.
        token_usage: Token usage statistics if available.
        metadata: Additional execution metadata.
    """

    agent_id: str
    role: str = ""
    partition: str = ""
    output: str = ""
    success: bool = True
    error: str = ""
    duration_ms: float = 0.0
    model_id: str = ""
    token_usage: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WaveResult(BaseModel):
    """Results from one execution wave.

    CONCEPT:ORCH-1.8 — Parallel Engine

    A wave is a group of agents that execute concurrently within
    a single scheduling epoch. Multiple waves execute sequentially,
    respecting DAG dependencies.

    Attributes:
        wave_index: Zero-based wave number.
        results: Individual agent results from this wave.
        duration_ms: Total wave execution time.
        success_rate: Fraction of agents that succeeded.
    """

    wave_index: int = 0
    results: list[AgentExecutionResult] = Field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """Fraction of agents that succeeded in this wave."""
        if not self.results:
            return 1.0
        return sum(1 for r in self.results if r.success) / len(self.results)


class ExecutionResult(BaseModel):
    """Complete execution outcome from the Parallel Engine.

    CONCEPT:ORCH-1.8 — Parallel Engine

    Attributes:
        manifest_id: ID of the executed manifest.
        execution_id: KG-persisted execution node ID.
        synthesis_output: Final synthesized output.
        wave_results: Per-wave results.
        agent_count: Total number of agent invocations.
        protocol: Coordination protocol that was applied.
        total_duration_ms: End-to-end execution time.
        synthesis_strategy: Which synthesis strategy was actually used.
        success: Overall execution success.
        timestamp: ISO timestamp of execution completion.
    """

    manifest_id: str = ""
    execution_id: str = ""
    synthesis_output: str = ""
    mermaid: str | None = None
    wave_results: list[WaveResult] = Field(default_factory=list)
    agent_count: int = 0
    protocol: str = ""
    total_duration_ms: float = 0.0
    synthesis_strategy: str = ""
    success: bool = True
    # CONCEPT:ORCH-1.32 — KG-Governed Agent Swarm
    # (swarm telemetry & verification fields; additive)
    critical_path_length: int = (
        0  # SWARM-3: longest dependency chain (true wall-clock floor)
    )
    parallelism_ratio: float = (
        1.0  # SWARM-3: agents / critical_path (higher = more parallel)
    )
    wave_count: int = 0  # SWARM-7
    verification: dict[str, Any] = Field(default_factory=dict)  # SWARM-2 verify summary
    telemetry: dict[str, Any] = Field(
        default_factory=dict
    )  # SWARM-7 per-wave cost/latency
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    @property
    def all_results(self) -> list[AgentExecutionResult]:
        """Flat list of all agent results across all waves."""
        return [r for w in self.wave_results for r in w.results]

    @property
    def overall_success_rate(self) -> float:
        """Overall success rate across all waves."""
        results = self.all_results
        if not results:
            return 1.0
        return sum(1 for r in results if r.success) / len(results)
