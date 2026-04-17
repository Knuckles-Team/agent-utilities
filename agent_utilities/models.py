#!/usr/bin/python
# coding: utf-8
"""Shared Data Models Module.

This module defines the Pydantic and dataclass models used for state management,
communication protocols, and configuration across the agent ecosystem. It includes
models for task tracking, memory, A2A registries, MCP tool metadata, and graph execution steps.
"""

import os
from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
from pydantic import BaseModel, Field

# Unified Execution Models defined below at line 220+

__version__ = "0.2.40"


class PeriodicTask(BaseModel):
    """Represents a scheduled task to be executed at regular intervals.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable name of the task.
        interval_minutes: Frequency of execution in minutes.
        prompt: The prompt used by the agent when executing the task.
        last_run: Timestamp of the last successful execution.
        active: Whether the task is currently enabled.

    """

    id: str
    name: str
    interval_minutes: int
    prompt: str
    last_run: datetime = Field(default_factory=datetime.now)
    active: bool = True


@dataclass
class AgentDeps:
    """Standard dependencies provided to agent tools via RunContext.

    These dependencies provide access to the workspace, session identifiers,
    communication queues for streaming and elicitation, and LLM configuration.

    Attributes:
        workspace_path: Path to the current active agent workspace.
        user_id: Identifier of the user interacting with the agent.
        session_id: Unique identifier for the current chat session.
        ssl_verify: Whether to verify SSL certificates for network requests.
        auth_token: Authentication token for API requests.
        elicitation_queue: Queue used to send elicitation requests back to the UI.
        graph_event_queue: Queue used to stream graph activity events to the UI.
        request_id: Correlation ID for the current request context.
        approval_timeout: Seconds to wait for human approval (0 = infinite).
        provider: LLM provider being used.
        model_id: Specific model identifier.
        base_url: Base URL for the LLM API.
        api_key: API key for the LLM provider.
        mcp_toolsets: List of initialized MCP toolsets available to the agent.

    """

    workspace_path: Path

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ssl_verify: bool = True
    auth_token: Optional[str] = None

    elicitation_queue: Optional[asyncio.Queue] = None
    graph_event_queue: Optional[asyncio.Queue] = None
    request_id: str = ""
    """Unique identifier for the current request context (correlation ID)."""
    approval_timeout: float = 0.0
    """Timeout in seconds for human approval. 0 = infinite."""
    provider: Optional[str] = None
    model_id: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    mcp_toolsets: List[Any] = field(default_factory=list)


class TaskStatus(str, Enum):
    """Enumeration of possible task execution states."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(BaseModel):
    """A single unit of work within a larger project or phase.

    Attributes:
        id: Unique identifier for the task.
        title: Concise title of the task.
        description: Detailed description of what needs to be accomplished.
        status: Current execution status (Pending, In Progress, etc.).
        assigned_to: The agent or user responsible for the task.
        dependencies: List of task IDs that must be completed before this one starts.
        result: The outcome or output of the task after completion.
        git_commit: Git commit hash associated with the task's implementation.
        metadata: Additional key-value pairs for task-specific data.

    """

    id: str = Field(default_factory=lambda: "task_" + os.urandom(4).hex())
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    result: Optional[str] = None
    git_commit: Optional[str] = None
    story_id: Optional[str] = Field(
        None, description="Associated user story ID (e.g., 'US1')"
    )
    file_paths: List[str] = Field(
        default_factory=list, description="List of files affected by this task"
    )
    tags: List[str] = Field(
        default_factory=list, description="Arbitrary tags for categorization"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskPhase(BaseModel):
    """A logical grouping of tasks representing a specific phase of a project.

    Attributes:
        name: Name of the phase (e.g., 'Research', 'Implementation').
        tasks: List of Tasks belonging to this phase.
        status: Overall status of the phase.

    """

    name: str
    goal: Optional[str] = Field(
        None, description="The high-level objective for this phase"
    )
    test_criteria: List[str] = Field(
        default_factory=list, description="Independent criteria to verify this phase"
    )
    tasks: List[Task] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING


class TaskList(BaseModel):
    """A comprehensive list of project phases and tasks used for tracking progress.

    Attributes:
        phases: Sequence of project phases.
        metadata: Project-level metadata.
        current_phase_index: Index of the currently active phase.
        version: Schema version of the task list.

    """

    phases: List[TaskPhase] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    current_phase_index: int = 0
    version: str = "1.0.0"


class ProgressEntry(BaseModel):
    """A single log entry recording progress or a milestone in a project.

    Attributes:
        timestamp: ISO 8601 timestamp of the entry.
        message: Description of the progress made.
        git_commit: Associated git commit hash, if applicable.
        logs: Detailed output or logs associated with this step.
        metadata: Additional context for the entry.

    """

    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat() + "Z")
    message: str
    git_commit: Optional[str] = None
    logs: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProgressLog(BaseModel):
    """A chronological log of progress entries for a project."""

    entries: List[ProgressEntry] = Field(default_factory=list)


class SprintContract(BaseModel):
    """A formal agreement defining the goals and success criteria for a development sprint.

    Attributes:
        goals: High-level objectives to be achieved during the sprint.
        definition_of_done: Criteria that must be met for a task to be considered 'Done'.
        test_criteria: Specific tests or validations required for acceptance.
        metadata: Sprint-specific metadata (e.g., dates, team).

    """

    goals: List[str] = Field(default_factory=list)
    definition_of_done: List[str] = Field(default_factory=list)
    test_criteria: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IdentityModel(BaseModel):
    """Definition of an agent's identity and persona.

    Attributes:
        name: Name of the agent.
        role: The specialized role (e.g., 'DevOps Engineer').
        emoji: Visual representation for UI.
        vibe: Description of the agent's personality or tone.
        system_prompt: The core system prompt defining behavior.

    """

    name: str = "AI Agent"
    role: str = ""
    emoji: str = "🤖"
    vibe: str = ""
    system_prompt: str = ""


class GraphResponse(BaseModel):
    """Standardized response from an agent graph execution.

    Attributes:
        status: Execution status ('pending', 'success', 'error').
        results: Dictionary containing the final output and intermediate results.
        mermaid: Mermaid JS string for visualizing the executed path.
        usage: Token usage statistics.
        error: Error message if execution failed.
        metadata: Additional execution metadata.

    """

    status: str = "pending"

    results: Dict[str, Any] = Field(default_factory=dict)
    mermaid: Optional[str] = None
    usage: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UserModel(BaseModel):
    """Metadata representing the human user interacting with the agent.

    Attributes:
        name: The display name of the user.
        emoji: Visual avatar for the user in the UI.

    """

    name: str = "User"
    emoji: str = "👤"


class A2APeerModel(BaseModel):
    """Metadata for a remote agent peer discovered via the A2A protocol.

    Attributes:
        name: Name of the remote agent.
        url: Endpoint URL for communicating with the agent.
        description: Summary of the agent's purpose.
        capabilities: Description of tools and skills offered by the agent.
        auth: Authentication method required.
        notes: Additional notes about the peer.

    """

    name: str
    url: str
    description: str = ""
    capabilities: str = ""
    auth: str = "none"
    notes: str = ""


class A2ARegistryModel(BaseModel):
    """A collection of A2A peer agents for discovery and communication.

    Attributes:
        peers: List of known remote agents.

    """

    peers: List[A2APeerModel] = Field(default_factory=list)


class MemoryEntryModel(BaseModel):
    """A single entry in an agent's long-term memory.

    Attributes:
        timestamp: ISO 8601 timestamp of when the memory was recorded.
        text: The actual content of the memory.

    """

    timestamp: str
    text: str


class MemoryModel(BaseModel):
    """A collection of long-term memory entries for an agent.

    Attributes:
        entries: Chronological list of memory records.

    """

    entries: List[MemoryEntryModel] = Field(default_factory=list)


class CronTaskModel(BaseModel):
    """Configuration for a scheduled periodic task.

    Attributes:
        id: Unique identifier for the task.
        name: Human-readable name.
        interval_minutes: Execution frequency in minutes.
        prompt: The prompt to be executed by the agent.
        last_run: Timestamp of last execution.
        next_approx: Estimated time of next execution.

    """

    id: str
    name: str
    interval_minutes: int
    prompt: str
    last_run: str = "—"
    next_approx: str = "—"


class CronRegistryModel(BaseModel):
    """A registry defining all scheduled periodic tasks for the workspace.

    Attributes:
        tasks: List of active cron task configurations.

    """

    tasks: List[CronTaskModel] = Field(default_factory=list)


class CronLogEntryModel(BaseModel):
    """A log entry recording the outcome of a scheduled task execution.

    Attributes:
        timestamp: Time of execution.
        task_id: ID of the executed task.
        task_name: Name of the executed task.
        status: Result status ('success', 'failed').
        message: Detailed result or error message.
        chat_id: Optional link to a chat session created for this task.

    """

    timestamp: str
    task_id: str
    task_name: str = ""
    status: str = "success"
    message: str = ""
    chat_id: Optional[str] = None


class CronLogModel(BaseModel):
    """Historical execution log for all scheduled periodic tasks.

    Attributes:
        entries: Chronological list of task execution results.

    """

    entries: List[CronLogEntryModel] = Field(default_factory=list)


class MCPConfigModel(BaseModel):
    """Structure for the Model Context Protocol configuration (mcp_config.json).

    Attributes:
        mcpServers: Map of server names to their specific configurations.

    """

    mcpServers: Dict[str, Any] = Field(default_factory=dict)


class MCPAgent(BaseModel):
    """Metadata for a specialized agent (Prompt, MCP, or A2A).

    Attributes:
        name: Unique identifier/tag for the agent (routing key).
        agent_type: The source type ('prompt', 'mcp', 'a2a').
        prompt_file: Path to the markdown prompt file (for 'prompt' agents).
        endpoint_url: Remote URL or stdio command (for 'a2a' or 'mcp' agents).
        description: Detailed description of the agent's specialization.
        system_prompt: The synthesized system prompt used to guide the agent.
        tools: List of tool names exposed by this agent.
        mcp_server: The source MCP server name (for 'mcp' agents).
        capabilities: List of skills/docs or rich capabilities string.
        mcp_tools: Specific MCP tool/tag patterns (for 'mcp' agents).
        extra_config: JSON/dict for additional discovery metadata.
        is_custom: Whether the agent was manually customized.
        tool_count: Number of tools assigned to this agent.
        avg_relevance_score: Mean relevance score across all assigned tools (0-100).

    """

    name: str = Field(description="Unique agent identifier / tag")
    agent_type: str = Field("prompt", description="Type: prompt, mcp, a2a")
    prompt_file: Optional[str] = Field(None, description="Markdown prompt file path")
    endpoint_url: Optional[str] = Field(None, description="Connection URL / cmd")
    description: str = Field("", description="Specialized agent description")
    system_prompt: str = Field("", description="Synthesized system prompt")
    tools: List[str] = Field(default_factory=list, description="Tool names")
    mcp_server: Optional[str] = Field(None, description="Source MCP server name")
    capabilities: List[str] = Field(
        default_factory=list, description="Skills/Capabilities"
    )
    mcp_tools: Optional[str] = Field(None, description="MCP tool/tag patterns")
    extra_config: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    is_custom: bool = Field(False, description="True if manually edited")
    tool_count: int = Field(default=0, description="Number of tools")
    avg_relevance_score: int = Field(default=0, description="Mean score (0-100)")


class MCPToolInfo(BaseModel):
    """Detailed metadata for a tool provided by an MCP server.

    Attributes:
        name: Full name of the tool.
        description: Description of what the tool does.
        tag: Primary tag used to assign this tool to a specific specialist agent.
        mcp_server: The source MCP server providing the tool.
        all_tags: All tags associated with the tool for flexible routing.
        relevance_score: Deterministic quality score (0-100) based on description
            richness, tag confidence, and name specificity.
        requires_approval: Whether this tool requires human-in-the-loop approval
            before execution.  Auto-populated by the tool guard during registry
            sync, but can be overridden manually in NODE_AGENTS.md.

    """

    name: str = Field(description="Full tool name")
    description: str = Field(description="Tool description")
    tag: Optional[str] = Field(None, description="Primary tool tag for partitioning")
    mcp_server: str = Field(description="Source MCP server")
    all_tags: List[str] = Field(
        default_factory=list, description="All tags associated with the tool"
    )
    relevance_score: int = Field(
        default=0, description="Deterministic quality score (0-100)"
    )
    requires_approval: bool = Field(
        default=False,
        description="Whether this tool requires human-in-the-loop approval",
    )


class MCPAgentRegistryModel(BaseModel):
    """A specialized registry containing discovered MCP specialists and their tools.

    Attributes:
        agents: List of derived specialist agents.
        tools: Full list of all available MCP tools.

    """

    agents: List[MCPAgent] = Field(default_factory=list)
    tools: List[MCPToolInfo] = Field(default_factory=list)


class DiscoveredSpecialist(BaseModel):
    """Unified representation of any specialist discovered during graph bootstrap.

    This model normalizes both local MCP agents (from ``NODE_AGENTS.md``) and
    remote A2A peers (from ``A2A_AGENTS.md``) into a single record so that the
    graph builder and router can treat them identically during the registration
    and planning phases.  Execution still follows type-specific paths.

    Attributes:
        tag: Routing key used by the dispatcher (e.g. ``'portainer'``).
        name: Human-readable display name.
        description: One-line summary of the specialist's purpose.
        source: Origin of the specialist (``'mcp'`` or ``'a2a'``).
        mcp_server: Source MCP server name (empty for A2A peers).
        tools: Known tool names (populated for MCP, may be empty for A2A).
        url: Endpoint URL (populated for A2A, empty for MCP).
        capabilities: Free-form capability metadata (A2A agent cards).

    """

    tag: str = Field(description="Routing key used by the dispatcher")
    name: str = Field(description="Human-readable display name")
    description: str = Field(default="", description="Specialist summary")
    source: str = Field(description="Origin: 'prompt', 'mcp', or 'a2a'")
    mcp_server: str = Field(default="", description="Source MCP server (MCP only)")
    tools: List[str] = Field(default_factory=list, description="Known tool names")
    url: str = Field(default="", description="Agent endpoint URL (A2A/MCP only)")
    capabilities: List[str] = Field(
        default_factory=list, description="Rich capabilities"
    )
    extra_config: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class UsageStatistics(BaseModel):
    """Real-time token usage and cost tracking for agent operations.

    Attributes:
        input_tokens: Number of tokens sent to the LLM.
        output_tokens: Number of tokens received from the LLM.
        total_tokens: Sum of input and output tokens.
        estimated_cost_usd: Estimated monetary cost in USD based on model pricing.

    """

    input_tokens: int = 0

    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class CostModel(BaseModel):
    """Pricing configuration for LLM cost estimation.

    Attributes:
        input_token_price: Cost per input token in USD.
        output_token_price: Cost per output token in USD.

    """

    input_token_price: float = 0.00000015  # Default for Sonnet 3.5 ($0.15 / 1M)

    output_token_price: float = 0.0000006  # Default for Sonnet 3.5 ($0.60 / 1M)


class ExecutionStep(BaseModel):
    """A single execution unit within a GraphPlan.

    Attributes:
        node_id: The ID of the functional node to execute (e.g., 'researcher').
        input_data: Input data passed to the step's function.
        is_parallel: Whether this step initiates a parallel batch of tasks.
        status: Current execution status ('pending', 'completed', 'failed').
        timeout: Execution timeout in seconds.
        depends_on: List of node IDs that must be completed before this step executes.

    """

    node_id: str = Field(description="The ID of the functional step to execute")

    input_data: Optional[Any] = Field(None, description="Input data passed to the step")
    is_parallel: bool = Field(
        False, description="Whether this step starts a parallel batch"
    )
    status: str = Field("pending", description="Current execution status")
    timeout: float = Field(120.0, description="Per-node timeout in seconds")
    depends_on: List[str] = Field(
        default_factory=list, description="Node IDs that must complete before this step"
    )


class ParallelBatch(BaseModel):
    """A collection of execution steps intended to be run concurrently.

    Attributes:
        tasks: The list of execution steps in the batch.

    """

    tasks: List[ExecutionStep] = Field(default_factory=list)


class GraphPlan(BaseModel):
    """A sequenced collection of execution steps forming a plan for the agent graph.

    Attributes:
        steps: Ordered list of ExecutionSteps or ParallelBatches to execute.
        metadata: Additional metadata about the plan (e.g., strategy used).

    """

    steps: List[ExecutionStep] = Field(default_factory=list)

    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_acp_plan_entries(self) -> list[dict[str, str]]:
        """Convert execution steps to ACP-compatible PlanEntry dicts.

        Maps each :class:`ExecutionStep` to a dict with ``content``,
        ``status``, and ``priority`` keys matching the ACP ``PlanEntry``
        schema.

        Returns:
            A list of plain dicts suitable for constructing
            ``acp.schema.PlanEntry`` objects.

        """
        _STATUS_MAP = {
            "pending": "pending",
            "in_progress": "in_progress",
            "completed": "completed",
            "failed": "pending",
        }
        entries: list[dict[str, str]] = []
        for step in self.steps:
            entries.append(
                {
                    "content": (
                        f"{step.node_id}: {step.input_data}"
                        if step.input_data
                        else step.node_id
                    ),
                    "status": _STATUS_MAP.get(step.status, "pending"),
                    "priority": "high" if not step.is_parallel else "medium",
                }
            )
        return entries


class MCPServerHealth(BaseModel):
    """Circuit breaker state for an MCP server to prevent cascading failures.

    Attributes:
        server_name: Name of the monitored MCP server.
        failures: Current count of consecutive failed requests.
        last_failure: Timestamp of the last recorded failure.
        state: Circuit state ('closed', 'open', 'half-open').
        cooldown_seconds: Time to wait before attempting a retry when open.
        max_failures: Threshold of failures before the circuit opens.

    """

    server_name: str = ""

    failures: int = 0
    last_failure: float = 0.0
    state: str = "closed"  # closed (healthy), open (unavailable), half-open (testing)
    cooldown_seconds: float = 60.0
    max_failures: int = 3

    def record_failure(self) -> None:
        """Increment failure count and open the circuit if threshold is reached."""
        import time

        self.failures += 1
        self.last_failure = time.time()
        if self.failures >= self.max_failures:
            self.state = "open"

    def record_success(self) -> None:
        """Reset failure count and close the circuit."""
        self.failures = 0
        self.state = "closed"

    def is_available(self) -> bool:
        """Check if the server is available for requests based on circuit state.

        Returns:
            True if the server can be called, False otherwise.

        """
        import time

        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure > self.cooldown_seconds:
                self.state = "half-open"
                return True  # Allow one test request
            return False
        # half-open: allow one request
        return True


# --- SDD (Spec-Driven Development) Domain Models ---


class ProjectConstitution(BaseModel):
    """Governing principles and technical standards for a project.

    Attributes:
        vision: High-level purpose of the project.
        mission: Strategic goals and primary value proposition.
        core_principles: List of guiding principles for development.
        tech_stack: Pre-approved languages, frameworks, and patterns.
        quality_gates: Mandatory checks (e.g., coverage, linting).
        metadata: Additional project-level data.
    """

    vision: str = ""
    mission: str = ""
    core_principles: List[str] = Field(default_factory=list)
    tech_stack: Dict[str, Any] = Field(default_factory=dict)
    quality_gates: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UserScenario(BaseModel):
    """A single user story or scenario within a feature specification.

    Attributes:
        id: Unique identifier (e.g., 'US1').
        goal: What the user wants to achieve.
        actor: Who is performing the action.
        context: The situation or prerequisites.
        result: The expected outcome.
        priority: Business priority (P1, P2, P3).
    """

    id: str
    goal: str
    actor: str = "User"
    context: str = ""
    result: str = ""
    priority: str = "P1"


class FeatureSpec(BaseModel):
    """Structured requirement specification for a single feature.

    Attributes:
        name: Short name of the feature.
        description: Natural language summary of the feature.
        user_scenarios: List of user stories.
        functional_requirements: Testable functional statements.
        success_criteria: Measurable metrics and outcomes.
        assumptions: Defaults or context relied upon.
        entities: Core data objects/entities involved.
        metadata: Additional specification metadata.
    """

    name: str
    description: str = ""
    user_scenarios: List[UserScenario] = Field(default_factory=list)
    functional_requirements: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ImplementationPlan(BaseModel):
    """Technical architecture and approach for implementing a feature.

    Attributes:
        technical_context: Research findings and architectural decisions.
        proposed_changes: Files to be created or modified.
        phases: Strategic implementation phases.
        tradeoffs: Decisions made and their consequences.
        risk_assessment: Potential blockers or security concerns.
        feature_id: Reference to the associated FeatureSpec.
    """

    technical_context: str = ""
    proposed_changes: List[str] = Field(default_factory=list)
    phases: List[str] = Field(default_factory=list)
    tradeoffs: Dict[str, str] = Field(default_factory=dict)
    risk_assessment: List[str] = Field(default_factory=list)
    feature_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
