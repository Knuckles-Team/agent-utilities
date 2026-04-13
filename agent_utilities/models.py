#!/usr/bin/python
# coding: utf-8
"""
Shared Data Models Module

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

__version__ = "0.2.39"


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
    """Execution status of a task."""
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
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskPhase(BaseModel):
    """A logical grouping of tasks representing a specific phase of a project.

    Attributes:
        name: Name of the phase (e.g., 'Research', 'Implementation').
        tasks: List of Tasks belonging to this phase.
        status: Overall status of the phase.
    """
    name: str
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
    """Basic profile of the user interacting with the agent."""
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
    """A registry of all discovered A2A peer agents."""
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
    """Collection of memory entries for an agent."""
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
    """A registry of all scheduled cron tasks."""
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
    """A collection of cron execution logs."""
    entries: List[CronLogEntryModel] = Field(default_factory=list)


class MCPConfigModel(BaseModel):
    """Schema for the mcp_config.json file.

    Attributes:
        mcpServers: Dictionary mapping server names to their configuration.
    """
    mcpServers: Dict[str, Any] = Field(default_factory=dict)


class MCPAgent(BaseModel):
    """Metadata for a specialized agent derived from an MCP server.

    Attributes:
        name: Display name of the agent.
        description: Detailed description of the agent's specialization.
        system_prompt: The synthesized system prompt used to guide the agent.
        tools: List of tool names exposed by this agent.
        mcp_server: The source MCP server that provides these tools.
        tag: Metadata tag used for routing and partitioning.
        is_custom: Whether the agent was manually customized.
    """
    name: str = Field(description="Display name of the agent")
    description: str = Field(description="Specialized agent description")
    system_prompt: str = Field(description="Synthesized system prompt")
    tools: List[str] = Field(default_factory=list, description="List of tool names")
    mcp_server: str = Field(description="Source MCP server name from config")
    tag: Optional[str] = Field(
        None, description="Metadata tag for sorting/customization"
    )
    is_custom: bool = Field(False, description="True if manually edited by user")


class MCPToolInfo(BaseModel):
    """Detailed metadata for a tool provided by an MCP server.

    Attributes:
        name: Full name of the tool.
        description: Description of what the tool does.
        tag: Primary tag used to assign this tool to a specific specialist agent.
        mcp_server: The source MCP server providing the tool.
        all_tags: All tags associated with the tool for flexible routing.
    """
    name: str = Field(description="Full tool name")
    description: str = Field(description="Tool description")
    tag: Optional[str] = Field(None, description="Primary tool tag for partitioning")
    mcp_server: str = Field(description="Source MCP server")
    all_tags: List[str] = Field(
        default_factory=list, description="All tags associated with the tool"
    )


class MCPAgentRegistryModel(BaseModel):
    """The complete registry of MCP agents and tools discovered from config.

    Attributes:
        agents: List of all specialized agents derived from the MCP servers.
        tools: Comprehensive list of all available MCP tools.
    """
    agents: List[MCPAgent] = Field(default_factory=list)
    tools: List[MCPToolInfo] = Field(default_factory=list)


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
    """Wrapper for a group of ExecutionSteps that can be run in parallel.

    This model prevents character-mapping issues and simplifies the handling 
    of fan-out execution patterns in pydantic-graph.

    Attributes:
        tasks: List of steps to execute concurrently.
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
