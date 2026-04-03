import os
from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# Unified Execution Models defined below at line 220+

__version__ = "0.2.39"


class PeriodicTask(BaseModel):
    id: str
    name: str
    interval_minutes: int
    prompt: str
    last_run: datetime = Field(default_factory=datetime.now)
    active: bool = True


from dataclasses import dataclass, field
from pathlib import Path
import asyncio


@dataclass
class AgentDeps:
    """Standard dependencies for agent tools."""

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
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(BaseModel):
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
    name: str
    tasks: List[Task] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING


class TaskList(BaseModel):
    phases: List[TaskPhase] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    current_phase_index: int = 0
    version: str = "1.0.0"


class ProgressEntry(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat() + "Z")
    message: str
    git_commit: Optional[str] = None
    logs: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProgressLog(BaseModel):
    entries: List[ProgressEntry] = Field(default_factory=list)


class SprintContract(BaseModel):
    goals: List[str] = Field(default_factory=list)
    definition_of_done: List[str] = Field(default_factory=list)
    test_criteria: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IdentityModel(BaseModel):
    name: str = "AI Agent"
    role: str = ""
    emoji: str = "🤖"
    vibe: str = ""
    system_prompt: str = ""


class GraphResponse(BaseModel):
    """Standardized response from an agent graph execution."""

    status: str = "pending"
    results: Dict[str, Any] = Field(default_factory=dict)
    mermaid: Optional[str] = None
    usage: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UserModel(BaseModel):
    name: str = "User"
    emoji: str = "👤"


class A2APeerModel(BaseModel):
    name: str
    url: str
    description: str = ""
    capabilities: str = ""
    auth: str = "none"
    notes: str = ""


class A2ARegistryModel(BaseModel):
    peers: List[A2APeerModel] = Field(default_factory=list)


class MemoryEntryModel(BaseModel):
    timestamp: str
    text: str


class MemoryModel(BaseModel):
    entries: List[MemoryEntryModel] = Field(default_factory=list)


class CronTaskModel(BaseModel):
    id: str
    name: str
    interval_minutes: int
    prompt: str
    last_run: str = "—"
    next_approx: str = "—"


class CronRegistryModel(BaseModel):
    tasks: List[CronTaskModel] = Field(default_factory=list)


class CronLogEntryModel(BaseModel):
    timestamp: str
    task_id: str
    task_name: str = ""
    status: str = "success"
    message: str = ""
    chat_id: Optional[str] = None


class CronLogModel(BaseModel):
    entries: List[CronLogEntryModel] = Field(default_factory=list)


class MCPConfigModel(BaseModel):
    mcpServers: Dict[str, Any] = Field(default_factory=dict)


class MCPAgent(BaseModel):
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
    name: str = Field(description="Full tool name")
    description: str = Field(description="Tool description")
    tag: Optional[str] = Field(None, description="Primary tool tag for partitioning")
    mcp_server: str = Field(description="Source MCP server")
    all_tags: List[str] = Field(
        default_factory=list, description="All tags associated with the tool"
    )


class MCPAgentRegistryModel(BaseModel):
    agents: List[MCPAgent] = Field(default_factory=list)
    tools: List[MCPToolInfo] = Field(default_factory=list)


class UsageStatistics(BaseModel):
    """Real-time token usage and cost tracking."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class CostModel(BaseModel):
    """Configuration for cost estimation."""

    input_token_price: float = 0.00000015  # Default for Sonnet 3.5 ($0.15 / 1M)
    output_token_price: float = 0.0000006  # Default for Sonnet 3.5 ($0.60 / 1M)


class ExecutionStep(BaseModel):
    """A single execution unit in a graph plan."""

    node_id: str = Field(description="The ID of the functional step to execute")
    input_data: Optional[Any] = Field(None, description="Input data passed to the step")
    is_parallel: bool = Field(
        False, description="Whether this step starts a parallel batch"
    )
    status: str = Field("pending", description="Current execution status")


class ParallelBatch(BaseModel):
    """Wrapper for a batch of parallel tasks to avoid character-mapping in pydantic-graph."""

    tasks: List[ExecutionStep] = Field(default_factory=list)


class GraphPlan(BaseModel):
    """A collection of sequential and parallel execution steps."""

    steps: List[ExecutionStep] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
