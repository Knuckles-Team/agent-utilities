"""Unified Models Package.

This package provides a structured and deduplicated set of data models
used across the agent ecosystem, following SDD standards.
"""

from .agent import (
    A2APeerModel,
    A2ARegistryModel,
    AgentDeps,
    IdentityModel,
    UserModel,
)
from .graph import (
    ExecutionStep,
    GraphPlan,
    GraphResponse,
    ParallelBatch,
)
from .mcp import (
    DiscoveredSpecialist,
    MCPAgent,
    MCPAgentRegistryModel,
    MCPConfigModel,
    MCPServerHealth,
    MCPToolInfo,
)
from .project import (
    ProgressEntry,
    ProgressLog,
    SprintContract,
)
from .scheduling import (
    CronLogEntryModel,
    CronLogModel,
    CronRegistryModel,
    CronTaskModel,
    PeriodicTask,
)
from .sdd import (
    ImplementationPlan,
    ProjectConstitution,
    Spec,
    Task,
    Tasks,
    TaskStatus,
    UserStory,
)
from .usage import (
    CostModel,
    UsageStatistics,
)

__all__ = [
    "TaskStatus",
    "UserStory",
    "Spec",
    "Task",
    "Tasks",
    "ImplementationPlan",
    "ProjectConstitution",
    "AgentDeps",
    "IdentityModel",
    "UserModel",
    "A2APeerModel",
    "A2ARegistryModel",
    "MCPConfigModel",
    "MCPAgent",
    "MCPToolInfo",
    "MCPAgentRegistryModel",
    "DiscoveredSpecialist",
    "MCPServerHealth",
    "GraphResponse",
    "ExecutionStep",
    "ParallelBatch",
    "GraphPlan",
    "PeriodicTask",
    "CronTaskModel",
    "CronRegistryModel",
    "CronLogEntryModel",
    "CronLogModel",
    "ProgressEntry",
    "ProgressLog",
    "SprintContract",
    "UsageStatistics",
    "CostModel",
]
