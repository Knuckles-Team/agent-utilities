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
from .codemap import (
    CodemapArtifact,
    CodemapNode,
)
from .graph import (
    ExecutionStep,
    GraphPlan,
    GraphResponse,
    ParallelBatch,
    WideSearchWorkboard,
)
from .mcp import (
    DiscoveredSpecialist,
    MCPAgent,
    MCPAgentRegistryModel,
    MCPConfigModel,
    MCPServerHealth,
    MCPToolInfo,
)
from .model_registry import (
    ModelCostRate,
    ModelDefinition,
    ModelRegistry,
    ModelTier,
)
from .project import (
    ProgressEntry,
    ProgressLog,
    SprintContract,
)
from .prompt import (
    NestedStructure,
    StructuredPrompt,
)
from .scheduling import (
    CronLogEntryModel,
    CronLogModel,
    CronRegistryModel,
    CronTaskModel,
    PeriodicTask,
)
from .sdd import (
    DesignDocument,
    ExtensionStrategy,
    ImplementationPlan,
    KGAnalysis,
    NearestConcept,
    NewConceptProposal,
    ProjectConstitution,
    RiskAssessment,
    Spec,
    Task,
    Tasks,
    TaskStatus,
    UserStory,
)
from .usage import (
    CostModel,
    ExecutionBudget,
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
    "WideSearchWorkboard",
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
    "ExecutionBudget",
    "StructuredPrompt",
    "NestedStructure",
    "CodemapArtifact",
    "CodemapNode",
    "ModelCostRate",
    "ModelDefinition",
    "ModelRegistry",
    "ModelTier",
    "DesignDocument",
    "KGAnalysis",
    "NearestConcept",
    "NewConceptProposal",
    "ExtensionStrategy",
    "RiskAssessment",
]
