"""Graph orchestration package - Hierarchical State Machine (HSM) Implementation.

This package provides a modular entrypoint for graph orchestration
"""

from ..orchestration.engine import AgentOrchestrationEngine
from .builder import (
    build_tag_env_map,
    # Builder
    create_agent,
    create_graph_agent,
    create_master_graph,
    initialize_graph_from_workspace,
)
from .config_helpers import (
    DEFAULT_GRAPH_TIMEOUT,
    emit_graph_event,
    get_discovery_registry,
    # Config helpers
    load_mcp_config,
    load_node_agents_registry,
    load_specialized_prompts,
    save_mcp_config,
)
from .executor import agent_matches_node_id, get_step_descriptions
from .graph_models import (
    # Models
    DomainChoice,
    MultiDomainChoice,
    ValidationResult,
)
from .horizon_curriculum import (
    # CONCEPT:AHE-3.4 — Horizon-Aware Task Curriculum
    CurriculumStage,
    HorizonCurriculum,
    HorizonStageConfig,
    MacroAction,
    PromotionPolicy,
    SubgoalCheckpoint,
)
from .hsm import (
    NodeResult,
    StateInvariantError,
    assert_state_valid,
    check_specialist_preconditions,
    # HSM Infrastructure,
    on_enter_specialist,
    on_exit_specialist,
    register_on_enter_hook,
    register_on_exit_hook,
    run_orthogonal_regions,
    static_route_query,
)
from .kg_graph_factory import (
    # KG-Driven Graph Materialization (CONCEPT:ORCH-1.4)
    KGGraphResult,
    KGMaterializedStep,
    build_pydantic_graph_from_kg,
)
from .manifest_generators import (
    # CONCEPT:ORCH-1.25 — Manifest Generators
    manifest_for_enterprise,
    manifest_from_department,
    manifest_from_planner,
    manifest_from_preset,
    manifest_from_teamconfig,
    manifest_from_workflow,
)
from .mermaid import (
    # Mermaid
    get_graph_mermaid,
)
from .protocol_agnostic_execution import (
    ACPEventHandler,
    GraphEventHandler,
    SSEEventHandler,
    # Unified execution (Protocol-agnositic entry points)
    execute_graph,
    execute_graph_iter,
    execute_graph_stream,
)
from .reactive import (
    BehaviorDispatcher,
    BudgetGuard,
    BudgetTrippedException,
    # CONCEPT:ORCH-1.28 — Graph-Native Reactive Event Sourcing and OS Guardrails
    EventLedger,
    reactive_behavior,
)
from .reward_decomposition import (
    # CONCEPT:AHE-3.1 — Decomposed Reward Signals
    DecomposedRewardRecord,
    RewardDecomposer,
    StepOutcome,
    StepReward,
    TrajectoryOutcome,
    TrajectoryReward,
)
from .state import (
    GraphDeps,
    # State
    GraphState,
)

__all__ = [
    # CONCEPT:ORCH-1.28 — Graph-Native Reactive Event Sourcing and OS Guardrails
    "EventLedger",
    "BehaviorDispatcher",
    "reactive_behavior",
    "BudgetGuard",
    "BudgetTrippedException",
    # State
    "GraphState",
    "GraphDeps",
    # Models
    "DomainChoice",
    "MultiDomainChoice",
    "ValidationResult",
    # HSM Infrastructure",
    "on_enter_specialist",
    "on_exit_specialist",
    "register_on_enter_hook",
    "register_on_exit_hook",
    "StateInvariantError",
    "assert_state_valid",
    "NodeResult",
    "check_specialist_preconditions",
    "run_orthogonal_regions",
    "static_route_query",
    # Config helpers
    "load_mcp_config",
    "load_node_agents_registry",
    "save_mcp_config",
    "emit_graph_event",
    "load_specialized_prompts",
    "get_discovery_registry",
    "DEFAULT_GRAPH_TIMEOUT",
    # Builder
    "get_step_descriptions",
    "agent_matches_node_id",
    "create_agent",
    "create_graph_agent",
    "create_master_graph",
    "initialize_graph_from_workspace",
    "build_tag_env_map",
    "AgentOrchestrationEngine",
    # Unified execution
    "execute_graph",
    "execute_graph_iter",
    "execute_graph_stream",
    "GraphEventHandler",
    "SSEEventHandler",
    "ACPEventHandler",
    # Mermaid
    "get_graph_mermaid",
    # CONCEPT:AHE-3.4 — Horizon-Aware Task Curriculum
    "HorizonCurriculum",
    "HorizonStageConfig",
    "CurriculumStage",
    "PromotionPolicy",
    "MacroAction",
    "SubgoalCheckpoint",
    # CONCEPT:AHE-3.1 — Decomposed Reward Signals
    "RewardDecomposer",
    "DecomposedRewardRecord",
    "StepReward",
    "TrajectoryReward",
    "StepOutcome",
    "TrajectoryOutcome",
    # CONCEPT:ORCH-1.4 — KG-Driven Graph Materialization
    "build_pydantic_graph_from_kg",
    "KGGraphResult",
    "KGMaterializedStep",
    "manifest_from_planner",
    "manifest_from_teamconfig",
    "manifest_from_workflow",
    "manifest_from_preset",
    "manifest_from_department",
    "manifest_for_enterprise",
]
