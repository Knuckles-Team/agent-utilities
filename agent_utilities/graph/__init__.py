"""Graph orchestration package - Hierarchical State Machine (HSM) Implementation.

This package provides a modular entrypoint for graph orchestration
"""

from .state import (
    # State
    GraphState,
    GraphDeps,
)
from .graph_models import (
    # Models
    DomainChoice,
    MultiDomainChoice,
    ValidationResult,
)
from .hsm import (
    # HSM Infrastructure,
    on_enter_specialist,
    on_exit_specialist,
    register_on_enter_hook,
    register_on_exit_hook,
    StateInvariantError,
    assert_state_valid,
    NodeResult,
    check_specialist_preconditions,
    run_orthogonal_regions,
    static_route_query,
)
from .config_helpers import (
    # Config helpers
    load_mcp_config,
    load_node_agents_registry,
    save_mcp_config,
    emit_graph_event,
    load_specialized_prompts,
    DEFAULT_GRAPH_TIMEOUT,
)
from .builder import (
    # Builder
    NODE_SKILL_MAP,
    create_graph_agent,
    create_master_graph,
    initialize_graph_from_workspace,
    build_tag_env_map,
)
from .executor import get_step_descriptions, agent_matches_node_id

from .runner import (
    # Runner
    run_graph,
    run_graph_stream,
    validate_graph,
)

from .unified import (
    # Unified execution (Protocol-agnositic entry points)
    execute_graph,
    execute_graph_stream,
    GraphEventHandler,
    SSEEventHandler,
    ACPEventHandler,
)

from .mermaid import (
    # Mermaid
    get_graph_mermaid,
)

__all__ = [
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
    "NODE_SKILL_MAP",
    "DEFAULT_GRAPH_TIMEOUT",
    # Builder
    "get_step_descriptions",
    "agent_matches_node_id",
    "create_graph_agent",
    "create_master_graph",
    "initialize_graph_from_workspace",
    "build_tag_env_map",
    # Runner
    "run_graph",
    "run_graph_stream",
    "validate_graph",
    # Unified execution
    "execute_graph",
    "execute_graph_stream",
    "GraphEventHandler",
    "SSEEventHandler",
    "ACPEventHandler",
    # Mermaid
    "get_graph_mermaid",
]
