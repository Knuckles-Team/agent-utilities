"""Routing strategies — each owns specific capability rows from the Plan 03
Capability Ledger (R1–R13). Strategies are composed in order by the Router.
"""

from .fallback import match_specialists_in_text, unstructured_fallback_prompt
from .fast_path import FastPathStrategy, is_trivial_query
from .learned import (
    CostAwareRouter,
    RoutingDecision,
    RuleBasedPolicy,
    TopologicalRoutingPolicy,
    TraceLearnedPolicy,
)
from .llm_planner import (
    is_complex_query,
    parse_rlm_plan,
    rlm_plan_instruction,
    subtask_and_widesearch_instructions,
)
from .optimization import (
    filter_by_pheromone,
    format_specialist_step_info,
    optimize_specialists,
    prune_by_telemetry,
)
from .policy import (
    LearnedAgentPolicy,
    PolicyDrivenRouter,
    PolicyStrategy,
    RoutingPolicy,
    SubagentLifecyclePolicy,
    SwarmPresetPolicy,
)
from .query_tier import QueryRouter, QueryTier, QueryType
from .team_reuse import select_reusable_team
from .workflow_context import ShieldedResult, WorkflowContextRouter

__all__ = [
    "FastPathStrategy",
    "is_trivial_query",
    "ShieldedResult",
    "WorkflowContextRouter",
    "RoutingPolicy",
    "SwarmPresetPolicy",
    "LearnedAgentPolicy",
    "SubagentLifecyclePolicy",
    "PolicyDrivenRouter",
    "PolicyStrategy",
    "RuleBasedPolicy",
    "TraceLearnedPolicy",
    "CostAwareRouter",
    "TopologicalRoutingPolicy",
    "RoutingDecision",
    "QueryRouter",
    "QueryTier",
    "QueryType",
    "select_reusable_team",
    "filter_by_pheromone",
    "prune_by_telemetry",
    "format_specialist_step_info",
    "optimize_specialists",
    "subtask_and_widesearch_instructions",
    "is_complex_query",
    "rlm_plan_instruction",
    "parse_rlm_plan",
    "unstructured_fallback_prompt",
    "match_specialists_in_text",
]
