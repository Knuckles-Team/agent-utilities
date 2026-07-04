"""Learned / adaptive routing policies (CONCEPT:AU-ORCH.adapter.hot-cache-invalidation).

Canonical location for the data-driven routing policies (rule-based, trace-learned,
cost-aware, topological). The implementation currently lives in the strangled
module ``graph/adaptive_agent_router.py`` and is re-exported here so the routing
strategy package is the single import surface; a future pass moves the bodies in.
"""

from __future__ import annotations

from ...adaptive_agent_router import (
    CostAwareRouter,
    ExecutionTrace,
    OntologicalFallbackChain,
    RoutingCandidate,
    RoutingDecision,
    RoutingPolicy,
    RoutingPrimitive,
    RuleBasedPolicy,
    TopologicalRoutingPolicy,
    TraceLearnedPolicy,
    extract_task_features,
)

__all__ = [
    "RoutingPolicy",
    "RuleBasedPolicy",
    "TraceLearnedPolicy",
    "CostAwareRouter",
    "TopologicalRoutingPolicy",
    "OntologicalFallbackChain",
    "RoutingPrimitive",
    "RoutingCandidate",
    "RoutingDecision",
    "ExecutionTrace",
    "extract_task_features",
]
