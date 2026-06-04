"""Unified routing package (Plan 03 Step 3 — single router entrypoint).

The R1–R13 routing capabilities currently live in the strangled implementation
module ``graph/_router_impl.py`` and are being extracted into composable
``strategies/`` behind the ``Router``/``RoutingStrategy`` framework (R1 fast-path
done). Every historical import path keeps working via re-export, so the
migration is non-breaking:

    from agent_utilities.graph.routing import router_step   # still works
    from agent_utilities.graph.routing import Router        # new framework
"""

from .. import _router_impl as _impl

# Re-export the monolith's public step functions (its __all__) and the module
# globals that callers/tests reference, so this package is a drop-in for the
# former graph/routing.py module.
router_step = _impl.router_step
dispatcher_step = _impl.dispatcher_step
parallel_batch_processor = _impl.parallel_batch_processor
expert_executor_step = _impl.expert_executor_step
dynamic_mcp_routing_step = _impl.dynamic_mcp_routing_step
mcp_server_step = _impl.mcp_server_step

# Back-compat module attributes some callers/tests patch.
Agent = _impl.Agent
logger = _impl.logger

from .enrichers import designate_specialists
from .strategies import (
    FastPathStrategy,
    ShieldedResult,
    WorkflowContextRouter,
    is_trivial_query,
)
from .strategy import Router, RoutingConfig, RoutingStrategy, default_router

__all__ = [
    # historical step functions
    "router_step",
    "dispatcher_step",
    "parallel_batch_processor",
    "expert_executor_step",
    "dynamic_mcp_routing_step",
    "mcp_server_step",
    # new composition framework
    "Router",
    "RoutingConfig",
    "RoutingStrategy",
    "default_router",
    "FastPathStrategy",
    "is_trivial_query",
    "ShieldedResult",
    "WorkflowContextRouter",
    "designate_specialists",
]
