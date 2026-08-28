"""Unified routing package (Plan 03 Step 3 — single router entrypoint).

The R1–R13 routing capabilities currently live in the strangled implementation
module ``graph/_router_impl.py`` and are being extracted into composable
``strategies/`` behind the ``Router``/``RoutingStrategy`` framework (R1 fast-path
done). Every historical import path keeps working via re-export, so the
migration is non-breaking:

    from agent_utilities.graph.routing import router_step   # still works
    from agent_utilities.graph.routing import Router        # new framework

Re-export mechanics (WD5-ARCH-02): the six step functions and ``logger``
below are re-exported from the sibling ``graph/_router_impl.py`` monolith
lazily via module ``__getattr__`` (PEP 562), not via a top-level
``from .. import _router_impl``. A top-level import of ``_router_impl``
here forms an eager 3-module cycle (``graph`` -> ``graph.builder`` ->
``graph.routing`` -> ``graph``, since ``_router_impl.py`` lives directly
inside the ``graph`` package) — see ``scripts/check_import_cycles.py``.
Lazy attribute access returns the exact same function objects
``_router_impl`` defines (no wrapping), so callers/tests that reach in via
``agent_utilities.graph._router_impl`` directly are unaffected; it also
means a caller that only needs e.g. ``Router``/``RoutingStrategy`` from
``.strategy`` no longer pays for importing the whole monolith. The
``if TYPE_CHECKING:`` block below exists purely so mypy still sees the
real, precise signatures for these names (it is never executed).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import _router_impl as _impl

    router_step = _impl.router_step
    dispatcher_step = _impl.dispatcher_step
    parallel_batch_processor = _impl.parallel_batch_processor
    expert_executor_step = _impl.expert_executor_step
    dynamic_mcp_routing_step = _impl.dynamic_mcp_routing_step
    mcp_server_step = _impl.mcp_server_step
    # Public module logger used by routing diagnostics.
    logger = _impl.logger

_LAZY_IMPL_NAMES = frozenset(
    {
        "router_step",
        "dispatcher_step",
        "parallel_batch_processor",
        "expert_executor_step",
        "dynamic_mcp_routing_step",
        "mcp_server_step",
        "logger",
    }
)


def __getattr__(name: str):
    """Lazily resolve the ``_router_impl`` re-exports (see module docstring)."""
    if name in _LAZY_IMPL_NAMES:
        from .. import _router_impl as _impl

        return getattr(_impl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
