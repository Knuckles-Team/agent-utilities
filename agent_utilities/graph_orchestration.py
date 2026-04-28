#!/usr/bin/python
"""Compatibility shim for the legacy ``graph_orchestration`` module.

The core graph orchestration engine now lives in the
``agent_utilities.graph`` subpackage. This module re-exports the public
API via ``from .graph import *`` and additionally provides backwards
compatible aliases for names that earlier agent-utilities releases
shipped (``ProjectState``, ``ProjectDeps``) plus minimal ``BaseNode``
stubs (``_RouterNodeBase``, ``BaseProjectInitializerNode``, ``RouterNode``,
``DomainNode``) so that downstream agents which still import the old
symbols continue to load. The stubs are intentionally no-op: the real
orchestration now uses the step-based ``pydantic_graph.beta.GraphBuilder``
topology in :mod:`agent_utilities.graph.builder`.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from pydantic_graph import BaseNode, End, GraphRunContext

from .graph import *  # noqa: F401, F403
from .graph.state import GraphDeps, GraphState

# ---------------------------------------------------------------------------
# Backwards-compatibility aliases for the old class names.
# ---------------------------------------------------------------------------

ProjectState = GraphState
ProjectDeps = GraphDeps


@dataclass
class _RouterNodeBase(BaseNode[GraphState, GraphDeps, Any]):
    """Legacy compat base node.

    Older agent-utilities releases exposed a ``_RouterNodeBase`` hierarchy
    that subclasses of :class:`pydantic_graph.BaseNode` inherited from.
    Concrete subclasses override :meth:`run` to perform real work; the
    stub here simply returns ``End(state.results)`` so that instances can
    be constructed without hitting ``BaseNode``'s abstract-method check.
    """

    async def run(
        self, ctx: GraphRunContext[GraphState, GraphDeps]
    ) -> End[dict[str, Any]]:
        return End(getattr(ctx.state, "results", {}) or {})


@dataclass
class BaseProjectInitializerNode(_RouterNodeBase):
    """Legacy compat initializer node.

    Subclasses typically call ``await super().run(ctx)`` for its side
    effects (loading persisted state) and then return the next node.
    """

    async def run(
        self, ctx: GraphRunContext[GraphState, GraphDeps]
    ) -> End[dict[str, Any]]:
        state = ctx.state
        loader = getattr(state, "load_from_disk", None)
        if callable(loader):
            with contextlib.suppress(Exception):
                # Loading from disk is best-effort in the compat shim.
                loader()
        return End(getattr(state, "results", {}) or {})


@dataclass
class RouterNode(_RouterNodeBase):
    """Legacy compat router node (terminal stub)."""


@dataclass
class DomainNode(_RouterNodeBase):
    """Legacy compat domain node (terminal stub)."""


__all__ = [
    "ProjectState",
    "ProjectDeps",
    "_RouterNodeBase",
    "BaseProjectInitializerNode",
    "RouterNode",
    "DomainNode",
]
