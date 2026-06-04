"""HTN decomposition / refinement — unified planning surface (Plan 03 Step 4).

This module is part of the unified ``graph/planning/`` package.  It re-exports
the Hierarchical Task Network (HTN) decomposition and refinement logic that
currently lives in :mod:`agent_utilities.graph.hierarchical_planner`.

STRANGLER NOTE: rather than physically relocating ~1300 lines (and risking
relative-import breakage across ``_router_impl``, ``steps``, ``parallel_engine``
and ``adaptive_agent_router``), we keep ``hierarchical_planner`` as the live
implementation and expose its public symbols here as the single planning
entrypoint.  The originals are documented as "Strangled by graph/planning/".

The re-exports are the *same* objects as in the original module (identity is
preserved), so there is exactly one source of the planning API.
"""

from __future__ import annotations

from ..hierarchical_planner import (
    AggregationStrategy,
    ConvergenceMonitor,
    EvolutionaryAggregator,
    GroupFitness,
    LATSPlanner,
    RecursionDepthExceeded,
    RecursiveContext,
    architect_step,
    execute_recursive_graph,
    fetch_epistemic_context,
    memory_selection_step,
    planner_step,
    researcher_step,
)

__all__ = [
    "AggregationStrategy",
    "ConvergenceMonitor",
    "EvolutionaryAggregator",
    "GroupFitness",
    "LATSPlanner",
    "RecursionDepthExceeded",
    "RecursiveContext",
    "architect_step",
    "execute_recursive_graph",
    "fetch_epistemic_context",
    "memory_selection_step",
    "planner_step",
    "researcher_step",
]
