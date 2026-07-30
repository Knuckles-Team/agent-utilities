#!/usr/bin/python
from __future__ import annotations

"""Reasoning algorithms as versioned graph topologies.

CONCEPT:AU-ORCH.planning.reasoning-graph-topologies — reasoning algorithms as versioned graph topologies
Ports "Graph Engineering: A
Unified Framework for Language Agent System Design" (arXiv:2505.24354): CoT,
self-consistent CoT, ToT (BFS/DFS), GoT, ReAct, and RAP are not different agent
frameworks — they are different **graph topologies** (a fixed set of node
contracts plus edge/routing functions) executed over ONE shared
:class:`~.state.ReasoningState`. Each topology module in this package publishes a
:class:`~.topology.TopologySpec` (a versioned, content-addressed,
graph-addressable resource — see :class:`agent_utilities.models.knowledge_graph.
ReasoningTopologyVersionNode`) and a ``run_*`` entry point that is
dependency-injected and network-free (generation/scoring/tool callables are
supplied by the caller), so every topology is unit-testable with no LLM and no
network, matching this repo's existing convention
(:mod:`agent_utilities.harness.graph_search_evolution`).

Survey (design note for the concept above — see
``reports/deferred/lane-4.3.md`` for the full account): before this package
existed, none of the six topologies existed as a named, reusable graph-topology
resource. The closest prior art:

* :mod:`agent_utilities.graph.test_time_diversity` (VPO) already provides a
  diverse fan-out width + MMR best-of-k selection — reused by
  :mod:`.cot`'s self-consistent variant rather than reimplemented.
* :class:`agent_utilities.harness.graph_search_evolution.GraphSearchEvolver`
  already runs a REAL UCT-selection + backpropagation MCTS loop — but scoped to
  code/ML-algorithm evolution search (MLEvolve, arXiv:2606.06473), not general
  task reasoning. Its backpropagation is genuinely correct (walks the full
  parent chain), so :mod:`.rap` reuses that exact algorithmic shape, generalized
  off the code-evolution specifics, rather than approximating it.
* :class:`agent_utilities.security.execution_stability_engine.DoomLoopDetector`
  is reused by :mod:`.react` for grounding/termination instead of a new
  loop-detector.
* :class:`agent_utilities.graph.reactive.budget.BudgetGuard` (time/token/cost)
  is reused by :mod:`.budgets` rather than re-implemented; loop-count and
  tool-call-count are the two axes it does not cover, added here.
* :class:`agent_utilities.graph.topology_engine.TopologyEngine` already
  demonstrates the "KG-tracked topology resource with an EMA success-rate
  outcome update" pattern for multi-agent team topologies — :mod:`.topology`
  reuses that exact pattern for reasoning topologies.

None of CoT / self-consistent CoT / ToT / GoT / ReAct / RAP existed as such —
all six are newly implemented here.
"""

from .benchmark import BenchmarkHarness, BenchmarkRecord, TopologyStats
from .budgets import (
    BudgetExhausted,
    Budgets,
    BudgetTracker,
    TerminationProof,
    TerminationReason,
)
from .cot import COT_SPEC, SELF_CONSISTENT_COT_SPEC, run_cot, run_self_consistent_cot
from .got import GOT_SPEC, GotOp, run_got
from .policy import DEFAULT_LADDER, EscalationDecision, EscalationPolicy
from .rap import RAP_SPEC, run_rap
from .react import REACT_SPEC, run_react
from .state import CandidateVote, NodeKind, ReasoningState, ThoughtNode, ToolCallRecord
from .topology import TopologySpec, record_topology_outcome, register_topology
from .tot import TOT_BFS_SPEC, TOT_DFS_SPEC, PruningPolicy, run_tot

__all__ = [
    "BenchmarkHarness",
    "BenchmarkRecord",
    "TopologyStats",
    "Budgets",
    "BudgetExhausted",
    "BudgetTracker",
    "TerminationProof",
    "TerminationReason",
    "COT_SPEC",
    "SELF_CONSISTENT_COT_SPEC",
    "run_cot",
    "run_self_consistent_cot",
    "GOT_SPEC",
    "GotOp",
    "run_got",
    "DEFAULT_LADDER",
    "EscalationDecision",
    "EscalationPolicy",
    "REACT_SPEC",
    "run_react",
    "RAP_SPEC",
    "run_rap",
    "CandidateVote",
    "NodeKind",
    "ReasoningState",
    "ThoughtNode",
    "ToolCallRecord",
    "TopologySpec",
    "register_topology",
    "record_topology_outcome",
    "TOT_BFS_SPEC",
    "TOT_DFS_SPEC",
    "PruningPolicy",
    "run_tot",
]
