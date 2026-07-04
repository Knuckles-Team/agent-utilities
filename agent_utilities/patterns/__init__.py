#!/usr/bin/python
"""Agentic Design Patterns — Native Implementations.

This package provides KG/OWL-native implementations of design patterns
identified as gaps in the agent-utilities architecture during the
comparative review against *Agentic Design Patterns* (Antonio Gulli).

Modules
-------
prompt_chain : CONCEPT:AU-ORCH.planning.recursion-nesting-depth
    Declarative multi-step prompt pipelines with branching.
prioritization : CONCEPT:AU-ORCH.planning.recursion-nesting-depth
    Multi-factor task prioritization with dependency tracking.
exploration : CONCEPT:AU-AHE.harness.evolutionary-aggregation
    Autonomous exploration and discovery loop.
"""

from agent_utilities.patterns.exploration import (
    Discovery,
    Experiment,
    ExplorationEngine,
    Hypothesis,
    KnowledgeGap,
    ReviewBundle,
    ReviewScore,
)
from agent_utilities.patterns.prioritization import (
    PrioritizationEngine,
    PrioritizedTask,
    PriorityScore,
)
from agent_utilities.patterns.prompt_chain import (
    ChainResult,
    PromptChain,
    PromptChainExecutor,
    PromptChainStep,
    StepResult,
)

__all__ = [
    # CONCEPT:AU-ORCH.planning.recursion-nesting-depth
    "ChainResult",
    "PromptChain",
    "PromptChainExecutor",
    "PromptChainStep",
    "StepResult",
    # CONCEPT:AU-ORCH.planning.recursion-nesting-depth
    "PrioritizationEngine",
    "PrioritizedTask",
    "PriorityScore",
    # CONCEPT:AU-AHE.harness.evolutionary-aggregation
    "Discovery",
    "Experiment",
    "ExplorationEngine",
    "Hypothesis",
    "KnowledgeGap",
    "ReviewBundle",
    "ReviewScore",
]
