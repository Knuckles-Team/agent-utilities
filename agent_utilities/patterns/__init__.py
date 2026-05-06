#!/usr/bin/python
"""Agentic Design Patterns — Native Implementations.

This package provides KG/OWL-native implementations of design patterns
identified as gaps in the agent-utilities architecture during the
comparative review against *Agentic Design Patterns* (Antonio Gulli).

Modules
-------
prompt_chain : CONCEPT:ORCH-1.1
    Declarative multi-step prompt pipelines with branching.
prioritization : CONCEPT:ORCH-1.1
    Multi-factor task prioritization with dependency tracking.
exploration : CONCEPT:AHE-3.2
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
    # CONCEPT:ORCH-1.1
    "ChainResult",
    "PromptChain",
    "PromptChainExecutor",
    "PromptChainStep",
    "StepResult",
    # CONCEPT:ORCH-1.1
    "PrioritizationEngine",
    "PrioritizedTask",
    "PriorityScore",
    # CONCEPT:AHE-3.2
    "Discovery",
    "Experiment",
    "ExplorationEngine",
    "Hypothesis",
    "KnowledgeGap",
    "ReviewBundle",
    "ReviewScore",
]
