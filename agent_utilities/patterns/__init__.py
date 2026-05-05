#!/usr/bin/python
"""Agentic Design Patterns — Native Implementations.

This package provides KG/OWL-native implementations of design patterns
identified as gaps in the agent-utilities architecture during the
comparative review against *Agentic Design Patterns* (Antonio Gulli).

Modules
-------
prompt_chain : AU-018
    Declarative multi-step prompt pipelines with branching.
prioritization : AU-021
    Multi-factor task prioritization with dependency tracking.
exploration : AU-022
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
    # AU-018
    "ChainResult",
    "PromptChain",
    "PromptChainExecutor",
    "PromptChainStep",
    "StepResult",
    # AU-021
    "PrioritizationEngine",
    "PrioritizedTask",
    "PriorityScore",
    # AU-022
    "Discovery",
    "Experiment",
    "ExplorationEngine",
    "Hypothesis",
    "KnowledgeGap",
    "ReviewBundle",
    "ReviewScore",
]
