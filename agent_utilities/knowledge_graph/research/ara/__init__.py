#!/usr/bin/python
"""Agent-Native Research Artifacts (ARA) — ontology-driven automated research.

Research as OWL/RDF reasoning over the one ecosystem knowledge-graph: promote any
array of information, reason to extrapolate new relationships/concepts/heuristics
across agents/services/research, and harvest them back as research outputs and the
next Loop iteration's inputs. The keystone is :mod:`reasoning_driver`. (CONCEPT:KG-2.79)
"""

from .artifact import (
    Claim,
    CodeSpec,
    Evidence,
    ExplorationKind,
    ExplorationNode,
    ResearchArtifact,
)
from .reasoning_driver import InferenceHarvest, OntologyReasoningDriver

__all__ = [
    "Claim",
    "CodeSpec",
    "Evidence",
    "ExplorationKind",
    "ExplorationNode",
    "InferenceHarvest",
    "OntologyReasoningDriver",
    "ResearchArtifact",
]
