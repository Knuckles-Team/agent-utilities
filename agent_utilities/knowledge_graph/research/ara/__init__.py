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
from .compiler import ARACompiler, CompileReport
from .exploration import ExplorationGraphBuilder, ResearchTrajectory
from .live_manager import LiveResearchManager, ResearchEvent
from .reasoning_driver import InferenceHarvest, OntologyReasoningDriver
from .seal import ARASeal, SealReport, SealViolation
from .service import ACTIONS, ARAService

__all__ = [
    "ACTIONS",
    "ARACompiler",
    "ARAService",
    "ARASeal",
    "Claim",
    "CodeSpec",
    "CompileReport",
    "Evidence",
    "ExplorationGraphBuilder",
    "ExplorationKind",
    "ExplorationNode",
    "InferenceHarvest",
    "LiveResearchManager",
    "OntologyReasoningDriver",
    "ResearchArtifact",
    "ResearchEvent",
    "ResearchTrajectory",
    "SealReport",
    "SealViolation",
]
