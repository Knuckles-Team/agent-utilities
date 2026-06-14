#!/usr/bin/python
"""Shortcut-resistant search-task synthesis over the epistemic graph.

Distills FORT-Searcher (arXiv:2606.12087) onto the agent-utilities evidence
graph: build a bounded evidence workspace around an answer entity
(:mod:`.evidence_subgraph`, CONCEPT:KG-2.70), detect the four shortcut risks
(:mod:`.shortcut_risks`, CONCEPT:KG-2.71), and formulate + adversarially refine a
verifiable question that forces genuine multi-hop search
(:mod:`.question_formulation`, CONCEPT:KG-2.72). Realized-difficulty diagnostics
(``solving_cost`` / ``answer_hit_time`` / ``prior_shortcut_rate``) live on the
reward spine in :mod:`agent_utilities.graph.training_signals` (CONCEPT:AHE-3.30).
"""

from __future__ import annotations

from .evidence_subgraph import build_evidence_subgraph, synthesize
from .models import (
    EvidenceFact,
    EvidenceGraph,
    RiskFinding,
    RiskReport,
    SearchTask,
)
from .question_formulation import formulate, refine
from .shortcut_risks import (
    diagnose,
    evidence_co_coverage,
    exposed_constants,
    prior_knowledge_binding,
    single_clue_selectivity,
)

__all__ = [
    "EvidenceFact",
    "EvidenceGraph",
    "RiskFinding",
    "RiskReport",
    "SearchTask",
    "build_evidence_subgraph",
    "synthesize",
    "formulate",
    "refine",
    "diagnose",
    "single_clue_selectivity",
    "evidence_co_coverage",
    "exposed_constants",
    "prior_knowledge_binding",
]
