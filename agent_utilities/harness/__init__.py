"""Harness subpackage for evaluation and backtesting. CONCEPT:AHE-3.4

Exposes OptimisticStateLocker, BranchMergeStateLocker, ToolContract, and ContractValidator.
"""

from .contract_validator import ContractValidator, ToolContract
from .distributed_state_manager import BranchMergeStateLocker, OptimisticStateLocker
from .evolving_memory import EvolvingMemoryStore, MemoryBank, MemoryRecord
from .provenance_gate import ProvenanceCriticGate, ProvenanceVerdict
from .red_team import ATTACK_CATALOG, AttackProbe, RedTeamReport, RedTeamRunner
from .reliability_scorers import (
    BrierSkillScorer,
    CitationQualityScorer,
    DeceptionScorer,
    FaithfulnessScorer,
    RetrievalRecallScorer,
    SafetyAccuracyScorer,
    ToolNecessityScorer,
    TopicCoverageScorer,
    TrapInjectionScorer,
    build_reliability_suite,
)
from .selection_operators import (
    bradley_terry_scores,
    conservative_rating,
    contribution_weighted_vote,
    rank_from_comparisons,
    select_top_k,
)

__all__ = [
    "OptimisticStateLocker",
    "BranchMergeStateLocker",
    "ToolContract",
    "ContractValidator",
    # Graph-native evolving-memory store (CONCEPT:KG-2.1)
    "EvolvingMemoryStore",
    "MemoryBank",
    "MemoryRecord",
    # Provenance-completeness critic gate (CONCEPT:AHE-3.13)
    "ProvenanceCriticGate",
    "ProvenanceVerdict",
    # Unified selection / aggregation operators (CONCEPT:ORCH-1.30)
    "bradley_terry_scores",
    "conservative_rating",
    "contribution_weighted_vote",
    "select_top_k",
    "rank_from_comparisons",
    # Agentic red-team harness (CONCEPT:AHE-3.1)
    "RedTeamRunner",
    "RedTeamReport",
    "AttackProbe",
    "ATTACK_CATALOG",
    # Reliability / guardrail eval scorers (CONCEPT:AHE-3.1)
    "build_reliability_suite",
    "FaithfulnessScorer",
    "SafetyAccuracyScorer",
    "TopicCoverageScorer",
    "ToolNecessityScorer",
    "DeceptionScorer",
    "CitationQualityScorer",
    "BrierSkillScorer",
    "RetrievalRecallScorer",
    "TrapInjectionScorer",
]
