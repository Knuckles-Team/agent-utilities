"""Harness subpackage for evaluation and backtesting. CONCEPT:AHE-3.4

Exposes OptimisticStateLocker, BranchMergeStateLocker, ToolContract, and ContractValidator.
"""

from .adaptation_benchmark import AdaptationBenchmark, BenchmarkEntry
from .adaptation_speed import AdaptationCurve, CurvePoint, marginal_speed_gain
from .contract_validator import ContractValidator, ToolContract
from .distributed_state_manager import BranchMergeStateLocker, OptimisticStateLocker
from .edit_engine import (
    Edit,
    EditOutcome,
    EditResult,
    apply_edits,
    apply_with_reflection,
    parse_edits,
)
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
from .sai_task import SpecializationTask, Verifier, VerifierResult
from .selection_operators import (
    bradley_terry_scores,
    conservative_rating,
    contribution_weighted_vote,
    rank_from_comparisons,
    select_top_k,
)
from .superhuman_gate import CertificationResult, SuperhumanCertifier
from .world_model_task import WorldModelVerifier, build_world_model_task

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
    # Robust multi-format edit-application engine (CONCEPT:ORCH-1.49)
    "Edit",
    "EditOutcome",
    "EditResult",
    "parse_edits",
    "apply_edits",
    "apply_with_reflection",
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
    # Adaptation-speed metric — SAI primary measure (CONCEPT:AHE-3.27)
    "AdaptationCurve",
    "CurvePoint",
    "marginal_speed_gain",
    # Specialization task + machine-verifiable Verifier contract (CONCEPT:AHE-3.28)
    "SpecializationTask",
    "Verifier",
    "VerifierResult",
    # World-model SAI specialization track (CONCEPT:KG-2.73)
    "WorldModelVerifier",
    "build_world_model_task",
    # Superhuman-certification gate (CONCEPT:SAFE-1.6)
    "SuperhumanCertifier",
    "CertificationResult",
    # Reproducible adaptation-speed benchmark (CONCEPT:SAFE-1.7)
    "AdaptationBenchmark",
    "BenchmarkEntry",
]
