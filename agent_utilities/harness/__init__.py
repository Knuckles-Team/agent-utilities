"""Harness subpackage for evaluation and backtesting. CONCEPT:AU-AHE.evaluation.backtest-harness

Exposes OptimisticStateLocker, BranchMergeStateLocker, ToolContract, and ContractValidator.
"""

from typing import Any

from .adaptation_benchmark import AdaptationBenchmark, BenchmarkEntry
from .adaptation_speed import AdaptationCurve, CurvePoint, marginal_speed_gain
from .baseline_overfit_gate import (
    GateVerdict,
    PreRunGate,
    baseline_gate,
    overfit_smoke_gate,
)
from .contract_validator import ContractValidator, ToolContract
from .decentralized_memory import Contribution, DecentralizedMemory, MemoryPool
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
from .fast_slow_controller import FastSlowController, SlowUpdate, Trace
from .forecasting import Forecast, ForecastBoard
from .graph_search_evolution import (
    GlobalCodeMemory,
    GraphSearchEvolver,
    SearchNode,
)
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
from .research_log import BeliefEntry, FailureCase, FailureTriage, ResearchLog
from .sai_task import SpecializationTask, Verifier, VerifierResult
from .selection_operators import (
    bradley_terry_scores,
    conservative_rating,
    contribution_weighted_vote,
    rank_from_comparisons,
    select_top_k,
)
from .self_guided_play import (
    Guide,
    GuideScore,
    PlayReport,
    PlayRound,
    SelfGuidedSelfPlay,
)
from .substrate_trainer import GrpoSample, SubstrateTrainer, TrainingJobSpec
from .world_model_task import WorldModelVerifier, build_world_model_task

# CONCEPT:AU-KG.compute.numpy-scipy-drop's compiled ``epistemic_graph.numeric`` kernel is
# not (yet) published on PyPI — importing it raises ``ImportError`` wherever only the
# published ``epistemic-graph`` client wheel is installed. ``assimilation_benchmark``,
# ``latent_efficiency_benchmark``, and ``superhuman_gate`` transitively require it
# (assimilation_benchmark -> retrieval.generative_recommender -> agent_utilities.numeric).
# Because importing ANY submodule of a package first runs that package's ``__init__``,
# eagerly importing these three here previously forced the numeric kernel onto every
# process's mandatory identity-minting hot path: ``observability.correlation`` imports
# ``agent_utilities.harness.tracing``, and ``security.request_identity.mint_graph_session``
# — now required by every engine-client process (ingest_worker, the messaging daemon,
# kg_server's process authority) — imports ``observability.correlation``. Deferred via a
# PEP 562 module ``__getattr__`` (matching the lazy-heavy-dependency convention already used
# for torch/transformers) so these names stay importable wherever the kernel IS installed,
# without gating the whole package — or the identity path — on it.
_LAZY_SUBMODULE_ATTRS: dict[str, tuple[str, str]] = {
    "BenchmarkResult": (".assimilation_benchmark", "BenchmarkResult"),
    "run_all": (".assimilation_benchmark", "run_all"),
    "to_markdown": (".assimilation_benchmark", "to_markdown"),
    "bench_latent_rollout_memory": (
        ".latent_efficiency_benchmark",
        "bench_latent_rollout_memory",
    ),
    "bench_ontology_prior_retrieval": (
        ".latent_efficiency_benchmark",
        "bench_ontology_prior_retrieval",
    ),
    "run_latent_efficiency_benchmarks": (".latent_efficiency_benchmark", "run_all"),
    "CertificationResult": (".superhuman_gate", "CertificationResult"),
    "SuperhumanCertifier": (".superhuman_gate", "SuperhumanCertifier"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_SUBMODULE_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attr_name = target
    value = getattr(importlib.import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value

__all__ = [
    # Assimilation empirical-parity benchmark suite (CONCEPT:AU-AHE.assimilation.empirical-parity-evidence-assimilation)
    "BenchmarkResult",
    "run_all",
    "to_markdown",
    # Latent-native efficiency benchmark (CONCEPT:AU-AHE.harness.empirical-evidence-that-latent)
    "bench_latent_rollout_memory",
    "bench_ontology_prior_retrieval",
    "run_latent_efficiency_benchmarks",
    "OptimisticStateLocker",
    "BranchMergeStateLocker",
    "ToolContract",
    "ContractValidator",
    # Graph-native evolving-memory store (CONCEPT:AU-KG.memory.tiered-memory-caching)
    "EvolvingMemoryStore",
    "MemoryBank",
    "MemoryRecord",
    # Decentralized per-agent memory + exploit/explore bandit (CONCEPT:AU-KG.memory.ahe-record-this-base/AHE-3.33)
    "DecentralizedMemory",
    "MemoryPool",
    "Contribution",
    # Research-craft disciplines (CONCEPT:AU-AHE.evaluation.predict-before-resolve-calibration/3.35/3.36)
    "Forecast",
    "ForecastBoard",
    "GateVerdict",
    "baseline_gate",
    "overfit_smoke_gate",
    "PreRunGate",
    "FailureCase",
    "FailureTriage",
    "BeliefEntry",
    "ResearchLog",
    # Self-guided self-play (CONCEPT:AU-AHE.harness.when-task-is-scope)
    "SelfGuidedSelfPlay",
    "Guide",
    "GuideScore",
    "PlayRound",
    "PlayReport",
    # Graph-search code evolution (CONCEPT:AU-KG.retrieval.monte-carlo-graph-search) + Fast-Slow controller (ORCH-1.56)
    "GraphSearchEvolver",
    "SearchNode",
    "GlobalCodeMemory",
    "FastSlowController",
    "Trace",
    "SlowUpdate",
    "SubstrateTrainer",
    "TrainingJobSpec",
    "GrpoSample",
    # Provenance-completeness critic gate (CONCEPT:AU-AHE.harness.pre-emit-quality-gate)
    "ProvenanceCriticGate",
    "ProvenanceVerdict",
    # Robust multi-format edit-application engine (CONCEPT:AU-ORCH.execution.robust-multi-format-edit)
    "Edit",
    "EditOutcome",
    "EditResult",
    "parse_edits",
    "apply_edits",
    "apply_with_reflection",
    # Unified selection / aggregation operators (CONCEPT:AU-ORCH.optimization.selection-on-unseen-data)
    "bradley_terry_scores",
    "conservative_rating",
    "contribution_weighted_vote",
    "select_top_k",
    "rank_from_comparisons",
    # Agentic red-team harness (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort)
    "RedTeamRunner",
    "RedTeamReport",
    "AttackProbe",
    "ATTACK_CATALOG",
    # Reliability / guardrail eval scorers (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort)
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
    # Adaptation-speed metric — SAI primary measure (CONCEPT:AU-AHE.harness.per-task-adaptation-speed)
    "AdaptationCurve",
    "CurvePoint",
    "marginal_speed_gain",
    # Specialization task + machine-verifiable Verifier contract (CONCEPT:AU-AHE.harness.sai-task)
    "SpecializationTask",
    "Verifier",
    "VerifierResult",
    # World-model SAI specialization track (CONCEPT:AU-KG.compute.kg-3)
    "WorldModelVerifier",
    "build_world_model_task",
    # Superhuman-certification gate (CONCEPT:AU-OS.safety.superhuman-gate)
    "SuperhumanCertifier",
    "CertificationResult",
    # Reproducible adaptation-speed benchmark (CONCEPT:AU-OS.scaling.safe-2)
    "AdaptationBenchmark",
    "BenchmarkEntry",
]
