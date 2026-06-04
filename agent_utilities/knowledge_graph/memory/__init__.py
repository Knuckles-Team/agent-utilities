"""Memory subsystem for the Knowledge Graph.

CONCEPT:KG-2.1 — Tiered Memory & Context
CONCEPT:KG-2.1 — Observational Memory Bridge

This package contains:
- Synthesis engine (KG-2.4) — Episode→Preference, Decision→Principle rules
- Memory materializer (KG-2.7) — KG→Markdown bidirectional sync
- Observer (KG-2.7) — LLM-powered transcript→observation extraction
- Reflector (KG-2.7) — Observation→reflection condensation
- Startup context builder (KG-2.7) — Budgeted payload for agent hooks
- Semantic compactor (KG-2.7) — Trace compaction to prevent graph explosion
"""

from .agent_context import (
    AgentContextManager,
    CompactedResult,
    CompactionStrategy,
    ContextCompactor,
    ContextOperator,
    ElasticContextManager,
    MemoryEntry,
    MemoryTimescale,
    PreemptiveCacheEngine,
    SemanticCompactor,
    TimescaleMemoryStore,
    compress_to_memento,
    estimate_message_tokens,
    estimate_tokens,
    get_recent_mementos,
    prune_context_by_semantic_distance,
)
from .memory_engine import (
    EvolvingMemoryAPI,
    MemoryEngine,
    MemoryMaterializer,
    StartupContextBuilder,
    StartupPayload,
    build_startup_payload,
    ingest_memory_edits,
    materialize_memory,
    memory_dir,
)

# Also expose observer routines if they exist in optimization engine
from .observer import observe_from_file, observe_transcript
from .optimization_engine import (
    MEMORY_HALF_LIVES,
    AutoSimilarityLinker,
    CKAResult,
    DecisionToPrincipleRule,
    EmbeddingHealthReport,
    EpisodeToPreferenceRule,
    EvaluationCapture,
    FusionResult,
    MemoryOptimizationEngine,
    SynthesisEngine,
    SynthesisProposal,
    TraceToSkillRule,
    apply_ewc_synthesis,
    check_knowledge_drift,
    compute_fisher_diagonal_proxy,
    ebbinghaus_decay,
    run_reflector,
)

__all__ = [
    # Memory Lifecycle
    "MemoryEngine",
    "EvolvingMemoryAPI",
    # Context
    "AgentContextManager",
    "ContextCompactor",
    "ContextOperator",
    "ElasticContextManager",
    "CompactedResult",
    "CompactionStrategy",
    "PreemptiveCacheEngine",
    "TimescaleMemoryStore",
    "MemoryTimescale",
    "MemoryEntry",
    "compress_to_memento",
    "estimate_message_tokens",
    "estimate_tokens",
    "get_recent_mementos",
    "prune_context_by_semantic_distance",
    # Synthesis
    "SynthesisEngine",
    "SynthesisProposal",
    # Materialization
    "MemoryMaterializer",
    "materialize_memory",
    "ingest_memory_edits",
    "memory_dir",
    # Observer
    "observe_transcript",
    "observe_from_file",
    # Reflector
    "run_reflector",
    # Context Builder
    "StartupContextBuilder",
    "StartupPayload",
    "build_startup_payload",
    # Compaction
    "SemanticCompactor",
    # Stability
    "MemoryOptimizationEngine",
    "AutoSimilarityLinker",
    "check_knowledge_drift",
    "apply_ewc_synthesis",
    "compute_fisher_diagonal_proxy",
    "DecisionToPrincipleRule",
    "EpisodeToPreferenceRule",
    "EvaluationCapture",
    "TraceToSkillRule",
    "ebbinghaus_decay",
    "MEMORY_HALF_LIVES",
    "CKAResult",
    "EmbeddingHealthReport",
    "FusionResult",
]
