"""Memory subsystem for the Knowledge Graph.

CONCEPT:KG-2.1 — Tiered Memory & Context
CONCEPT:KG-2.1 — Observational Memory Bridge

This package contains:
- Consolidation engine (KG-2.4) — Episode→Preference, Decision→Principle rules
- Memory materializer (KG-2.10) — KG→Markdown bidirectional sync
- Observer (KG-2.10) — LLM-powered transcript→observation extraction
- Reflector (KG-2.10) — Observation→reflection condensation
- Startup context builder (KG-2.10) — Budgeted payload for agent hooks
- Semantic compactor (KG-2.20) — Trace compaction to prevent graph explosion
"""

from .consolidation import ConsolidationEngine, ConsolidationProposal
from .memory_compaction import SemanticCompactor
from .memory_materializer import (
    MemoryMaterializer,
    ingest_memory_edits,
    materialize_memory,
    memory_dir,
)
from .observer import observe_from_file, observe_transcript
from .reflector import run_reflector
from .startup_context import (
    StartupContextBuilder,
    StartupPayload,
    build_startup_payload,
)
from .unified_memory import MemoryLifecycleManager

__all__ = [
    # Memory Lifecycle Manager (KG-2.1)
    "MemoryLifecycleManager",
    # Consolidation (KG-2.4)
    "ConsolidationEngine",
    "ConsolidationProposal",
    # Memory Materializer (KG-2.10)
    "MemoryMaterializer",
    "materialize_memory",
    "ingest_memory_edits",
    "memory_dir",
    # Observer (KG-2.10)
    "observe_transcript",
    "observe_from_file",
    # Reflector (KG-2.10)
    "run_reflector",
    # Startup Context (KG-2.10)
    "StartupContextBuilder",
    "StartupPayload",
    "build_startup_payload",
    # Semantic Compactor (KG-2.20)
    "SemanticCompactor",
]
