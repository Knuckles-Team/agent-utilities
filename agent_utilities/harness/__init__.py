"""Agentic Harness Engineering (AHE) Package.

CONCEPT:AU-012 — Agentic Harness Engineering

This package implements the AHE closed-loop evolution framework for
self-improving AI agent harnesses. It provides:

- **Component Manifest** (``manifest.py``): Pydantic models for tracking
  versioned edits to harness components (prompts, tools, middleware, skills,
  sub-agents, memory) with falsifiable predictions.
- **Component Registry** (``component_registry.py``): File-to-component
  mapping with version tracking and git-based rollback.
- **Evidence Corpus** (``evidence_corpus.py``): Layered evidence system
  with progressive disclosure (overview -> per-task -> processed -> raw).
- **Trace Backend** (``trace_backend.py``): Agnostic trace ingestion
  abstraction supporting Langfuse and OTel/Logfire backends.
- **Trace Distiller** (``trace_distiller.py``): Automated distillation
  pipeline transforming raw traces into actionable evidence.
- **Evolve Agent** (``evolve_agent.py``): The AHE evolution agent that
  reads evidence, proposes edits, and records predictions.
- **Manifest Verifier** (``verifier.py``): Verifies evolution predictions
  against actual outcomes and auto-reverts regressions.
- **Constraint Engine** (``constraint_engine.py``): Hierarchical constraint
  enforcement with automatic escalation.

Architecture (Hybrid Model):
    - **Epistemic State** (what the agent knows): Knowledge Graph
    - **Normative State** (what the agent is allowed to do): Filesystem + git
    - **Causal Boundary** (what caused improvement): Change Manifests

References:
    - AHE whitepaper: ``prompts/conversations/new/AHE.pdf``
    - Implementation plan: ``implementation_plan.md``
    - Architecture docs: ``docs/AHE_ARCHITECTURE.md``
"""

from .component_registry import HarnessComponentRegistry
from .constraint_engine import ConstraintEngine, ConstraintLevel, HierarchicalConstraint
from .evidence_corpus import (
    EvidenceCorpus,
    EvidenceEntry,
    EvidenceLayer,
    FailureCluster,
)
from .evolve_agent import EvolveAgent
from .manifest import ChangeManifest, ComponentEdit, ComponentType, VerificationResult
from .trace_backend import (
    FileTraceBackend,
    LangfuseTraceBackend,
    OTelTraceBackend,
    TraceBackend,
    create_trace_backend,
)
from .trace_distiller import DistillationConfig, TraceDistiller
from .verifier import ManifestVerifier

__all__ = [
    # Manifest (Phase 1)
    "ChangeManifest",
    "ComponentEdit",
    "ComponentType",
    "VerificationResult",
    # Evidence Corpus (Phase 2)
    "EvidenceCorpus",
    "EvidenceEntry",
    "EvidenceLayer",
    "FailureCluster",
    # Component Registry (Phase 1)
    "HarnessComponentRegistry",
    # Trace Backend (Phase 2)
    "TraceBackend",
    "LangfuseTraceBackend",
    "OTelTraceBackend",
    "FileTraceBackend",
    "create_trace_backend",
    # Trace Distiller (Phase 2)
    "TraceDistiller",
    "DistillationConfig",
    # Evolve Agent (Phase 3)
    "EvolveAgent",
    # Verifier (Phase 3)
    "ManifestVerifier",
    # Constraint Engine (Phase 4)
    "ConstraintEngine",
    "ConstraintLevel",
    "HierarchicalConstraint",
]
