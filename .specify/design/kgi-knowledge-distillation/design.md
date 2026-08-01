# Design Document: Structured knowledge distillation as pure Pydantic models over the existing LLM/embedding stack, not a ported dependency

> `agent_utilities/knowledge_graph/distillation/__init__.py`,
> `distillation_engine.py`, `deduplicator.py`, `lsh_index.py`.

CONCEPT:AU-KG.ingest.knowledge-distillation

## Decision — reimplement Blockify's architecture natively, don't depend on it

`distillation/__init__.py:1-14`, `distillation_engine.py:1-13`.

**The design chosen**: an IdeaBlock-compatible structured-knowledge pipeline
— ingestion, embedding generation, semantic deduplication (LSH + cosine
similarity), and iterative LLM-driven merging — natively integrated with the
KG, OWL ontology, and tiered memory systems. `DistillationEngine` is
explicitly the "High-Level Knowledge Distillation Orchestrator" tying these
stages into one cohesive pipeline.

**The rejected alternative, named directly in the module docstring**: taking
a dependency on the source architecture this is "derived from" (Blockify
Agentic Data Optimization) rather than reimplementing its ideas. Instead the
engine is "implemented as pure Pydantic models using our existing LLM and
embedding infrastructure" — no external Blockify package, no separate
LLM/embedding client, no separate persistence layer. The IdeaBlock shape and
LSH+cosine dedup strategy are adopted; the implementation and every I/O
boundary (LLM calls, embeddings, graph writes) go through the SAME
infrastructure every other KG subsystem uses.

**Why this matters as a decision, not just an implementation detail**: taking
Blockify as a dependency would mean a second, parallel LLM/embedding/config
surface living inside the KG stack, with its own versioning and its own
failure modes independent of the rest of the ingestion pipeline. Reimplementing
natively means distillation inherits the same observability, config,
retry/backoff, and provider-routing behavior as every other LLM-backed KG
subsystem, at the cost of needing to track upstream Blockify improvements
manually rather than via a dependency bump.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/distillation/*`
  (self-contained package: `__init__.py`, `distillation_engine.py`,
  `deduplicator.py`, `lsh_index.py`).
- **Backward Compatible**: Yes — an additive subsystem; nothing outside
  `distillation/` depends on its internals.
- **Breaking Changes**: None.
- **Known weak point**: because the architecture is reimplemented rather than
  imported, correctness/behavior parity with upstream Blockify is not
  mechanically verified — a divergence (intentional or not) between this
  implementation and the reference architecture it was derived from would
  only surface as an observed quality regression, not a version mismatch.
