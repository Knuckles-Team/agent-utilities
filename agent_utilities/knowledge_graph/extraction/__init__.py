"""Document → atomic-triple fact extraction (CONCEPT:KG-2.64).

A *fact* is one graph edge — ``(subject) --[predicate]--> (object)`` — plus a
title, description, verbatim ``evidence_span``, ``confidence``, and ``tags``.
This package turns arbitrary document text into a stream of such facts, dedups
them semantically across rounds and files, and persists them as graph edges
whose properties the engine already carries (confidence / provenance / metadata).

Assimilated from the open-source ``knowledge-graph-extractor`` (hanxiao): the
canonical-entity-forcing prompt that drives graph connectivity, the incremental
streaming JSON parser, and the seed-varied multi-round recall loop. Reuses our
own embedder (``create_embedding_model``) for dedup rather than a second model,
and our engine edge model for persistence — no new infrastructure.
"""

from .fact_extractor import (
    FACT_EXTRACTION_PROMPT,
    FACT_JSON_SCHEMA,
    ExtractedFact,
    FactDeduper,
    extract_facts,
    facts_to_jsonl,
    parse_facts_incremental,
    persist_facts,
)
from .job_manager import (
    EngineStoreAdapter,
    ExtractionJobManager,
    GraphCheckpointStore,
)

__all__ = [
    "FACT_EXTRACTION_PROMPT",
    "FACT_JSON_SCHEMA",
    "EngineStoreAdapter",
    "ExtractedFact",
    "ExtractionJobManager",
    "FactDeduper",
    "GraphCheckpointStore",
    "extract_facts",
    "facts_to_jsonl",
    "parse_facts_incremental",
    "persist_facts",
]
