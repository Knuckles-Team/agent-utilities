# Design Document: A `Document`'s full verbatim body is retained in the KG, not just extracted metadata

CONCEPT:AU-KG.ingest.standardized-document-ingestion

> `agent_utilities/knowledge_graph/enrichment/models.py:152-168`.

## Decision — the standardized `Document` contract keeps the full verbatim body text as a `content` field, so a document is faithfully re-materialisable from the KG

`Document` (`models.py:152-168`) is the standardized non-code ingested
artifact (paper, email, BRD, SOW, book, …): `doc_type` drives type-specific
metadata extraction, `metadata` holds the extracted fields, `concept_ids`
link to mentioned concepts — and, separately, `content: str = ""` retains
the full verbatim body, annotated directly at the field: **"Full verbatim
body text — retained so the document is faithfully re-materialisable from
the KG (e.g. distilled back into a skill-graph)."**

**The rejected alternative is storing only the extracted structure** — the
type-specific metadata and concept links, discarding the raw body once
extraction has produced its structured summary. That is the more
storage-efficient design (metadata/entities/concepts are a small fraction of
a document's raw text) and it loses on the property the field comment names
directly: re-materialisability. If only extracted metadata survives, the KG
can answer "what concepts does this document mention" but can never
reconstruct or re-process the document itself — a downstream consumer that
needs the actual text (e.g. distilling a document back into a skill-graph,
re-running a different extraction pass with a new prompt, or simply showing
a user the original content) has nothing to work from. Keeping `content`
verbatim makes every document a durable source-of-truth artifact in the KG,
not just a pointer to extracted facts about it.

This is the standardized ingestion contract every non-code content type
converges on (`AU-KG.ingest.ingestion-engine`'s `DOCUMENT` content type,
`AU-KG.ingest.deterministic-extraction-default`'s entity/claim extraction
target) — the decision to keep `content` verbatim is what makes those other
passes re-runnable against the same document later without re-fetching it.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/enrichment/models.py`'s `Document`
  model; every ingestion adaptor that constructs one.
- **Backward Compatible**: Yes — this documents the existing field.
- **Breaking Changes**: None.
- **Known weak point**: storing full verbatim content for every ingested
  document (a book can be millions of tokens) has a direct storage-cost
  tradeoff versus a metadata-only design; this document doesn't itself
  record any size cap or externalization strategy (e.g. content-hash
  dedup, blob offload) for very large documents.
