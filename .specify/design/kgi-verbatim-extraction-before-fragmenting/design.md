# Design Document: Read the format reader's raw output, never the KB parser's word-count-flattened text

> `agent_utilities/knowledge_graph/ontology/document_processing.py:846-862`
> (`_read_file`), fixing bug D-ES-1.

CONCEPT:AU-KG.ingest.verbatim-extraction-before-fragmenting

## Decision — bypass `KBDocumentParser.parse_file()`, call its `_read_file` reader directly

`document_processing.py:846-862`.

**The bug this fixes, stated precisely in the docstring**: `parse_file()`
extracts text AND then runs it through `_chunk_text` — a WORD-COUNT splitter
(`text.split()` then `" ".join(...)`) built for `KBDocumentParser`'s own
retrieval use. That splitter collapses EVERY whitespace boundary — blank
lines between paragraphs, heading/list/table line breaks — before this
processor's own code ever saw the text. Both this processor's own chunker
(`chunk_text`) AND the evidence-spine fragmenter (`fragment_markdown`) then
ran on that ALREADY-FLATTENED text, so a markdown file's headings,
paragraphs, and tables were invisible to BOTH — fragments degraded to
whole-document granularity, defeating the exact structural addressing
`evidence_spine` exists to provide (see
`.specify/design/kgi-evidence-spine-contract/design.md`).

**The rejected alternative is the prior behavior itself**: calling
`parse_file()` (the convenient, already-available KB parser entrypoint) and
accepting its word-count-flattened output as input to structural chunking.
It worked in the sense that text was extracted, but silently defeated the
structural-fragmenting invariant every consumer downstream assumed held.

**The design chosen**: `_read_file` calls `KBDocumentParser._read_file`
DIRECTLY — the format reader's raw output only (still verbatim md/txt/html,
or genuinely-extracted pdf/docx/epub text; those formats have no "verbatim"
original to preserve beyond what extraction already produces) — skipping
`parse_file()`'s word-count chunking step entirely. This processor's OWN
chunker and the evidence-spine fragmenter now see the text's real structure
(blank lines, headings, list/table breaks) intact.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/ontology/document_processing.py`
  (`DocumentProcessor._read_file`), and transitively every consumer of
  file-sourced markdown chunking/fragmenting (`chunk_text`,
  `fragment_markdown`, the evidence spine).
- **Backward Compatible**: Behaviorally a fix, not additive — documents
  ingested BEFORE this fix have fragments/chunks at whole-document
  granularity for file-sourced markdown; a re-ingest after the fix produces
  finer-grained fragments for the SAME source, which is a content change to
  existing citations' granularity, not merely new data.
- **Breaking Changes**: A fragment address computed against the
  pre-fix (flattened) structure will not exist post-fix for a re-ingested
  document — any citation stored against the old, coarser fragment id
  becomes unresolvable after re-ingestion.
- **Known weak point**: the fix is scoped to the FILE-sourced path
  specifically (`_read_file`); any OTHER caller of `KBDocumentParser.parse_file()`
  elsewhere in the codebase that also feeds structural chunking/fragmenting
  would need its own equivalent fix — this change does not touch
  `parse_file()` itself, so the underlying flattening behavior still exists
  for whatever legitimately wants it (KB's own retrieval use case).
