# Design Document: The Foundry doc-processing pipeline, ported as a real end-to-end processor — never a stub

> `agent_utilities/knowledge_graph/ontology/document_processing.py`.

CONCEPT:AU-KG.ingest.chunk-overlap-stage

## Decision — real recursive separator-priority chunking with overlap, committed through one atomic envelope

`document_processing.py:1-45`, `146-205`.

**The reference pattern**: Palantir Foundry's `ontology / document-processing`
pipeline — media-set → text-extraction/OCR → chunk-with-overlap → explode →
embed → materialize `Chunk` objects linked to the source `Document`. This
module ports that pipeline shape, explicitly "never a stub."

**The rejected alternative, implicit in "never a stub"**: a naive
fixed-size/hard-cut chunker (split every N characters regardless of
structure) — the cheap implementation that would satisfy "chunking exists"
without satisfying "chunking is useful for retrieval." `ChunkingConfig`
instead uses REAL recursive separator-priority splitting (paragraph → line →
sentence → word → hard character cut, `DEFAULT_SEPARATORS`): the text is
segmented on the highest-priority separator that produces sub-`chunk_size`
pieces, pieces are packed greedily into windows up to `chunk_size`, and each
new window repeats the trailing `overlap` characters of the previous window
so retrieval keeps cross-boundary context. Character spans are tracked
against the original string, guaranteeing `char_start` is monotonic and
`span[i+1].char_start <= span[i].char_end` (asserted by the unit test) — a
real, verifiable contract, not an approximate one.

**A second explicit "never a stub" commitment**: PDF extraction uses
`pypdf`/`pdfminer` when importable and degrades to a CLEAR, EXPLICIT error
path otherwise — rejecting the alternative of a silent empty stub that would
make a missing PDF dependency indistinguishable from "this PDF legitimately
has no text."

**Materialization is atomic, not two-phase**: runtime commit goes through ONE
engine-native `ChangeEnvelope` for the complete document/chunk/section slice
(see `.specify/design/kgi-change-envelope-atomic/design.md`). The historical
`add_node`/`add_edge` facade writer is kept ONLY for offline construction and
explicitly-enabled test fixtures — the rejected alternative of using the
facade writer as the primary runtime path was replaced, not removed, since
offline/test callers still need it.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/ontology/document_processing.py`
  (`ChunkingConfig`, `ChunkSpan`, `chunk_text`, `DocumentProcessor`).
- **Backward Compatible**: Yes — `ChunkingConfig` defaults
  (`chunk_size=800`, `overlap=120`) reproduce prior chunking behavior;
  callers not customizing config are unaffected.
- **Breaking Changes**: None.
- **Known weak point**: `overlap` must be `< chunk_size` (validated), but
  there is no upper bound relative to `min_chunk_chars` — an aggressive
  overlap configuration close to `chunk_size` can produce heavily
  duplicated near-adjacent chunks, inflating embedding/storage cost without
  a guard rail warning the caller.
