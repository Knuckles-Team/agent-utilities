# Design Document: A PageIndex-style reasoning tree, self-verified before it is ever materialized

> `agent_utilities/knowledge_graph/ontology/document_processing.py`
> (`build_section_tree`, `verify_section_tree`, `build_section_tree_from_pages`);
> `agent_utilities/knowledge_graph/pipeline/document_ingestion.py:55-65`
> (`build_section_tree_for`, the commit-gating caller).

CONCEPT:AU-KG.ingest.structure-verify ·
CONCEPT:AU-KG.ingest.toc-detection

Related: `AU-KG.retrieval.section-tree` (the retrieval-side concept for
walking this tree — see `document_processing.py:705-707`); this doc covers
the INGEST-side build+verify+detect decisions specifically. Also related,
but a distinct family: `.specify/design/okf-skillgraph-roundtrip/design.md`
(`AU-KG.ingest.broken-link-tolerance`), which covers the markdown-link
extractor's dangling-node tolerance in the SAME file, a different pass over
the same document.

## Decision — self-verify the deterministic tree; LLM-assist ONLY the paginated path where headings can't be trusted

`document_processing.py:1909-1920` (`verify_section_tree`), `1395-1415` +
`1735-1745` (`build_section_tree_from_pages`).

**The problem**: a tree built purely from markdown headings is
self-consistent by construction (a heading IS its own char range). But the
LLM-assisted TOC-detection path — needed for paginated (PDF) documents that
have no markdown headings to walk — and aggressive token-budget thinning can
both produce a node whose claimed title is not actually found inside its
`[char_start, char_end)` span, since an LLM's reported page/title can drift
from the source text's true position.

**The rejected alternative**: trusting the LLM-detected TOC's char ranges
unconditionally once transformed into `{title, level, page}` entries — the
node commits with whatever span the detection pass produced, with no
independent check.

**The design chosen — `CONCEPT:AU-KG.ingest.structure-verify`**:
`verify_section_tree`, a port of PageIndex's `verify_toc`/`fix_incorrect_toc`,
checks EVERY node: does the title actually appear within its claimed
`[char_start, char_end)`? When `fix=True` (the default the ingestion caller
uses) and the title is found ELSEWHERE in the document, the node's
`char_start` is re-anchored to the title's true position (extending
`char_end` if needed) — repair, not rejection. `document_ingestion.py:55-65`
makes this a COMMIT GATE: "confirm every section title is inside its
claimed char range (repairing drift) *before* commit," returning the verify
report + section count so the caller can gate on structural integrity before
persisting.

**`CONCEPT:AU-KG.ingest.toc-detection`** — `build_section_tree_from_pages`
is the PAGINATED-document counterpart: the deterministic markdown path
(`extract_nodes_from_markdown` + level-stack tree building, ported from
PageIndex's `pageindex/page_index_md.py`) is LLM-FREE, but a PDF has no
markdown headings to walk deterministically. The leading `max_pages_for_toc`
pages are instead scanned with `llm_fn` (reusing the SAME remote vLLM
lite-LLM the contextual enricher already uses — no new model dependency) for
a table of contents; a detected TOC is transformed into
`{title, level, page}` entries and each entry's char span is resolved —
which is exactly what makes `structure-verify`'s drift-repair pass necessary
downstream: LLM-detected page/title mappings are the primary source of the
drift `verify_section_tree` exists to catch.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/ontology/document_processing.py`,
  `agent_utilities/knowledge_graph/pipeline/document_ingestion.py`.
- **Backward Compatible**: Yes — section-tree building is opt-in
  (`section_tree` param, off by default so the chunk-only pipeline stays
  byte-identical for callers who don't request it).
- **Breaking Changes**: None.
- **Known weak point**: `verify_section_tree`'s repair only succeeds when the
  title is found ELSEWHERE in the document — a title that was itself
  mis-transcribed by the LLM detection pass (not merely mis-positioned) has
  no correct location to repair to, and the mismatch is only logged
  (`report["mismatched"]`), not blocked from materializing.
