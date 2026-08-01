# Design Document: Artifact → Fragment — a stable citation address that survives both inserts AND edits, by splitting identity from revision

> `agent_utilities/knowledge_graph/ingestion/evidence_spine.py`.

CONCEPT:AU-KG.ingest.evidence-spine-artifact ·
CONCEPT:AU-KG.ingest.stable-fragment-address

## Decision — two fields answer two different questions; neither alone can

`evidence_spine.py:1-40`, `250-262`.

**The gap, precisely stated**: `ChangeEnvelope` already carries the whole
connector contract and `ingest_graph_slice` already commits atomically (see
`.specify/design/kgi-change-envelope-atomic/design.md`); the engine's wire
protocol even declares an `Artifact` projection (`digest`, `content_ref`,
`segment_ids`, `loci`) — but NOTHING in Python ever constructed one, and
`segment_ids` had no Python type behind it at all. Separately,
`document_processing.py`'s existing chunk ids
(`{doc}::chunk::{index}:{sha(text)[:12]}`) are BOTH positional and
content-hashed — which breaks on an insert above (position shifts) AND on a
typo fix (hash changes).

**The rejected alternatives, both named explicitly**:

1. **A purely positional id** (`chunk 7`) — stable across body edits, but
   every insert above it renumbers every later fragment: one added
   paragraph invalidates the entire tail of the document's citations.
2. **A purely content-hashed id** — stable across inserts, but changes the
   moment a typo is fixed, silently breaking a citation to a passage that
   still says the same thing in substance.

**The design chosen**: two fields, two questions.

- **`CONCEPT:AU-KG.ingest.evidence-spine-artifact`** — `Artifact` is one
  retrieved source object: the thing a `digest`/`content_ref` identifies as a
  whole (a document, a page, an API response).
- **`CONCEPT:AU-KG.ingest.stable-fragment-address`** — `Fragment.fragment_id`
  is the **address**: derived from the artifact id plus a SCOPED STRUCTURAL
  PATH (e.g. `h2:getting-started/p:2`), never from the fragment's own body
  text — this is what a citation stores, and it survives a body edit because
  it's derived from structural position (heading/paragraph path), not
  content. `Fragment.content_hash` is the separate **revision** field —
  `sha256` over the fragment's normalized text — telling a reader whether
  the cited passage still says what it said, without that check being
  entangled with the citation's stability.

This is the general-purpose contract that `verbatim-extraction-before-fragmenting`
(see `.specify/design/kgi-verbatim-extraction-before-fragmenting/design.md`)
protects the INPUT structure for, and that
`markdown_fragmenter.py`/`ontology.ttl`/`shapes/governance.shapes.ttl` realize
as concrete node types and SHACL constraints.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/ingestion/evidence_spine.py`,
  `agent_utilities/knowledge_graph/domain_packs/markdown_fragmenter.py`,
  `agent_utilities/knowledge_graph/ontology.ttl`,
  `agent_utilities/knowledge_graph/shapes/governance.shapes.ttl`,
  `agent_utilities/protocols/source_connectors/connectors/git_markdown.py`.
- **Backward Compatible**: Yes — additive; existing chunk-only ingestion
  (without `Artifact`/`Fragment` construction) is unaffected.
- **Breaking Changes**: None for existing chunk ids; `Fragment`/`Artifact`
  are a NEW, parallel addressing scheme, not a replacement of
  `document_processing.py`'s chunk ids in place.
- **Known weak point**: the structural path (`path`, e.g.
  `h2:getting-started/p:2`) is itself sensitive to STRUCTURAL edits — moving
  a paragraph to a different heading section changes its path and therefore
  its `fragment_id`, even though the paragraph's own content and hash are
  unchanged. The scheme trades stability-under-content-edit for
  instability-under-structural-reorganization; it does not solve both
  simultaneously.
