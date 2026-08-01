# Design Document: Citation quality is precision/recall/F1 against retrieved docs, not a binary "cited or not" check

CONCEPT:AU-AHE.harness.citation-quality-tracking

> `agent_utilities/harness/citation_tracker.py`.

## Decision — track precision, recall, and F1 as separate metrics against the retrieved-doc set, not one pass/fail citation flag

`citation_tracker.py:4-13` states the motivation: inspired by BrowseComp-Plus
(arXiv:2508.06600), which reports citation precision/recall as **separate**
metrics specifically to prove that agents with better retrievers cite more
accurately. `CitationTracker.extract_citations` recognizes multiple citation
shapes in a response (`[KG:node-id]`/`[source:node-id]`, `CONCEPT:X`
references, external URLs, `file:///` references, arXiv IDs) and
`evaluate_citations` scores the extracted set against `retrieved_doc_ids` and
`gold_doc_ids` to produce a `CitationReport`: precision, recall, F1,
`hallucinated_citations` (cited but not in the retrieved set), and
`uncited_evidence` (retrieved but never cited).

**The rejected alternative is a single binary check — "did the response cite
anything at all" (or "did it cite the right document," collapsed into one
pass/fail).** That conflates two independently diagnosable failure modes: a
response can under-cite (leaving retrieved evidence unused — caught by
recall) or over-cite/hallucinate (citing something never actually retrieved —
caught by precision and surfaced explicitly in `hallucinated_citations`). A
single flag can't distinguish "the retriever found the right thing but the
generator didn't cite it" from "the generator invented a citation" — which
is exactly the diagnostic BrowseComp-Plus's separated metrics are built to
make, and which this tracker's shape preserves.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/citation_tracker.py` only — a
  standalone evaluation utility, not wired into any gate that blocks a
  response.
- **Backward Compatible**: Yes — additive instrumentation.
- **Known weak point**: `_CONCEPT_PATTERN`'s regex predates the OKF-CIS id
  migration and, per its own comment, was written to match the legacy
  numeric scheme (`KG-2.63`) and only incidentally also matches current
  `<SLUG>-<PILLAR>.<domain>.<concept>` ids — a citation format the pattern
  doesn't anticipate would silently fail to extract, undercounting recall
  rather than raising a visible parse error.
