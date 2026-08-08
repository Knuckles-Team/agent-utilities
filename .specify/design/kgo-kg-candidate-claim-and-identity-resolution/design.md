# Design Document: Ambiguous extraction and identity resolution both stop at a candidate — never a direct write, never an auto-merge

CONCEPT:AU-KG.enrichment.candidate-claim-extraction ·
CONCEPT:AU-KG.identity.entity-resolution-candidates

> `agent_utilities/knowledge_graph/extraction/candidate_claims.py:1-30`
> and `agent_utilities/knowledge_graph/assimilation/identity_candidates.py:1-25`
> (module docstrings). Universal-ingestion program, Tracks 4 and 5
> (`reports/program/universal-ingestion.md`).

## Decision — extend the existing extraction/resolution machinery with a proposal type that has no write authority, rather than let either track collapse ambiguity itself

Both tracks solve the same shape of problem — a model or a similarity ladder
produces a plausible-but-uncertain claim about the graph — and both were
built by surveying and reusing what already existed rather than rebuilding
it, then adding exactly one new thing: a first-class **candidate** record
with **no direct write authority**.

**Track 4 (`candidate-claim-extraction`).** The extraction machinery already
existed — `extraction_schema` (ontology-guided closed-vocabulary prompting),
`fact_extractor` (the canonical prompt, streaming JSON parser, streaming
model factory, semantic dedup), `ontology_grounding`
(surface-form -> canonical OWL type), and the 7-state `ExtractionOutcome`
taxonomy. The one genuine gap: no `CandidateClaim` type existed repo-wide.
`propose` drives its own multi-round loop over the low-level primitives
rather than the higher-level `extract_facts` wrapper for one deliberate
reason: `extract_facts` hands back an already-`ExtractedFact`-coerced dict,
and `_coerce_fact` defaults a missing/unparsable `confidence` to `0` — which
is exactly the fabrication this module's "confidence honesty" requirement
refuses. Reading the raw parsed JSON before coercion is the only way to tell
"the model said zero confidence" apart from "the model said nothing at all".

**Track 5 (`entity-resolution-candidates`).** Deciding whether `payments
platform`, `payments-platform`, and a CMDB id are the same entity —
preserving ambiguity rather than merging incorrectly, because a wrong merge
is worse than no merge. `resolve_identity_candidates` reuses
`entity_resolution` (the entropy-gated exact + MinHash/LSH ladder) and
`fact_extractor.aggregate_confidence` (product-complement evidence
combination) directly. **The rejected alternative is named directly in the
docstring**: extend `.dedup`'s existing `SIMILAR_TO`/`SUPERSEDES` auto-merge
pass for the Feature/Article/SDDFeature research corpus instead of writing a
new module. Rejected on purpose — that pass auto-applies once a similarity
threshold clears, which is exactly the behavior this track's own charter
forbids for general entity identity. `identity_candidates.py` is a
deliberately separate, ambiguity-preserving sibling: same corpus of ideas
(normalized name matching, confidence combination), different write
discipline — `EntityResolutionCandidate` is never auto-confirmed.

## Risk Assessment

- **Blast Radius**:
  `agent_utilities/knowledge_graph/extraction/candidate_claims.py`,
  `agent_utilities/knowledge_graph/assimilation/identity_candidates.py`,
  exposed via `agent_utilities/mcp/tools/candidate_claim_tools.py`.
- **Backward Compatible**: Yes — both are additive proposal-only surfaces;
  neither touches the existing extraction/resolution/dedup code paths they
  reuse.
- **Known weak point**: both tracks are deliberately **useless without a
  confirming actor** — a `CandidateClaim`/`EntityResolutionCandidate` that
  nothing ever reviews just accumulates. Neither module's docstring claims
  to solve that; it is a downstream consumer's responsibility.
