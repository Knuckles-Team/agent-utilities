# Design Document: Deterministic entity/claim extraction becomes the DEFAULT third enrichment pass for every text ingestion, not an opt-in action

CONCEPT:AU-KG.ingest.deterministic-extraction-default

> `agent_utilities/knowledge_graph/ingestion/engine.py:1377-1400,1461-1470,1550-1563`
> (primary), `agent_utilities/knowledge_graph/kb/entity_claim_extractor.py:293-300`,
> `agent_utilities/models/knowledge_graph.py:2105` (the `ExtractionRun.model_ref`
> field this feeds), `tests/unit/knowledge_graph/test_extraction_run.py`,
> `tests/unit/knowledge_graph/test_ingestion_enrich.py:270`.

## Decision — make `EntityClaimExtractor` a mandatory generic-ingestion stage, deterministic-first, with typed failure instead of silent "no facts found"

`engine.py:1377-1387` names the decision directly: `EntityClaimExtractor`
was **"previously reachable only from the explicit `graph_analyze
extract_claims` action and second-brain sync"**; this makes it **"the
DEFAULT generic-ingestion stage: every text-bearing ingestion funnels
through this seam, so codebase/document/connector content all get
deterministic entity/claim extraction with no per-content-type opt-in."**

**The rejected alternative is exactly the prior state**: extraction as an
opt-in action a caller must explicitly request (`graph_analyze
extract_claims`) or that only ran inside the separate second-brain sync
path. It loses because it means most ingested content — a document, a
connector row, a codebase file — never gets entity/claim extraction unless
something downstream happens to call the action on it. Folding it into the
"unified always-on intelligence layer" (`engine.py:1455-1456`, the same seam
that already runs Concepts and Facts extraction) makes enrichment global
rather than per-adaptor bespoke — the module calls this out as "*Native by
default*" (`engine.py:1456`).

**The cascade rule bundled into this decision**: deterministic first, LLM
only for what a schema pack's regex link-inference can't resolve — "no
generative call on this path" (`engine.py:1391-1393,1550-1552`). This is
consistent with the codebase's minimal-LLM-cascade convention elsewhere, and
here it's load-bearing: because the pass is deterministic (regex + pack
extraction), it needs no LLM slot, so making it mandatory-default doesn't
consume the same scarce LLM capacity budget that the Concepts/Facts layers
do (`engine.py:1555-1561`).

**The rejected failure mode this decision explicitly guards against**: an
extractor import/construction failure recorded as a typed
`extractor_unavailable` run rather than raised, "so a broken environment can
never masquerade as 'no facts found'" (`engine.py:1394-1397`,
`entity_claim_extractor.py:291-300`). The alternative — letting a
construction failure surface as an empty/default `ExtractionRun` — would be
indistinguishable from a legitimate "the document really had no
extractable facts" outcome, silently corrupting the extraction-outcome
telemetry (`CONCEPT:AU-KG.enrichment.extraction-outcome-taxonomy`, a
separate documented concept this one composes with).

**Cost containment, not scope reduction**: because this now runs on every
document by default, cost is bounded by *priority*, not by opting whole
content types out. `engine.py:55-62` reuses
`AU-ORCH.scheduling.resource-priority-edict` (not a second priority notion)
so a whole-workspace ingest run tagged `BACKGROUND_INGESTION` only extracts
the first N enrichment windows per document (`_EXTRACTION_BACKGROUND_WINDOW_CAP`);
the rest are recorded `BUDGET_DEFERRED`, never silently dropped
(`engine.py:1568-1576`). An `INTERACTIVE`/`ORCHESTRATION`/`HYDRATION`-tagged
call (e.g. the explicit `graph_analyze extract_claims` action) stays
untagged/high-priority and is not capped this way.

`models/knowledge_graph.py:2103-2105`'s `ExtractionRun.model_ref: str | None`
is the durable trace of the cascade rule: `None` specifically means the run
was purely deterministic (no generative call) — the cascade default this
concept names.

## Risk Assessment

- **Blast Radius**: `ingestion/engine.py` (the unified enrichment pass, run
  on every text-bearing ingestion), `kb/entity_claim_extractor.py`,
  `models/knowledge_graph.py` (`ExtractionRun`).
- **Backward Compatible**: Yes — the explicit `graph_analyze extract_claims`
  action and second-brain sync paths still work unchanged; this adds a
  default path rather than removing the opt-in one.
- **Breaking Changes**: extraction now runs (bounded, best-effort) on every
  ingestion by default where it previously ran on none — a behavior change
  in volume of `ExtractionRun`/`Entity`/`Claim` nodes produced, not an API
  break.
- **Known weak point**: the background-priority window cap means a large
  document under `BACKGROUND_INGESTION` priority gets partial extraction
  coverage by design (`BUDGET_DEFERRED` for the rest) — callers that assume
  "ingested" implies "fully extracted" need to check the outcome taxonomy,
  not just success/failure.
