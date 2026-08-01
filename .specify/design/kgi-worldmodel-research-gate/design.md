# Design Document: One shared novelty signal gates both the research pipeline and the news/finance/tech world-model pipeline — and research items always converge to the SAME path regardless of which feed found them

> `agent_utilities/knowledge_graph/assimilation/concept_matcher.py` (the
> shared matcher), `agent_utilities/automation/worldmodel_pipeline.py` (the
> world-model tiering runner), `agent_utilities/automation/research_pipeline.py`
> (the sibling research pipeline that reuses the matcher).

CONCEPT:AU-KG.ingest.world-model-gate ·
CONCEPT:AU-KG.ingest.news-finance-tech-sibling ·
CONCEPT:AU-KG.ingest.worldmodel-gated-ingestion

## Decision 1 — a robust, multi-signal matcher replaces a single weak cosine fallback

`CONCEPT:AU-KG.ingest.world-model-gate` — `concept_matcher.py:1-20`, `188`.

**The problem, quantified in the module docstring**: the first gap matcher
(`gap_analysis.auto_satisfy`) recognized a built capability only when a
feature *cited its concept id*, with "a single weak cosine fallback its own
docstring measured at '0/21 known-built capabilities … argmax wrong 71%'."
External research papers never cite internal `CONCEPT:` ids, so every paper
looked like an open gap no matter how much of the ecosystem was already
built.

**The rejected alternative is that exact prior matcher**: single weak cosine
similarity as the sole signal when a feature carries no `CONCEPT:` citation —
empirically wrong on the majority of real cases (71% argmax-wrong).

**The design chosen**: `ConceptMatcher`, a multi-signal, defense-in-depth
matcher deciding, for each feature (research `Article` / `sdd_feature` /
`capability`) against the ecosystem `Concept` registry, whether the feature's
contribution already exists. `concept_novelty` (`research_pipeline.py:369-384`)
exposes this as a cheap, no-LLM cosine probe returning novelty in `[0,1]`,
explicitly documented as "the shared 'relevant-enough-vs-existing-KG' signal
behind BOTH the research pipeline and the world-model gate" — one matcher, two
callers, rather than two independently-drifting novelty heuristics.

### Pointer — `CONCEPT:AU-KG.ingest.news-finance-tech-sibling`

`worldmodel_pipeline.py:1-22`. `WorldModelPipelineRunner` is explicitly named
"the news/finance/tech sibling of the KG-2.6 research pipeline": where the
research pipeline acquires academic papers, this builds a *world model* from
curated FreshRSS items. **The rejected alternative**: ingesting every
subscribed feed item unconditionally. Instead each item is tiered using the
shared novelty signal above plus a taxonomy score: **relevant** (score ≥
threshold AND not already-covered) gets a full native ingest via the
KG-2.48 `DocumentProcessor`; **marginal** (score ≥ marginal threshold, OR
highly novel but low score) gets a lightweight `ArticleNode` footprint only,
so the item is remembered without paying full-ingest cost; **skipped**
(below threshold, not novel) is dropped. An agent may still force-include an
item (`agent_force`) to override the automatic tier. Best-effort throughout —
a single bad item never aborts the sweep.

### Pointer — `CONCEPT:AU-KG.ingest.worldmodel-gated-ingestion`

`worldmodel_pipeline.py:445-457` (`_ingest_research`). **The rejected
alternative**: letting a Research/ScholarX/arXiv-tagged RSS item go through
the SAME world-model tiering as a news article. Instead, items under a
"Research (ScholarX)"/arXiv feed are routed OUT of the world-model tiering
entirely and INTO the unified research path — graded via
`grade_and_enqueue_paper` and enqueued as a prioritized
`research_paper_fetch` task (the KG-2.114 path) — so a research item arriving
via ANY feed (native RSS, ScholarX, FreshRSS-arXiv; see
`.specify/design/kgi-research-feed-convergence/design.md`) is graded and
fetched the SAME way, best-graded first, keyed off the canonical arXiv id so
duplicates across feeds collapse to one node.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/assimilation/concept_matcher.py`,
  `agent_utilities/automation/worldmodel_pipeline.py`,
  `agent_utilities/automation/research_pipeline.py`.
- **Backward Compatible**: Yes — `concept_novelty` degrades to `None` (no
  demotion) when no engine/concepts/embedder is available, so a host without
  the matcher wired up simply skips the novelty signal rather than failing.
- **Breaking Changes**: None.
- **Known weak point**: the taxonomy/novelty thresholds
  (`relevant_threshold`, `marginal_threshold`, `novelty_floor`,
  `redundancy_floor`) are static dataclass defaults on `WorldModelConfig`,
  not learned or auto-tuned — a taxonomy drift over time (new domains the KG
  cares about) requires a manual threshold/taxonomy update, not something the
  gate detects and adapts to on its own.
