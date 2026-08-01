# Design Document: Search-task synthesis reuses the persistent evidence graph and adversarially self-tests each question before it is used, instead of re-mining/one-shot generation

CONCEPT:AU-KG.retrieval.evidence-subgraph-construction ·
CONCEPT:AU-KG.retrieval.evidence-graph-workspace ·
CONCEPT:AU-KG.retrieval.formulate-adversarially-refine ·
CONCEPT:AU-KG.retrieval.question-formulation-adversarial-refinement

> `agent_utilities/knowledge_graph/search_synthesis/` (`evidence_subgraph.py`,
> `shortcut_risks.py`, `question_formulation.py`, `__init__.py`), invoked from
> `agent_utilities/knowledge_graph/research/loop_controller.py:570`.

## Decision — the FORT-Searcher pipeline is distilled onto the PERSISTENT epistemic graph, not re-mined per run, and questions are adversarially refined against real shortcut detectors before they are trusted

`CONCEPT:AU-KG.retrieval.evidence-subgraph-construction`

`evidence_subgraph.py:4-17` states the rejected alternative explicitly: "Where
FORT re-mines Wikidata cycles per run, this checks out a bounded neighborhood
of a chosen answer entity from the *persistent, provenance-rich* epistemic
graph and converts incident facts into an `EvidenceGraph` workspace — reusing
existing provenance (`source`/`document_id`) so the downstream co-coverage
detector is an exact source-sharing test rather than a heuristic." Re-mining a
fresh graph per run (FORT's approach, over freshly scraped Wikidata) throws
away the provenance the KG already carries and turns a decidable check
(do two clues share a source?) into an estimate. Atomic-fact extraction is
deterministic; LLM-driven derived-fact construction and exact-value fuzzing
are an explicit pluggable `enrich` seam, omitted by default for a fully
deterministic, CPU-testable extraction — the rejected default there is
requiring an LLM call just to build the workspace.

### Pointer — `CONCEPT:AU-KG.retrieval.evidence-graph-workspace`

`search_synthesis/__init__.py:6`, `loop_controller.py:570`. This id names the
SAME `evidence_subgraph` module from the calling side — the package init
docstring and the research loop-controller both refer to the evidence-graph
workspace construction step by this second id where `evidence_subgraph.py`
itself uses `evidence-subgraph-construction`. It is not a second decision:
`loop_controller.py:570-573` invokes it as stage 6 of the research cycle
("SELF-PLAY SEARCH-TASK SYNTHESIS ... build shortcut-resistant deep-search
tasks from the evidence graph and draft a training corpus"), gated
**opt-in** (`synthesize_search`) specifically because it "does not depend on
open topics and is skipped by default to keep the zero-infra cycle cheap" —
the rejected alternative at the call site is running it unconditionally on
every research cycle regardless of cost.

### Pointer — `CONCEPT:AU-KG.retrieval.formulate-adversarially-refine`

`shortcut_risks.py:4-27`. The four *actionable shortcut risks* FORT-Searcher
formalizes (§2.3) — single-clue selectivity, evidence co-coverage, exposed
constants, prior-knowledge binding — as deterministic checks over the
provenance-rich `EvidenceGraph`. **The rejected alternative, named
explicitly**: "Unlike FORT (heuristics over freshly scraped pages) these run
as deterministic checks... so co-coverage is an exact source-sharing test,
not an estimate." A generated question that trips one of these is not a hard
multi-hop search task at all — it is a task an agent can shortcut past
without doing the intended evidence acquisition, which defeats the point of
synthesizing a search task in the first place.

### Pointer — `CONCEPT:AU-KG.retrieval.question-formulation-adversarial-refinement`

`question_formulation.py:4-19`. The last two FORT-Searcher stages: formulate
a natural-language question from a selected answer-bearing subgraph while
**withholding intermediate names** (so the question is not trivially
executable from its own surface), then run the four shortcut detectors above
as an adversary against the draft and repair it — pruning redundant
co-covered clues, generalizing over-selective clues to a larger candidate
pool (`_GENERIC_POOL`), withholding exposed constants — or report residual
risk so the caller can discard an unrepairable draft. **The rejected
alternative** is one-shot question generation with no adversarial check: a
question that happens to leak an intermediate name or over-narrow a single
clue looks fine on inspection but is not the hard multi-hop task it was
meant to be. The module additionally reuses the live `RetrievalQualityGate`
failure modes as an auxiliary reject signal when available, rather than
relying solely on the four synthetic detectors.

## Risk Assessment

- **Blast Radius**: `search_synthesis/*.py`, `loop_controller.py`'s stage 6.
  Opt-in (`synthesize_search`), so disabled call sites are unaffected.
- **Backward Compatible**: Yes — the whole pipeline is a new opt-in stage, not
  a modification of an existing one.
- **Known weak point**: the `enrich` seam and the `PriorProbe` closed-book
  check are both optional hooks defaulted to "off" (deterministic /
  CPU-testable). Without them, derived-fact construction and prior-knowledge
  binding detection are weaker than the full FORT-Searcher method describes —
  a deliberate CPU-first trade, but one that means the "adversarial" refinement
  is only as strong as the detectors actually wired in for a given run.
