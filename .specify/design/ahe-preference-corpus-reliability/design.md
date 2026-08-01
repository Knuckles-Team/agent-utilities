# Design Document: One deduplicated, reliability-filtered preference-pair store, not three disjoint implicit signals

CONCEPT:AU-AHE.harness.preference-corpus-reliability

> `agent_utilities/harness/preference_pairs.py`.

## Decision — consolidate three separately-generated preference signals into one DPO-ready store, then filter out unreliable pairs before they're used

The module docstring (`preference_pairs.py:4-27`) states the prior state
directly: this codebase already generated *implicit* preference signal in
three unrelated places — `EvalCorpus` regression cases carrying a
rejected/actual outcome in metadata, the trace distiller's
`EpisodeToPreferenceRule` (successful vs. failed episode over the same
context), and `FeedbackService` human corrections (`corrected_value` chosen
vs. original rejected) — "but never consolidated it into one clean,
DPO-ready store." `PreferencePairExporter` is that consolidation, producing
deduplicated `PreferencePair` records.

**The rejected alternative is what existed before: three disjoint sources, each requiring its own bespoke glue
code for any DPO-family trainer to consume**, and no deduplication across
them (the same underlying preference could show up independently from two
sources). On top of consolidation, `reliability_filter` implements RAPPO's
"keep the best, forget the rest" — dropping ambiguous or low-margin pairs
and logging the drop count. **The rejected alternative there is training on
every collected pair regardless of margin** — an ambiguous pair (where
"chosen" barely beats "rejected") contributes noisy gradient signal to a DPO
run; filtering them out before export is a smaller, second decision the
concept id names directly (reliability, not just consolidation).

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/preference_pairs.py`,
  `agent_utilities/harness/eval_corpus.py`,
  `agent_utilities/knowledge_graph/adaptation/feedback.py`.
- **Backward Compatible**: Yes — an additive export/consolidation layer over
  existing signal sources; none of the three source mechanisms changed.
- **Known weak point**: `reliability_filter`'s low-margin threshold is a
  single global cutoff — a source that inherently produces smaller reward
  margins than another (e.g. human corrections vs. eval-corpus pairs) isn't
  normalized before filtering, so one source could be systematically
  under-represented in the final corpus relative to its true reliability.
