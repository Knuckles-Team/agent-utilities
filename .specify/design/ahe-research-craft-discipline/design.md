# Design Document: Explicit disconfirming-evidence tracking + biggest-pile-first triage, not implicit confirmation bias

CONCEPT:AU-AHE.harness.research-craft-discipline

> `agent_utilities/harness/research_log.py`.

## Decision — operationalize two specific researcher-craft habits as deterministic, pure-Python structures

The module docstring (`research_log.py:3-25`) names both habits and why each
matters. `FailureTriage` operationalizes Andrew Ng's "Be good at research"
advice: pull a batch of failures, sort them into piles by label, and surface
the LARGEST pile — so remediation effort goes where it removes the most
loss, rather than being spent on whichever failure happened to be looked at
first. `ResearchLog` operationalizes Darwin's rule of recording any fact
that runs *against* a working hypothesis the moment it appears, because "the
mind forgets inconvenient evidence far faster than convenient evidence" —
guarding against exactly the failure mode Feynman named ("you are the
easiest person to fool"). It records belief evidence with an explicit
supports/refutes flag and flags hypotheses that are *contested* (carrying
both).

**The rejected alternative is the default failure mode both structures
exist to prevent: informal, memory-based triage and belief-tracking.**
Without a triage queue, failures get attacked in whatever order they're
noticed, not by loss-reduction priority. Without an explicit
disconfirming-evidence log, evidence against a working hypothesis is
subject to exactly the human/model bias Darwin and Feynman both named —
it's noticed, then forgotten, while confirming evidence accumulates
unchallenged. Both structures are deliberately pure-Python and deterministic
(no LLM call), and `FailureTriage.from_evidence_corpus` duck-types the
existing `continuous_evaluation_engine` `EvidenceCorpus` shape so triage is
fed directly from trace distillation rather than requiring a separate data
pipeline.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/research_log.py` only — a
  standalone instrumentation module, not wired into any gate.
- **Backward Compatible**: Yes — additive.
- **Known weak point**: "contested" hypothesis detection depends entirely on
  callers actually logging disconfirming evidence when they see it — the
  structure makes disconfirming evidence first-class and queryable *once
  recorded*, but nothing forces a caller to record it, so the log is only as
  honest as its callers' discipline in using it.
