# Design Document: Test-time search compute is captured back into training data, not spent once and discarded

CONCEPT:AU-AHE.harness.search-distillation-harvester

> `agent_utilities/harness/search_distillation.py`.

## Decision — harvest the reasoning router's verified high-scoring results and best-of-k winners into a collapse-guarded SFT + preference-pair corpus

The module docstring (`search_distillation.py:4-16`) names the source
paper's answer to the data wall (§5.1/§5.3) directly: convert test-time
compute into better training data, AlphaZero-style — distil search-improved
outputs back into a corpus that trains the next prior. This harvester taps
the KG-2.68 reasoning router's scored paradigm results and test-time-diversity's
scored best-of-k candidate sets, rejection-samples the winners into
`(prompt → completion)` SFT rows and `(chosen, rejected)` preference pairs,
gates each through the SAFE-1.4 model-collapse guard, and persists a
versioned `SyntheticCorpus`.

**The rejected alternative is spending expensive test-time search compute on
a single answer and discarding it once that query is answered.** A
best-of-k search or a multi-paradigm reasoning pass is costly precisely
because it explores more than a single greedy generation would — throwing
that exploration away after producing one final answer wastes the signal
it generated about which reasoning paths actually worked. A second
decision inside the same harvester: every harvested row is gated through
the model-collapse guard before being persisted. **The rejected alternative
there is training directly on model-generated outputs with no distributional
check** — the well-documented failure mode where a model increasingly trains
on its own (increasingly narrow) outputs and collapses toward reduced
diversity; gating at harvest time keeps that check upstream of the trainer
rather than trusting the trainer to catch it.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/search_distillation.py`,
  the KG-2.68 reasoning router, the in-house trainer (ML-001..007) that
  consumes `SyntheticCorpus`.
- **Backward Compatible**: Yes — the harvest + curation step is pure and
  in-repo; actual fine-tune consumption needs external compute and is a
  separate, later step.
- **Known weak point**: rejection sampling keeps only the winners from
  scored search — a systematically-hard subclass of prompts that the
  reasoning router never scores highly on any paradigm contributes nothing
  to the harvested corpus, so the training signal skews toward
  already-tractable problems rather than the frontier the search was
  exploring.
