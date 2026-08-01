# Design Document: Deliberation is triggered by disagreement UNDER uncertainty, not by disagreement or low confidence alone

CONCEPT:AU-AHE.harness.proposal-group-analysis

> `agent_utilities/graph/workspace_attention.py:556-610`.

## Decision — `deliberation_needed = diversity × (1 − |confidence − 0.5| × 2)`, peaking exactly where trajectories disagree AND are collectively unsure

`deliberation_score` (`workspace_attention.py:556-582`) analyzes a group of
parallel reasoning trajectories (proposals) to decide whether they warrant
sequential deliberation. It computes `consensus` (fraction agreeing on the
majority answer), `diversity` (normalized unique-answer count), mean
`confidence`, and combines diversity with a confidence-uncertainty term that
peaks when mean confidence is near 0.5 — the formula is explicit in-code:
"Deliberation is most beneficial when confidence is moderate but diversity
is high (trajectories disagree with uncertain reasoning)."

**The rejected alternative is triggering deliberation from a single signal
in isolation** — diversity alone (would also fire when trajectories
disagree but each is individually confident, where deliberation adds little)
or confidence alone (would fire on low-confidence agreement, where the
trajectories already converged and deliberation has nothing to reconcile).
The combined formula specifically suppresses both of those cases: high
diversity with confident trajectories on either extreme
(`confidence_uncertainty` near 0 at confidence near 0 or 1) yields a low
score, so the group is judged not to need deliberation even though it's
diverse — deliberation is reserved for the case where disagreement AND
genuine uncertainty coincide.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/workspace_attention.py` only —
  feeds heavy-thinking/deliberation integration decisions, does not itself
  gate anything destructive.
- **Backward Compatible**: Yes — a pure scoring function; callers decide
  what threshold on `deliberation_needed` triggers actual deliberation.
- **Known weak point**: the diversity fingerprint is the first 200
  characters of each proposal's output, lowercased — two proposals that
  reach the same conclusion via different reasoning but phrase their answer
  differently in the first 200 characters are counted as diverse, and two
  that reach different conclusions but share boilerplate opening text could
  be under-counted as consensus.
