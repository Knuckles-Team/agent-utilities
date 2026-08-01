# Design Document: Race scaffolding vs. weight updates against one verifier; promote only past a monotone ratchet

CONCEPT:AU-AHE.harness.sai-controller

> `agent_utilities/knowledge_graph/research/sai_factory.py`.

## Decision — the SAI factory composes the scaffolding arm and the weight arm against the SAME verifier and keeps whichever bought more adaptation speed

The module docstring (`sai_factory.py:4-32`) states what was missing before
this module directly: "AU had the scaffolding evolver, the trainer, the
verifier suite, and the harvest seam — but *no factory between them*." The
controller runs two improvement arms against the same machine-verifiable
reward — the **scaffolding arm** (search prompt/scaffold variants, keep the
one with highest verified reward) and the **weight arm** (harvest
verified-winning candidates into training data, fine-tune a specialist
generator via an injected callable) — and attributes each round to whichever
arm bought the larger marginal adaptation-speed gain, directly answering the
"better scaffolding vs. weight updates: which helped" question.

**The rejected alternative is treating scaffolding and weight-training as
two separate, uncoordinated improvement paths** — each with its own
promotion logic and no shared measurement of which one is actually
producing gains for a given task. Racing them against one verifier and one
`AdaptationCurve` makes the comparison apples-to-apples. Promotion is a
**monotone per-task ratchet**: the incumbent specialist is replaced only
when a challenger's verified reward on a *fresh* evaluation is ≥ incumbent −
tolerance, otherwise the round rolls back. **The rejected alternative here is
always promoting the most recent candidate** — without the ratchet, a
regression from either arm would silently become the new incumbent. The
heavy arms (real generator, real trainer, real harvest) are
constructor-injected callables specifically so the factory itself stays
GPU-free and testable in CI with default toy implementations.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/research/sai_factory.py`,
  `agent_utilities/harness/adaptation_speed.py` (`AdaptationCurve`), the
  certifier (SAFE-1.1), `agent_utilities/mcp/tools/analysis_tools.py`.
- **Backward Compatible**: Yes — a new orchestration layer over
  pre-existing evolver/trainer/verifier/harvest components; none of those
  changed their own contracts.
- **Known weak point**: the ratchet's `tolerance` allows a challenger that's
  *slightly worse* than incumbent to still be promoted (≥ incumbent −
  tolerance, not strictly ≥ incumbent) — over many rounds, a sequence of
  small within-tolerance regressions could compound into meaningful drift
  without any single round tripping the ratchet.
