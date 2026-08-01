# Design Document: Sample toward a diverse candidate set for test-time compute, not a single best answer

CONCEPT:AU-AHE.harness.width-diverse-best-k

> `agent_utilities/graph/test_time_diversity.py`.

## Decision — fan out a diversity-widened candidate set and select via MMR (quality vs. diversity tradeoff), instead of sampling toward one best answer

The module docstring (`test_time_diversity.py:4-21`) states the prior
approach and the change directly: this codebase already scales test-time
compute (`harness/reasoning_effort`) and fans out subagents
(`SubagentLifecyclePolicy`, `rlm/` parallel sub-calls), "but we sample toward
a *single* best answer." VPO (arXiv:2605.22817) shows that optimizing for a
*diverse* candidate set instead raises test-time best@k/pass@k.
`diverse_fan_out_width` reads the effort-derived width straight off the live
`ReasoningBudget.diversity_width` (no new agent-facing knob), and
`select_diverse` does MMR-style best-of-k selection explicitly trading
quality against diversity rather than ranking by quality alone.

**The rejected alternative is named in the docstring's own contrast: sample
N candidates and keep the single top-scored one.** That's what existed
before — more compute bought more attempts at the same target, but
converging attempts toward the same mode wastes the extra samples once
they cluster near each other; a diverse set instead explores more of the
solution space per unit of compute, which is what actually moves best@k/
pass@k. A second, smaller decision: the diversity kernel itself has a
default and an optional upgrade — `mean_pairwise_distance`/`select_diverse`
use a dependency-free embedding-spread computation by default; the
graph-native `epistemic-graph` `personalized_pagerank` kernel is optional.
**The rejected alternative there is making the graph-native kernel a hard
dependency** — instead diversity selection works standalone, with the
richer graph-seeded kernel as an upgrade path, not a requirement.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/test_time_diversity.py`,
  `agent_utilities/harness/reasoning_effort.py` (`ReasoningBudget.diversity_width`),
  `agent_utilities/knowledge_graph/core/owl_bridge.py`.
- **Backward Compatible**: Yes — reads an existing budget field; a caller
  not setting `diversity_width` gets the prior single-best behavior by
  default width.
- **Known weak point**: `select_diverse`'s MMR tradeoff is a fixed
  quality-vs-diversity balance point (not task-adaptive) — a task where
  quality variance among top candidates is small but genuinely meaningful
  could have its best candidate de-prioritized in favor of a more diverse
  but weaker one.
