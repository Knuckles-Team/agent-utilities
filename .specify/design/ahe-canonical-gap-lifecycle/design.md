# Design Document: ONE canonical `:Gap` node type, replacing four disjoint gap representations

CONCEPT:AU-AHE.harness.canonical-gap-lifecycle

> `agent_utilities/knowledge_graph/research/gaps.py` (primary), consumed by
> `agent_utilities/knowledge_graph/adaptation/failure_analyzer.py`,
> `agent_utilities/knowledge_graph/adaptation/skill_evolver.py`,
> `agent_utilities/knowledge_graph/assimilation/plan_synthesis.py`,
> `agent_utilities/knowledge_graph/research/change_publisher.py`,
> `agent_utilities/knowledge_graph/research/spec_proposals.py`,
> `agent_utilities/mcp/tools/state_tools.py`,
> `agent_utilities/capabilities/governed_capability_authoring.py`.

## Decision — every discovery track submits the same `:Gap` node type and rides the same lifecycle

The module docstring (`gaps.py:1-22`) states the prior state explicitly:
before this module, the platform had **three unrelated notions of "gap"** — a
transient `failure_gap` `:Concept` dict, a dead-end `sdd_plan` node, and an
in-memory `SkillGap` — plus a fourth about to be added (code-audit findings),
joined by 7+ disjoint id schemes. A `KnowledgeGapNode` model already existed
in `agent_utilities.models.knowledge_graph` but was never actually
instantiated by anything.

This module repurposes that dormant model as the single gap representation:

- `submit_gap` persists one canonical `:Gap` node (id `gap:<source>:<signature>`)
  **and gives it a WorkItem/lease** via `ensure_loop_work_item`, so a
  discovered gap is a first-class, leaseable, schedulable work item — not a
  transient dict that only lives in one caller's memory.
- `link_gap_to_spec` writes the `(:Gap)-[:SPECIFIED_BY]->(:SpecProposal)` edge —
  the first hop of a unified provenance chain.
- `mark_gap_resolved` / `resolve_gaps_for_loop` close the loop: when the gap's
  derived develop-Loop publishes, the origin gap's status flips to `resolved`
  — a visible end state the previous three-representation model never had.

**The rejected alternative is what already existed: each discovery track
(production-failure via `failure_analyzer.py`, research/OSS via
`plan_synthesis.py`, skill-coverage via `skill_evolver.py`, and the
about-to-be-added code-audit track) keeps its own gap shape and its own id
scheme.** That model can't answer cross-track questions ("how many gaps are
open right now, across all sources?") without reconciling four incompatible
representations, and a `sdd_plan` node that dead-ends has no path back to
"resolved" at all. All reads/writes here are deliberately best-effort and
backend-agnostic — status is recovered from either the node's top-level prop
or its folded `metadata` JSON — mirroring the same discipline
`spec_proposals` already uses, so the module works on both strict and
schemaless backends without a second code path.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/research/gaps.py` and
  every discovery-track module that now calls `submit_gap` instead of writing
  its own representation (7 call sites across
  `failure_analyzer.py`, `skill_evolver.py`, `plan_synthesis.py`,
  `change_publisher.py`, `spec_proposals.py`, `state_tools.py`,
  `governed_capability_authoring.py`).
- **Backward Compatible**: The unification is additive going forward; any
  pre-existing `failure_gap`/`sdd_plan`/`SkillGap` records created before this
  module are not migrated by it.
- **Known weak point**: the id scheme (`gap:<source>:<signature>`) depends on
  each track computing a stable, non-colliding `signature` — two tracks that
  independently derive the same signature for genuinely different underlying
  problems would silently fold into one `:Gap` node.
