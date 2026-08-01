# Design Document: Knowledge gaps and their exploration are formal, BFO-aligned graph nodes, not informal tracking

CONCEPT:AU-AHE.harness.exploration-discovery

> `agent_utilities/models/knowledge_graph.py:3524-3573` (`KnowledgeGapNode`,
> `ExplorationExperimentNode`).

## Decision — a gap and the experiment that explores it are typed, ontology-aligned nodes with a structured hypothesis/results/review chain

`KnowledgeGapNode` (`knowledge_graph.py:3524-3548`) represents a domain or
topic where the agent has identified insufficient knowledge — explicitly
aligned as `BFO:SpecificallyDependentContinuant` (an `:Observation`) — with
`hypothesis_ids` and `discovered_fact_ids` linking it forward into the rest
of the graph, and a closed `status` vocabulary (`identified | exploring |
filled | deferred`). `ExplorationExperimentNode` (`knowledge_graph.py:3550-3573`)
is the paired process type — `BFO:Process` (a `:Procedure`) — carrying
`design`, `variables` (a name→description map), `success_criteria`, `results`,
and `review_scores` for **multi-reviewer structured scoring**, not a single
scalar pass/fail.

**The rejected alternative is tracking exploration informally** — a gap noted
in a TODO, a prose comment, or an unstructured metadata blob, and an
experiment's outcome recorded as a single success/failure flag with no
design record or reviewer breakdown. That shape can't answer "which
hypothesis did this gap spawn," can't distinguish a gap that's actively being
explored from one that's been deferred, and collapses a multi-reviewer
evaluation into one number that hides reviewer disagreement. Making both a
formal BFO-aligned node type instead means a gap's lifecycle
(identified→exploring→filled/deferred) and an experiment's structure (design,
variables, success criteria, per-reviewer scores) are queryable graph state,
not prose scattered across docs and code comments.

## Risk Assessment

- **Blast Radius**: `agent_utilities/models/knowledge_graph.py` (node model
  definitions only — this is additive schema, not a rewrite of an existing
  gap-tracking mechanism).
- **Backward Compatible**: Yes — new node types.
- **Known weak point**: nothing in the two node models enforces that
  `hypothesis_ids`/`discovered_fact_ids` actually point at nodes that exist —
  these are plain string-id lists, so a stale or typo'd id silently breaks
  the graph traversal rather than failing at write time.
