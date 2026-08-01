# Design Document: World-model feed ingestion gets its own scheduling lane so it never head-of-line-blocks behind research or codebase work

CONCEPT:AU-KG.ingest.worldview-stream

> `agent_utilities/knowledge_graph/core/task_lanes.py:75-100`.

## Decision — `feed_ingest` (relevance-gated news/world-event articles) runs in a dedicated `"worldview"` lane, separate from the `"research"` and file-ingestion lanes it would otherwise queue behind

`task_lanes.py:89-94` states the rationale directly at the lane
definition: **"the WORLDVIEW stream: relevance-gated news/world-event
articles (`feed_ingest`) build the world model. Its OWN lane so it drains in
parallel with — and never head-of-line-blocks behind — research-paper fetch
(which feeds agent-utilities-evolution) or the heavy codebase backlog. The
world-model gate is the router that splits feed items into research vs
here."**

**The rejected alternative is folding `feed_ingest` into an existing lane**
— either the `"research"` lane (task types `research_paper_fetch`,
`background_research`, `cohort_synthesize`) or the general/maintenance
file-ingestion lane. It loses because a fair-share/round-robin scheduler
serving one shared lane makes every task type in that lane wait behind
whatever else is queued there — a large codebase backlog or a slow research
cohort would delay worldview feed items with no way to prioritize between
them within one lane. Adjacent code in the same file demonstrates this is
not a hypothetical: the `"research"` lane's own comment
(`task_lanes.py:80-84`) documents a real prior bug of exactly this shape —
"under heavy cohort ingestion the maint floor-cap starved the gate so the
matrix never synthesized" — which is the precedent this decision is
deliberately avoiding for the worldview stream by giving it a lane of its
own up front, rather than discovering the same starvation failure later.

**The routing boundary is explicit, not implicit**: "The world-model gate is
the router that splits feed items into research vs here" — the same
underlying feed can produce items destined for either lane depending on
their relevance classification, so the lane split is a scheduling decision
downstream of a separate content-classification decision, not a hard
per-source partition.

`model_role: "lite"` (`task_lanes.py:96`) is the paired resourcing decision:
worldview items use the lite model role, consistent with the lane's purpose
being high-volume, relevance-gated intake rather than deep analysis (which
lives in the separate `"extraction"` lane, `task_lanes.py:99-105`, keyed to
`model_role: "learner"`).

## Risk Assessment

- **Blast Radius**: `core/task_lanes.py`'s lane table; the fair-share
  scheduler that reads it.
- **Backward Compatible**: Yes — this is the existing, shipped lane
  configuration.
- **Breaking Changes**: None.
- **Known weak point**: an additional dedicated lane increases the total
  number of lanes the fair-share scheduler must round-robin across, diluting
  each lane's effective share of worker capacity under contention — the fix
  for one starvation failure mode (worldview behind research/codebase)
  trades against overall throughput per lane as more lanes are added.
