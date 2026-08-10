# Design Document: The unified `:DurableExecutionUnit` KG mirror (DE1)

CONCEPT:AU-KG.storage.durable-execution-unit

> `agent_utilities/knowledge_graph/durable_execution_kg.py`

## Decision — a provenance MIRROR, never a second durability store

Full architecture: `docs/architecture/durable-execution.md` ("Durable
Execution — the unified plane, supersedes `restate` natively", program
`plans/au-eg-program/program/durable-execution-native.md`, register `D-DE-1`,
approved). The headline finding of that program is that epistemic-graph +
agent-utilities already implement the large majority of what `restate`
provides, in four independently-built subsystems (`eg-mutation-store`,
`eg-jobs`, `eg-statechart`,
`agent_utilities.orchestration.durable_execution.DurableRun`) — the gap was
never a missing durability mechanism, it was that these four surfaces were not
modeled as ONE KG-queryable concept family.

DE0 (lane `w6-de0-contracts`) reserved the concept block and wrote the
mirrored-property/edge schema into the canonical ontology
(`agent_utilities/knowledge_graph/ontology_orchestration.ttl`): one abstract
`:DurableExecutionUnit` class plus four subclasses (`:StatechartInstance`,
`:AnalyticsJob`, `:SagaCoordination`, `:DurableRun`), each a mirror of an
already-real, already-durable backend row. `durable_execution_kg.py` is DE1's
mirror-on-write implementation for the one backend fully reachable from Python
today — `DurableRun` — writing through the SAME `engine.add_node`/
`engine.link_nodes` surface `KgAuditSink`/`RunTrace` already use, not a
parallel write path. The other three subclasses have no writer here: their
backend rows are authoritative Rust state with no Python-callable read/write
path yet (a real, tracked gap — see `docs/architecture/durable-execution.md`'s
gap table — not silently narrowed away).

**The rejected alternative** was building a new, unified durability store that
all four subsystems write through. That would mean migrating three already-
working, independently-durable backends (`eg-mutation-store`'s saga
coordinator, `eg-jobs`'s fenced-lease jobs, `eg-statechart`'s OCC state) onto a
new storage layer for no reason beyond queryability — the mirror approach gets
the SAME cross-subsystem query ("what is durably in flight, waiting on what")
with zero migration risk to the systems already proven durable.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/durable_execution_kg.py`,
  `agent_utilities/knowledge_graph/ontology_orchestration.ttl`.
- **Backward Compatible**: Yes — additive KG schema and mirror-on-write path;
  no existing durability backend's write path is modified.
- **Known weak point**: fail-soft by design (a mirror write is a silent
  no-op, logged at DEBUG, when no engine is supplied) — a KG outage means the
  mirror silently falls behind the authoritative backend state, never that the
  durable run itself is affected. Three of four subclasses have no writer at
  all yet (statechart, analytics job, saga) — the mirror is incomplete by
  design until those wire-protocol surfaces exist.
