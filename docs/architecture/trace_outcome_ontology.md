# Canonical execution trace and outcome ontology

Runtime provenance has one durable shape:

```text
RunTrace -[:USED_TOOL]-> ToolCall
RunTrace -[:PRODUCED_OUTCOME]-> OutcomeEvaluation
```

`agent_utilities.observability.trace_ontology` is the schema authority. Agent execution, workflow
execution, workspace provenance, evaluation, failure mining, placement mining, context compilation,
variant fitness, and skill synthesis consume this same shape. `Episode` remains a domain-memory type;
it is not an alternate execution-trace schema.

## Identity and privacy

Run identifiers are converted to stable opaque references before persistence. Durable traces store
`attribution_ref`, `actor_ref`, `tenant_ref`, `correlation_ref`, `skill_ref`, and `server_ref`; they do
not store the source identity strings. Task, argument, result, error, and feedback text is
metadata-only: durable rows contain keyed opaque references, character counts, and a redaction report,
never the free text itself. Workspace action identifiers also derive from the opaque trace reference,
so a run id, user name, or machine path is not copied into node ids.

Call `trace_id(run_id)` whenever code needs the durable node id. Do not construct `trace:<run_id>` or
`toolcall:<run_id>:<sequence>` directly.

## Ordered incremental consumption

Every trace, tool call, and outcome has an integer `event_sequence`/`event_cursor`. Incremental
consumers use `TraceCursor` and resume with `event_sequence > after_sequence`. Each consumer's
greatest fully completed sequence is stored as a `TraceConsumerCursor` node through
`load_trace_cursor`/`save_trace_cursor`; the consumer key is an opaque reference, and no cursor is
written to a local file. Each advancement appends an immutable, sequence-addressed checkpoint and
updates a compatibility head. Readers select the greatest checkpoint, so concurrent hosts cannot
regress the durable cursor even if their mutable-head updates race. Timestamps and opaque
identifiers are display/identity values, never lexical ordering cursors.

Bounded queries conservatively drop the final trace whenever a page fills, because its tool-call
rows may be incomplete. The next pass re-reads that trace. A miner advances its graph-resident
cursor only after mining, proposal persistence, validation, and policy routing complete without an
error; failed batches replay safely.

## Outcomes

Every runtime writer attaches one stable `OutcomeEvaluation` id to its `RunTrace`. The normalized
outcome carries status, bounded reward, success, criteria, feedback digest, and the same event
sequence as its trace. Evaluation and analytics read this edge rather than inferring success from a
different node family.

## Querying a run

Use `graph_jobs action=status` with the public run handle. It resolves the opaque trace id and returns
the trace plus ordered tool calls. For direct graph traversal, use the returned `trace_id`:

```cypher
MATCH (r:RunTrace {id: $trace_id})-[:USED_TOOL]->(t:ToolCall)
RETURN t.tool_name, t.status, t.sequence
ORDER BY t.sequence
```

Miners persist only the greatest completed numeric cursor. A failed batch therefore retries without
skipping traces, while a successful batch advances deterministically across process restarts.
