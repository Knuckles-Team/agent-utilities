# Graph-Native Durable Execution (CONCEPT:AU-ECO.messaging.native-backend-abstraction)

Durable orchestration uses the Rust-native `epistemic-graph` engine as its sole
work-state authority. Every resumable unit is a `WorkItem`; no parallel store,
lease table, or task record owns execution state.

## Authority model

<div class="admonition architecture" markdown>
<p class="admonition-title">Submit, claim, run, commit — every step recorded in the authority</p>

A GraphOS dispatch submits a deterministic `WorkItem`, which is natively
claimed (lease epoch + fencing token), then executes a bounded step. Before
any side effect, the lease is renewed via heartbeat, then committed
natively and idempotently, producing result/provenance references. Submit,
claim, and the final result all write into the epistemic-graph authority.
</div>

The authoritative lifecycle is:

```text
submitted -> ready -> leased -> running
    -> succeeded | failed | cancelled | dead_letter
```

The native transaction family owns claim, heartbeat, lease expiry, fencing,
dependency release, retry, cancellation, and terminal commit. A transition is
accepted only when its tenant, lease owner, lease epoch, and fencing token still
match. If the engine-native verb is unavailable, execution fails closed.

## Checkpoints and replay safety

A WorkItem carries the portable execution references needed to resume work:
`payload_ref`, `checkpoint_id`, dependency identifiers, attempt and retry state,
and terminal `result_ref` or `error_ref`. Checkpoint bodies and large results are
addressed by opaque references; machine paths, display names, credentials, and
raw payloads do not become work metadata.

Replay safety comes from two engine-enforced invariants:

- **Deterministic identity and idempotency.** A logical dispatch reuses its
  deterministic WorkItem/idempotency key. Repeated delivery cannot create a
  second execution authority, and a repeated matching terminal commit is a
  no-op.
- **Fenced effects.** Workers renew their lease immediately before a bounded
  side effect and commit with the matching fence. A stale worker cannot commit
  after a newer lease epoch has been issued.

Queue acknowledgements happen only after the fenced WorkItem result is durable.
A crash before acknowledgement therefore causes a safe redelivery; a crash after
the native commit resolves through the same idempotency key without applying the
effect twice.

## GraphOS surface

Submit durable work through `graph_jobs` (or the higher-level
`graph_orchestrate` and workflow tools), then inspect the same WorkItem by its
returned job identifier:

```json
{
  "tool": "graph_jobs",
  "arguments": {
    "action": "dispatch",
    "task": "Summarize the latest governed ingestion results",
    "dependencies": "[]"
  }
}
```

```json
{
  "tool": "graph_jobs",
  "arguments": {
    "action": "status",
    "job_id": "<opaque job identifier>"
  }
}
```

`status` is a read-only projection of the WorkItem. It never exposes lease
capabilities or creates another writable lifecycle. Raw WorkItems remain
queryable through governed `graph_query` Cypher when an operator needs the DAG
or audit view.

`graph_jobs(action="cancel")` is the matching cooperative cancellation request.
It uses the native `CancelWorkItem` transition and returns `not_cancelled` when
the job is missing, terminal, or cannot be cancelled under its active lease.

### MCP Tasks compatibility

GraphOS maps asynchronous job handles to the durable WorkItem authority; it
does not create a second task store. The 2026-07-28 MCP Tasks extension requires
per-request capability negotiation, `server/discover`, polymorphic
`resultType: "task"` responses, and `tasks/get`, `tasks/update`, and
`tasks/cancel` wire handlers. FastMCP 3.4.5 explicitly depends on MCP Python SDK
`<2` and exposes the incompatible 2025-11-25 experimental lifecycle instead.
The official MCP Python SDK 2.0.0 now implements the new protocol, but it cannot
be installed underneath this FastMCP release without violating that dependency
contract. GraphOS therefore disables FastMCP's legacy Tasks capability rather
than falsely advertising the extension. Status and cancellation remain
available through `graph_jobs` and `/api/graph/jobs`; full Tasks support requires
the governed MCP SDK v2 migration or a tested dual-stack protocol adapter.

The tested MCP v2 gateway is that dual-stack adapter. Its public requests remain
stateless, and each downstream operation uses a fresh short-lived legacy GraphOS
MCP session. Discovery and listing use one downstream session. A normal tool call
uses one session for its authorization-filtered catalog and another for the call.
A durable dispatch uses three: catalog, dispatch, and the status poll that verifies
the WorkItem before returning its task handle.

Every session that advertises or calls `graph_jobs` activates exactly that gated
tool with `load_tools(tools=["graph_jobs"], auto_unload=True)` and confirms it in a
second `tools/list` on the same session. Calls auto-retract the tool after use;
list-only and error paths perform an idempotent `unload_tools` before terminating
the session. Empty multiplexer visibility records are pruned, so concurrent
short-lived sessions neither share visibility nor accumulate process-global state.
A failed activation or confirmation remains fail-closed: Tasks are not advertised
and `graph_jobs` is not called. Authorization, tenant parameters, and trace headers
are forwarded unchanged through every session step.

<div class="admonition architecture" markdown>
<p class="admonition-title">The v2 gateway spends three short-lived legacy sessions per dispatch</p>

An MCP v2 client sends `graph_jobs` dispatch with tasks to the v2 gateway,
which talks to the GraphOS legacy MCP over three separate, short-lived
sessions, each fully torn down (activate → ... → unload → DELETE): session
1 lists the catalog, session 2 dispatches with auto-unload, session 3 polls
status with auto-unload. The gateway returns one durable task handle to
the client, hiding all three legacy sessions behind it.
</div>

## Operational checks

- Run `agent-utilities-doctor --only engine a2a_persistence` before dispatch.
- Run `agent-dispatch-worker` for queued agent turns; workers claim and commit
  through native WorkItem operations.
- Treat `dead_letter`, repeated lease loss, or an unavailable native verb as an
  operational failure. Do not add a local fallback authority.

See [Queue-Driven Agent Dispatch](../architecture/agent_dispatch.md),
[Graph Authority Convergence](../architecture/graph-authority-convergence.md),
and the [Unified Scheduling recipe](../recipes/unified-scheduling.md) for the
worker, queue, and dependency-DAG details.
