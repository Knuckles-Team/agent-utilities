# Durable Execution — the unified plane (supersedes `restate` natively)

> Program: `plans/au-eg-program/program/durable-execution-native.md` (register
> `D-DE-1`, approved). Lanes DE0–DE8 (DE4 deferred into the open
> `engine-sql-graph-program`). This page is DE8's deliverable: state the
> unified plane, name every restate mechanism's disposition, and record what
> is landed vs. still a tracked gap — honestly, not as a closure fabrication
> (`AGENTS.md` → *Stand your evidence up before you act on it*).

## The headline finding

epistemic-graph + agent-utilities already implemented the large majority of
what `restate` (`open-source-libraries/restate`) provides — under different
names, in four subsystems built independently for other reasons:

| Subsystem | What it is | Already durable? |
|---|---|---|
| `eg-mutation-store` | Commit-before-ack journal (`BATCHES`/`IDEMPOTENCY`/`VERSIONS`/`OUTBOX`) **plus a real saga/2PC coordinator** (`prepare_saga`/`commit_saga`) restate itself does not have | Yes |
| `eg-jobs` | Durable async jobs, fenced-lease CAS (`claim_next`/`renew_lease`/`checkpoint_fenced`), cron/interval/manual triggers | Yes |
| `eg-statechart` | Durable keyed OCC state (`MachineInstance`), routed through the same `MutationBatch`/`eg-mutation-store` gateway `eg-jobs` uses | Yes |
| `agent_utilities.orchestration.durable_execution` (`DurableExecutionManager`/`DurableRun`) | Python-side named-step checkpoint-and-resume, SQLite/Postgres-backed | Yes |

**The gap was never a missing durability mechanism.** It was that these four
surfaces were not modeled as ONE KG-queryable concept family, so nobody could
ask *"what is durably in flight, waiting on what"* across all of them. That
unification — not a faster journal — is what "superior and synergized" means
here, and it is exactly the kind of gap this workspace's ontology-driven KG
exists to close.

## Architecture

```text
 agent / tool code
        │  durable_run() · durable_sleep() · [durable_call/state_get/state_set: see gaps below]
        ▼
   agent_utilities.orchestration.durable_tool_surface  (DE2 — the ONE mental model,
        │                                                the ctx.* analog)
        │        eg-durable  (crates/eg-durable — the routing CONTRACT, DE0)
        │
        ├─ keyed single-writer state  → eg-statechart (MachineInstance, OCC)        [not Python-reachable yet]
        ├─ async/checkpointed work    → eg-jobs (AnalyticsJob, fenced leases)        [not Python-reachable yet]
        ├─ cross-store atomic step    → eg-mutation-store (saga/2PC coordinator)     [not Python-reachable yet]
        └─ agent-loop continuations   → DurableRun (Python step checkpoints)         [LIVE — durable_run/durable_sleep]
        │
        ▼  every path already commits through
   eg-mutation-store  (commit-before-ack journal: BATCHES/IDEMPOTENCY/VERSIONS/OUTBOX)
        │
        ▼  mirrored as KG nodes/edges (DE1 — implemented for :DurableRun only, see gap below)
   :DurableExecutionUnit  ⊂ {:StatechartInstance, :AnalyticsJob, :SagaCoordination, :DurableRun}
        │  ─PRODUCED→ RunTrace/ToolCall   ─AWAITS→ :DurableExecutionUnit   ─COORDINATED_BY→ :SagaCoordination
        ▼
   durable_tool_surface.durable_status() / graph_durable(action="status")
   "what is durably in flight, for whom, waiting on what" — today: :DurableRun only
```

The routing decision itself (which backend serves which call shape) is
`crates/eg-durable`'s `CallShape`/`WorkShape`/`DurableBackendKind` contract
(DE0) — a pure, dependency-free Rust crate, default-off (`durable` feature,
excluded from `full`, same posture as `viz`/`quantum`), linking none of the
four real backends. `agent_utilities.orchestration.durable_tool_surface` (DE2)
is its one AU-side caller.

## Ontology (DE0 schema, DE1 mirror-on-write)

Reserved concept block (`docs/concept_reservations.d/w6-de0-contracts.yaml`,
OKF-CIS): `AU-KG.storage.durable-execution-unit`,
`AU-KG.storage.statechart-instance-mirror`,
`AU-KG.txn.saga-coordination-mirror`,
`AU-ORCH.scheduling.analytics-job-mirror`,
`AU-ORCH.runvcs.durable-run-mirror`,
`AU-ORCH.execution.durable-routing-trait`. Schema landed in
`agent_utilities/knowledge_graph/ontology_orchestration.ttl`: one abstract
`:DurableExecutionUnit` class, four subclasses (`:StatechartInstance`,
`:AnalyticsJob`, `:SagaCoordination`, `:DurableRun`), properties `:backendRef`
/ `:checkpointRef` / `:definitionVersion` / `:durableStatus` /
`:idempotencyKey` / `:leaseEpoch`, and edges `:awaits` / `:coordinatedBy` /
`:produced` ↔ `:producedBy`.

Every `:DurableExecutionUnit` node is a **PROVENANCE MIRROR** — a foreign key
back to the authoritative row in its owning backend (`:backendRef`), never a
second store, the exact pattern `RunTrace`'s own checkpoint linkage
(`graph_checkpoint_ids`/`graph_resume_supported`) already established.

**Mirror-on-write is implemented for `:DurableRun` only**
(`agent_utilities/knowledge_graph/durable_execution_kg.py`, wired into
`DurableRun.step()`/`.finish()`/`__init__` via an optional `engine=` param,
fail-soft exactly like `KgAuditSink(engine=None)` — a provenance write can
never fail the run it describes). One live call site:
`LoopController.run_one_cycle` (the `KG_LOOP` daemon tick) passes its own
engine, so every `golden-loop` stage transition mirrors live.
`list_durable_execution_units()` / `graph_durable(action="status")` already
query across all four labels — a future writer for `:AnalyticsJob` /
`:StatechartInstance` / `:SagaCoordination` needs no change to the read side,
only a writer.

## Tool surface (DE2)

One MCP tool, `graph_durable` (`agent_utilities/mcp/tools/durable_tools.py`,
action-routed, registered in `kg_server.py` alongside `graph_jobs`), fronting
`agent_utilities/orchestration/durable_tool_surface.py`. Registered into
`ACTION_TOOL_ROUTES` exactly like `graph_jobs`, so `kg_server`'s generic REST
mount loop serves `POST /api/graph/durable` for free — "Two surfaces by
default" from day one, not a follow-on gap.

| Action | `CallShape` | Routed backend (DE0) | Backed today? |
|---|---|---|---|
| `run` | `Run` | `PythonDurableRun` (`DurableRun`) | **Yes** |
| `sleep` | `Sleep` | `Jobs` (DE0's own routing) | **Yes, but on a different backend** — see gap below |
| `status` | — (DE1 read path) | all four (mirrored today: `DurableRun` only) | **Yes** |
| `state_get` / `state_set` | `StateGet`/`StateSet` | `Statechart` | **No** |
| `call` | `Call` | `MutationStoreSaga` | **No** |

`state_get`/`state_set`/`call` raise `DurableCallNotBacked` with the exact
reason (named below) rather than silently no-opping or routing to the wrong
backend — the same "fail closed, never a degraded success" discipline this
repo's other gates use.

## DE3 — continuation ergonomics (narrowed, not closed)

`DurableRun.auto_step()` (a decorator) removes the hand-named-step burden for
the shape of workload this repo actually has (agent-loop boundaries: one tool
call, one LLM turn, called repeatedly in a loop) — the step name is derived
from the wrapped callable's `__qualname__` (or an explicit `name=`) plus a
per-run auto-incrementing call sequence. This is **not** restate's
general-purpose bytecode/coroutine replay of arbitrary imperative code at
every `await` point — it only removes the naming burden from the existing
named-step model. Caveat: the `__qualname__`-derived default label is only
resume-stable for an ordinary module-/class-level function; pass `name=`
explicitly for a locally-defined closure.

`graph_durable(action="checkpoint")` / `durable_tool_surface.durable_checkpoint`
exposes `auto_step` as a stateless MCP/REST action — with its own honestly-
documented boundary: each call reconstructs a fresh `DurableRun`, so the
in-memory per-run sequence counter resets every call. Two `checkpoint` calls
with the same `session_id`+`step_label` resolve to the SAME auto-sequenced
step and the second REPLAYS the first result rather than recording a new one
(`DurableRun.step`'s own idempotency-key contract, working as designed).
`auto_step`'s actual "no hand-picked name" value is realized when it drives a
loop on ONE long-lived `DurableRun` in-process — not across independently
reconstructed stateless calls with a repeated label.

## DE5 — deployment/definition-version pinning (closed for `DurableRun`)

`DurableRun(session_id, definition_version=...)`: a fresh run stores its
`definition_version` in the pointer checkpoint; a resume with a **different**
non-empty `definition_version` raises `DurableRunVersionMismatch` — fails
loud, names both versions, never silently resumes under different semantics.
Omitting `definition_version` on either side skips enforcement (an opt-in
pin, matching the ontology's own "`None` mirrors absence, never a sentinel"
convention). **Not closed** for `eg-statechart`/`eg-jobs`: `:definitionVersion`
is schema-reserved in the ontology (DE0) but has no Rust-side enforcement —
real, scoped, tracked-not-fabricated gap; a natural pairing with `D-DE7-2`'s
clock-injection finding (below) for a future `eg-statechart`/`eg-jobs` lane.

## DE6 — durable timer-to-continuation binding (narrowed, not closed)

`DurableRun.sleep_until(name, wake_at_ms)` binds a durable deadline to exactly
one run's exactly one named step: `False` (durably checkpointed `WAITING`)
before the deadline, `True` (durably completed, idempotent replay after) once
it passes. This is **poll-based**, not push-based — there is no wake
callback; a caller's own retry loop (e.g. the next `KG_LOOP` tick) re-checks a
durably-remembered deadline without ever losing which one it was waiting on.
`graph_durable(action="sleep")` backs `CallShape::Sleep` with this
`DurableRun` mechanism rather than DE0's own routing table (`Sleep` →
`WorkShape::AsyncCheckpointedWork` → `DurableBackendKind::Jobs`) — a
**documented, honest substitution**: `eg-jobs`' cron/interval triggers have no
Python-reachable wire-protocol surface either, so this backs the externally
observed contract (one durable deadline, one in-flight execution) on the one
backend Python can actually reach today. A caller that needs `eg-jobs`'s own
tick-driven semantics should keep using that backend directly, not this tool.

## DE7 — replay-determinism test discipline (closed)

Property/fuzz tests proving same-input-same-seed replays bit-identically
across a simulated crash, red/green/red-again evidence per subsystem — landed
for `eg-jobs`/`eg-statechart` (`epistemic-graph` merge `24874de`) and
`DurableRun` (`tests/orchestration/test_durable_run_replay_determinism.py`).

**Genuine finding, tracked as `D-DE7-2` (still open):** `eg-statechart`'s
`instantiate`/`send_event` read the real wall clock internally with no
caller-supplied override, unlike `eg-jobs`'s fenced transition methods (which
all take an explicit `now_ms: i64`). Concrete consequence: two wall-clock-
separated deliveries of the identical event can never be recognized as the
same batch by `eg-mutation-store`'s idempotent-replay path — `eg-jobs` gets
this essentially for free; `eg-statechart` cannot, today. A real,
source-cited instance of exactly the restate-gap class (deterministic
replay). Not fixed by DE7 (test-discipline-only charter); candidate fix:
thread an explicit `now_ms: i64` through both methods, mirroring `eg-jobs`,
folding naturally into a `D-DE7-2`/`eg-statechart` clock-injection lane or
DE5's enforcement extension above.

## DE4 — cross-cluster keyed ordering (deferred, not this program's job)

`eg-statechart`'s own module doc already flags single-node OCC-correctness
(not cross-cluster-ordered) as a deferred item (`crates/eg-statechart/src/lib.rs:54-57`).
This program does **not** fork a parallel multi-Raft track — execution rides
the already-open `engine-sql-graph-program`'s multi-Raft groups landing.

## Feature-by-feature parity sweep vs. `restate`

| # | restate capability | Disposition |
|---|---|---|
| 1 | Durable promises/futures + byte-for-byte journal replay | **Matched, narrower.** Named-step (`DurableRun.step`) rather than any-await-point; DE3's `auto_step` narrows the naming burden without copying restate's bytecode/coroutine mechanism. |
| 2 | Virtual objects: keyed single-writer, durable FIFO lock from log order | **Matched (single-node); cross-cluster deferred.** `eg-statechart` OCC is single-node correct; DE4 → `engine-sql-graph-program`. |
| 3 | Durable RPC + idempotency | **Not a gap.** `MutationBatch.idempotency_key` ledger + `eg-jobs::AnalyticsJob::result_ref()`. |
| 4 | Journaling/replay for exactly-once effects | **Matched, narrower.** `eg-mutation-store` group-commit journal; scoped to declared mutation/job/statechart types, not an arbitrary "wrap any closure" primitive uniformly exposed to Python callers. |
| 5 | Suspend/resume of long-running handlers | **Matched, ergonomics gap narrowed (DE3).** Explicit step/checkpoint design vs. restate's zero-design-cost automatic suspend/resume. |
| 6 | Saga/compensation | **We are ahead.** `eg-mutation-store::prepare_saga`/`commit_saga` — restate ships none. |
| 7 | Timers (durable, log-committed firing) | **Matched for `DurableRun` (DE6), narrower for `eg-jobs`.** Poll-based single-execution binding vs. restate's push-based wake. |
| 8 | Per-keyed-object state store, journaled mutations | **Not a gap** beyond #2's cross-cluster item. |
| 9 | Deployment/version pinning for replay determinism | **Closed for `DurableRun` (DE5). Open for `eg-statechart`/`eg-jobs`** — schema-reserved, not enforced. |
| 10 | Two consensus protocols split by access pattern | **Not a gap for this program** — `engine-sql-graph-program` territory. |
| 11 | Backup/PITR, message broker, OTel tracing | **We are ahead.** `src/server/persistence/backup.rs` (online MVCC snapshot), native broker (AMQP/MQTT/STOMP), `EPISTEMIC_GRAPH_OTLP_ENDPOINT` — restate has none of these as durability-story features. |
| 12 | Polyglot durable-RPC control plane for arbitrary external microservices | **Deliberately out of scope, named honestly.** Not our thesis — we are the durable substrate for our own agents/tools/statecharts/jobs, governed by one ontology. |
| 13 | Execution provenance / observability tied to durability state | **We are ahead in kind, unification in progress.** `RunTrace.graph_checkpoint_ids`/`graph_resume_supported` already bind traces to checkpoint state; DE1's `:produced`/`:producedBy` completes the reverse edge for `DurableRun`; the other three backends' reverse edges await their own KG writers. |

### What is still genuinely missing (stated plainly, not narrowed away)

1. **No Python-reachable wire-protocol surface for `eg-statechart` or
   `eg-mutation-store`'s saga coordinator.** `protocols/epistemic_operations/_generated.py`
   carries an `AnalyticsJob` DTO but no live query/mutate method for it, and no
   DTO at all for a statechart instance or a saga (`Method::Statechart{...}`,
   `Method::Saga{...}` do not exist). This is why `graph_durable`'s
   `state_get`/`state_set`/`call` actions are not backed, and why DE1's KG
   mirror is implemented for `:DurableRun` only. Closing it is new engine
   wire-protocol work (a Rust `Method` variant + generated DTO + Python
   caller), out of scope for this lane.
2. **`eg-statechart`/`eg-jobs` have no `:definitionVersion` enforcement**
   (DE5 schema-only there) and `eg-statechart` has no caller-injectable clock
   (`D-DE7-2`, open).
3. **DE1's mirror-on-write covers `:DurableRun` only** — `:AnalyticsJob` /
   `:StatechartInstance` / `:SagaCoordination` are schema-complete
   (queryable the moment a writer exists) but have no writer yet, blocked on
   gap #1 above.

`graph_durable`'s REST twin (`POST /api/graph/durable`) is already live —
registered into `ACTION_TOOL_ROUTES` exactly like `graph_jobs`, so this is
NOT a gap; verified via `scripts/check_surface_parity.py`'s tool<->route
drift check (zero new violations) and the generated `_graphos_action_manifest.py`
carrying all six `graph_durable_*` verbose ops.

None of these three narrow the headline finding: the durability *mechanisms*
were already real before this program started. What this program added is
the unified concept family, the one small tool surface, the version-pinning
and timer-binding narrowing for the one fully-Python-reachable backend, and —
critically — an honest, source-cited map of exactly where the remaining seams
are, so the next lane starts from a known gap instead of rediscovering it.
