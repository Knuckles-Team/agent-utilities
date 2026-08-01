# ACL Registration Convergence — Write-Time Registration + Read-Time Fallback

> **Pillar 2 — Epistemic Knowledge Graph** · Concepts: `KG-2.6` (Company Brain),
> `AU-KG.backend.company-brain-write-guard`, `AU-KG.compute.data-is-private-its`

## The defect

`secured_reads.permit()` is default-deny for any node with no registered ACL —
`DataLevelPermissions.check_permission` (`company_brain.py`)
hardcodes `if acl is None: return allowed=False, reason="No ACL defined — default
deny"`, with no owner/tenant fallback and no privileged-actor exception. That is
correct, intended behaviour for a security boundary.

The bug is the other half: **nothing on the generic write path ever registered an
ACL.** `classify_node()`/`set_acl()` were called nowhere in production code except
two narrow, independently-discovered call sites (`sessions.py::_persist_goal`,
`engine_ingestion.py::ingest_mcp_server`, both from `fix/green-slice-7`, not yet on
`main` at the time of this convergence). Consequence: data was writable but
**permanently unreadable — even by its own creator.**

Four lanes hit this independently in one night and wrote four partial,
overlapping fixes. This document is the converged, canonical design; the
sections below say precisely what each fix contributed, what is superseded,
and what remains open.

## The four inputs

| # | Branch | Contribution | Status at convergence time |
|---|---|---|---|
| 1 | `fix/green-slice-3` | Read-time synthesis: `_hydrate_missing_acls` falls back to a private, owner-readable ACL from `_owner_id`/`tenant_id` when no `external_access` descriptor exists. Also proposed write-time governance stamping at `GraphComputeEngine.add_node` and `IntelligenceGraphEngine._upsert_node`, LadybugDB `_GOVERNANCE_COLUMNS` schema declarations, and a Ladybug `execute_read` transient-close-ordering fix. | Not yet on `main`. |
| 2 | `fix/w2-w2-tail` | Right-store read: `_durable_access_rows` reads via `active.backend.execute_read(...)` instead of `active.graph_compute` (which is only the same store as the backend in the single-process `EpistemicGraphBackend` topology). | **Already merged to `main`** (commit `063acc07`). |
| 3 | `fix/green-slice-7` | Write-time per-call-site registration: explicit `classify_node(INTERNAL, data_owner=...)` in `_persist_goal`, and a `_classify_mcp_node()` helper (`PUBLIC`) in `ingest_mcp_server`. Its own deferred item (D-GS27-5) flags this as narrow and not swept. | Not yet on `main`. |
| 4 | `fix/green-slice-3`'s D-GS3-2 | Names a second split-store variant: nodes whose write only reaches `_upsert_node` (backend) and never `graph_compute` (`delete_memory`, `ingest_mcp_server`, `ingest_a2a_agent_card`, `ingest_agent_skill`). | Resolved as a side effect of #2 (see below) — the read side no longer depends on `graph_compute` having a copy at all. |

**Verifying claim #2 vs #1**: they are complementary, not competing, and the
branch author's framing is correct. #2 fixes "an ACL exists but the hydrator
read the wrong store to find it" (a node the write path already wrote
`external_access`/`classification` onto, in the durable backend, was
invisible because the read went to a different, unwritten compute
scratchpad). #1 fixes "no durable ACL material exists for this node at all"
(first-party nodes never carry `external_access`). Both must be true for a
first-party write to become readable: the data has to actually be in the
store the read queries (#2), and that data has to carry ACL material for the
read to synthesize from (#1 — and, per this convergence, the write side that
actually stamps that material, which #1 assumed but did not add).

## Canonical design: defence in depth, not one layer

**Register an ACL at write time, at the write chokepoints every node write
funnels through, so correctness does not depend on lazy hydration — and keep
a read-time fallback for nodes written before this fix, or through a path
that still bypasses every chokepoint.**

This is deliberately **not** a single-layer fix:

- **Write-time only** (à la slice-7's per-call-site `classify_node()` calls)
  does not scale: `classify_node()`/`set_acl()` are called nowhere in the
  generic write path, and the grep for `add_node`/`_upsert_node` call sites in
  this sweep turned up **~150 files** calling some form of `add_node` and
  **~55 files** calling `_upsert_node` directly. Sprinkling explicit
  `classify_node()` calls at each one is unreviewable, will drift the moment a
  new call site is added, and is exactly what D-GS27-5 (slice-7's own deferred
  item) already concluded: *"classify_node() is called nowhere else in the
  entire agent_utilities production tree... A systemic fix belongs in
  BrainGuardedBackend's write guard... or IntelligenceGraphEngine.add_node
  itself."* Per-call-site classification remains the *right* tool for the
  handful of cases where the call site has semantic knowledge a bare node
  label cannot express (see Classification Policy below) — but as the
  *primary* mechanism it is the wrong shape.
- **Read-time-only** (à la slice-3's original `_hydrate_missing_acls`
  fallback) leaves every write until the next read cold-starting the ACL from
  whatever happens to be in `_owner_id`/`tenant_id`/`classification` — but
  nothing durable stamped `classification` at all before this convergence, so
  the fallback had nothing to synthesize besides a hardcoded guess. It is also
  silently unable to recover once `_owner_id` is missing (system/background
  writes) — there is no way to reconstruct authorial intent after the fact.
  Read-time synthesis is the right **safety net**, not the right **primary**
  mechanism.

Together: the write-time stamp guarantees every future write is self-describing
(carries its own owner/tenant/classification durably, so a *future* process
restart or a different reader can reconstruct the ACL without guessing), and
the read-time fallback guarantees the millions of rows written before this fix
shipped — and any path this sweep didn't find — degrade to "owner can read it,
no one else gains anything" instead of staying permanently denied.

## Mechanism

### Write time — `tenant_sharing.stamp_classification()`

A new sibling to the existing `tenant_sharing.stamp_ownership()` (which stamps
`tenant_id`/`_owner_id`/`_shared_scope`). `stamp_classification(properties,
label)` stamps a durable `classification` property in place:

- `label` in a short, explicit `PUBLIC_CATALOG_LABELS` set (`ToolMetadata`,
  `CallableResource`) → `PUBLIC`.
- everything else → `CONFIDENTIAL` (private-by-default; combined with
  `stamp_ownership`'s `_owner_id`, this is what makes the node readable by its
  own creator — the exact defect being fixed — without granting anyone else
  anything new).

Existing `classification` values are never overwritten (`setdefault`
semantics) — a re-write or an explicit reclassification is never silently
reset.

### Write-time chokepoints (four found and fixed)

Both `stamp_ownership` and `stamp_classification` are now called, best-effort
(caught `PermissionError` for unauthenticated/system/background writes —
identical no-op to pre-fix behaviour for that class of write), at every write
seam this sweep found that reaches durable storage without going through
another already-stamped seam:

1. **`IntelligenceGraphEngine._upsert_node`** (`engine.py`) — the seam ~55
   call sites across the codebase call directly (`engine_memory.py`,
   `engine_ingestion.py`, `engine_ahe.py`, `core/registry/kg_adapter.py`,
   `security/policy_ingestor.py`, `security/rule_ingestor.py`, …), and that
   `IntelligenceGraphEngine.add_node()` itself calls for its Tier-1
   (backend-first) write.
2. **`GraphComputeEngine.add_node`** (`graph_compute.py`) — `self.graph` and
   `self.graph_compute` name the **same native graph authority** in the
   standard (`EpistemicGraphBackend`) topology (the class's own docstring);
   dozens of ingestion/pipeline modules (`kb/ingestion.py`, `kb/x_ingestion.py`,
   `pipeline/document_*.py`, `security/*_ingestor.py`, `kb/entity_claim_extractor.py`,
   …) call `self.graph.add_node(...)` directly, bypassing `_upsert_node`
   entirely.
3. **`BrainGuardedBackend.add_node`/`_add_node_level`** (`brain_guarded_backend.py`)
   — the transparent proxy that "activates the dormant Company Brain on the
   write path without editing the dozens of writers" (its own docstring);
   already stamped `_owner_id`/`tenant_id` via `stamp_ownership` but never
   `classification`. Reached directly by `adaptation/feedback.py`,
   `enrichment/pipeline.py`, `core/engine_tasks.py`, and others that call
   `self.backend.add_node(...)`.
4. **`_BatchedBackend.add_node`** (`enrichment/pipeline.py`) — the buffered
   bulk-RPC path for the KG-2.9g code-symbol ingest (tens of thousands of
   nodes per repo); flushes through `graph.bulk_mutate`/`batch_update`,
   bypassing all three chokepoints above.

Also required (without these, the stamped properties silently vanished or the
write itself errored on the default Ladybug/Kuzu backend):

- `materialization._GOVERNANCE_COLUMNS` / `schema_valid_keys` — declares
  `tenant_id`/`_owner_id`/`_shared_scope`/`classification` as valid columns so
  a schema-backed backend's SET-clause filter keeps them instead of folding
  them into the `metadata` JSON catch-all.
- `ladybug_backend.py` `_GOVERNANCE_COLUMNS` — Kuzu is strict-schema; an
  undeclared column reference raises a Binder exception rather than silently
  dropping the value. Declared on every node table (initial `CREATE` +
  best-effort `ALTER TABLE` migration for pre-existing tables) and on the
  generic auto-created table for unknown labels.
- `ladybug_backend.py::execute_read` transient-close-ordering fix — the
  transient/test-mode connection close ran before the lazy `rows_as_dict()`
  read, raising `RuntimeError: Query result is closed` on every transient-mode
  `execute_read` call. `secured_reads._durable_access_rows` (the read-time
  fallback below) depends on `execute_read` — this fix is a load-bearing
  dependency for the fallback to function at all in the test/transient
  topology, not a tangential cleanup.
- `IntelligenceGraphEngine._upsert_node` also gained `prepared.setdefault("id",
  node_id)` — the raw-Cypher `MERGE (n:{label} {id: $id})` branch binds `$id`
  from the params dict; a caller whose `data` never included an `"id"` key
  (e.g. `skill_workflow_ingest.ingest_runnable_skill`) sent `$id` in the query
  text with nothing to bind it to, which the native parser rejects outright.

### Read time — `secured_reads._hydrate_missing_acls`

Generalized (not left as slice-3's hardcoded-`CONFIDENTIAL` version) to
respect whatever the write-time stamp actually recorded:

1. `external_access` present as a dict → unchanged connector-sourced path
   (`sync_access`).
2. No `external_access`, but a durable `classification`/`_owner_id` (now added
   to the `_durable_access_rows` projection) → classification `PUBLIC`
   registers regardless of owner; otherwise an owner-present node registers a
   `NodeACL` with the stamped classification (falling back to `CONFIDENTIAL`
   for legacy data written before this fix, matching slice-3's original,
   narrower intent) and `data_owner=<owner_id>`.
3. Neither `external_access` nor an owner nor `PUBLIC` → **nothing is
   synthesized; the node stays denied.** This is unchanged, fail-closed
   behaviour — unowned/system data with no explicit grant was never made
   readable by anyone, and still isn't.

## Classification policy — deliberate, not blanket

The mandate is explicit that classification must be deliberate per node
class, not a single default slapped on everything. The policy actually
implemented:

- **A short, evidence-backed `PUBLIC` list** (`ToolMetadata`, `CallableResource`)
  for labels that are, by their own design intent, a discoverable capability
  catalog — the self-describing tool registry
  (`CONCEPT:AU-ECO.toolkit.self-describing-registry`). This was slice-7's
  reasoning for `ingest_mcp_server`'s two labels; the chokepoint generalizes
  it to *every* writer of those labels for free — `kg_adapter.register_callable_resource`
  ("discoverable through the same graph query" — its own docstring),
  `ingest_a2a_agent_card`, and `skill_workflow_ingest.ingest_runnable_skill`'s
  `CallableResource` node are now covered without any code at those sites.
- **A conservative `CONFIDENTIAL` + owner default for everything else** — not
  because every other node class is equally sensitive, but because the
  chokepoint cannot know, from a bare label alone, whether a `Memory` node
  holds a shopping list or a legal privilege review. Defaulting to "readable
  by its creator only" is the *narrowest* fix that resolves the reported
  defect (data unreadable by its own creator) without guessing at broader
  visibility. This is the same "data is private, its owner" default the
  sibling `stamp_ownership` already applies — the two stamps are deliberately
  symmetric.
- **Explicit per-call-site refinement stays available and is used once**:
  `sessions.py::_persist_goal` calls `classify_node(INTERNAL, data_owner=...)`
  after the chokepoint has already run, because the label it writes under
  (`Concept`) is overloaded across the codebase and only the call site knows
  "this particular Concept is a user's goal." This does not change *whether*
  the actor can read it (the chokepoint default already guarantees that) —
  only the classification label used for entailment-propagation strictness
  ordering (`secured_reads.inherit_inferred_acl`) and RESTRICTED-tier audit
  posture. Per-call-site `classify_node()` remains the right tool for this
  narrow class of "the label lies, the call site knows better" cases — it is
  wrong only as the *primary*, unswept mechanism.

Why not blanket-PUBLIC or blanket-INTERNAL: `check_permission`'s only
special-cased classification is `PUBLIC` (unconditional read allow); every
other classification (`INTERNAL`/`CONFIDENTIAL`/`RESTRICTED`) is functionally
identical in the current implementation (owner/actor/role gated) except for
`RESTRICTED`'s `audit_on_access` flag and `inherit_inferred_acl`'s strictness
ordering. Given that, blanket-assigning `PUBLIC` to fix the defect would be a
straightforward tenant-isolation-preserving-but-actor-isolation-breaking data
exposure vulnerability (every actor in the tenant could read every node), and
is exactly what the task's "line you must not cross" forbids.

## What this does **not** touch (explicitly out of scope)

- **`DataLevelPermissions.check_permission`/`secured_reads.permit()` are
  unmodified.** Still hardcodes `if acl is None: return allowed=False`. No
  owner/tenant fallback, no privileged-actor bypass was added there — the
  fallback lives entirely in `_hydrate_missing_acls`, which only ever
  *registers* an ACL (through the same `classify_node()` used everywhere
  else); it never changes how a registered ACL is evaluated.
- **`company_brain.TenancyManager.scope_cypher_query`** (the tenant-isolation
  predicate, `KG-2.6`) is untouched. Both `fix/green-slice-3` and
  `fix/green-slice-7` independently proposed changes here (a `tenant_id IS
  NULL OR tenant_id = ''` commons-fallback, matching
  `PostgreSQLBackend.rls_statements`'s documented RLS policy) — that is a real,
  confirmed inconsistency between the Cypher-level scoping and the SQL RLS
  layer, but it is a different security mechanism (`scope()`, tenant
  isolation) from the one in this mandate (`permit()`, per-node ACL), already
  owned by `fix/w2-tenant-predicate` (merged to `main`), and risked an
  unreviewed conflict with that lane under time pressure. Filed as
  `D-ACL-3` (see Deferred).
- **The external-source materialization path** (`enrichment/registry.write_batch`
  → `core/materialization.write_entities`, used by `materialize_source`/
  `ingest_graph_slice` for the whole connector fleet) stamps `stamp_source`
  (provenance) but neither `stamp_ownership` nor `stamp_classification`.
  Connector-sourced entities largely carry their own `external_access`
  descriptor stamped at the connector layer (verified: `rest.py`, `ard.py`,
  `web.py`, `filesystem.py`, `database.py`, `mcp_tool.py`, … all stamp
  `external_access=`), which is the pre-existing, already-working branch of
  `_hydrate_missing_acls` — but `write_batch`'s own docstring says internal,
  non-connector callers ("finance/synthesize" batches) pass `source=None` and
  stay untagged, with no `external_access` and now no `classification`/`_owner_id`
  fallback either. This is a materially different, fleet-wide surface
  (`agent-utilities-source-integration`) that deserves its own dedicated audit
  rather than an unverified blind stamp under this lane's time budget. Filed
  as `D-ACL-4`.

## Cross-tenant safety

The fallback is gated by the pre-existing tenant check in
`_hydrate_missing_acls` (`if str(properties.get("tenant_id") or "") !=
actor.tenant_id: continue`) — unchanged. A node stamped with a different
tenant's id is never even considered for ACL synthesis, regardless of
`classification`/`_owner_id`. See
`tests/unit/knowledge_graph/test_secured_reads.py::test_owner_fallback_widens_only_the_owner_not_other_tenant_or_other_actor`
for the proof: the owner gains read access; a same-tenant non-owner and a
cross-tenant actor (even one that happens to share the same `actor_id`) are
both still denied.
