# Epistemic OS Hardening (Phase 0–2 + Exceed X-series, AU 1.21.0)

This is the **audit catalog** for what `agent-utilities` (AU) actually shipped in the
"Epistemic OS Hardening" program — the AU half of a coordinated two-repo release with
`epistemic-graph` (EG). It exists so a future static audit does not have to re-derive
"what changed" from commit archaeology: every capability below is verified directly
against `agent_utilities/` source at the commit this doc was written against
(`89da1265` + the AU-owned pre-commit-gate closeout commits `93270411`/`071f6d86` —
the version bump itself is the one remaining release step, so `pyproject.toml` still
reads `1.20.0` at the time of writing; every capability described here is already in
the tree).

**Companion documents** (read these first for the summary; this doc goes deeper):

- [`CHANGELOG.md`](../../CHANGELOG.md) `## [1.21.0]` — the shipped-reality summary,
  including what it explicitly **declines** to claim.
- [`docs/capabilities.md`](../capabilities.md) — the user-facing capability tour
  (this doc is its detailed, code-anchored backing reference for one program).
- `plans/program-tracker-2026-07-10-epistemic-os-hardening.md` (workspace root) — the
  full workstream ledger (IDs, branches, commits, every closed follow-up).
- `plans/epistemic-os-evolution-roadmap-2026-07-11.md` (workspace root) — the
  **forward-looking** half: where this architecture goes next (the "seams to tie"
  that follow this program).

**Scope discipline (read this before trusting any claim below).** The original
13-workstream plan attributed several Phase-3/exceed items to AU that actually landed
**on the `epistemic-graph` side** (the Rust `eg-epistemic` crate): calibrated causal
inference (do-calculus intervene/counterfactual), proof redaction
(`explain_belief_redacted`), and bitemporal epistemic-status ops. **Those are EG's
1.21.0-cycle CHANGELOG, not this one** — they are cross-linked where AU *consumes*
them (X-7's evidence-quality columns) but not re-claimed as AU work here.
`belief_revision`/the epistemic Claim/Evidence/BeliefState substrate itself is
**earlier** work — it shipped in the 1.12.0 **epistemic-substrate** release, not this
program; this doc only covers what changed *in this cycle* on top of that foundation.
Two items from the original plan (an Arrow-Flight external heavy-compute path and an
analytics-job **scheduler** beyond the registries below) are **not in this codebase**
and are not claimed — the CHANGELOG already states this explicitly and this doc
repeats it rather than silently dropping it.

## How to read each entry

Every capability is presented as: **what it is** → **code anchor** (file +
class/function, not just a directory) → **CONCEPT id** (the registry key in
`docs/concepts.yaml`) → **default posture** (on by default vs. opt-in, and the exact
env var if one gates it) → **the two surfaces** (MCP tool + action, REST route) →
**honest limitations**. Where a capability has **no** second surface, that is stated
plainly, not implied by omission — per AGENTS.md's *"Two surfaces by default... there
is no third option and no 'internal-only' exemption."*

---

## 1. Phase 0 recap — the trustworthy core this program builds on

Phase 0 (AU 1.20.0, the prior release) is not re-described in full here, but every
Phase-1/2/X item below *depends* on it, so the load-bearing pieces are named with
their exact anchors:

| Capability | Code anchor | CONCEPT | Default posture |
|---|---|---|---|
| `GraphSession` — one explicit actor/tenant/scope/graph/trace/policy-version currency | `knowledge_graph/core/session.py::GraphSession` (`from_ambient`, `use_session`, `require_scope`) | — (module docstring: AU-P0-1) | Always constructible; enforcement (`require_scope`) is opt-in per caller — a session with no scopes recorded denies nothing until a caller actually checks |
| Native-Cypher authority, never silent `[]` | `knowledge_graph/backends/epistemic_graph_backend.py::CypherEngineError`; 11 `NotImplementedError` sites (grep `ABSTRACT-OK.*AU-P0-2`) | `AU-KG.query.vendor-agnostic-traversal`, `AU-KG.query.object-graph-mapper` | Always on — an unsupported Cypher shape raises, it never degrades to an empty result |
| Engine-native `WorkItem`/claim, fencing that **fails closed** on the engine-native path | `orchestration/agent_dispatch_worker.py::_fence_still_valid` (branches on `claim["_claim_backend"]`) | (AU-P0-3, hardened this cycle: L15) | `AGENT_CLAIM_BACKEND` = `kg` (default, best-effort fail-open fencing) \| `engine` \| `workitem` |
| Fail-closed connectors (quarantine unknown ACL, manifest gate, tombstone guard) | `protocols/source_connectors/base.py::ExternalAccess.quarantined`/`default_external_access`; `knowledge_graph/ontology/connector_manifest_gate.py`; `knowledge_graph/core/source_sync.py::_reconcile` | `AU-P0-4` | `CONNECTOR_DEFAULT_PUBLIC=false` (default), `CONNECTOR_MANIFEST_REQUIRE_ENTERPRISE` (opt-in allowlist), `SOURCE_SYNC_ALLOW_EMPTY_TOMBSTONE` (opt-in allowlist) |
| Tenant RLS end-to-end, fail-**closed** cross-tenant GUC fix | `knowledge_graph/backends/postgresql_backend.py::set_request_tenant` (raises on a failed `SET LOCAL` for a *non-empty* tenant rather than serving a stale GUC) | (AU-P0-5) | Always on for a non-empty tenant; fail-open only for the unscoped/system baseline (`tenant_id == ""`) |
| Scoped low-level engine tools (`engine_<domain>`) + bounded client pool | `mcp/tools/engine_tools.py::ADMIN_DOMAINS`/`ENGINE_ADMIN_SCOPE`/`_enforce_admin_scope`/`_client_for` | `AU-P0-6` | `ENGINE_ADMIN_SCOPE = "kg:admin"` required for `tenants`/`resharding`/`consensus`/`rbac`/`admin` domains (fail-closed for any *unclassified future* domain too); pool bound `KG_ENGINE_TOOL_POOL_SIZE` (default `16`, LRU-evicted) |

**Two surfaces for the Phase-0 pieces that are directly operator-facing:** the
`engine_<domain>` tools (one per engine sub-client — `engine_tenants`,
`engine_resharding`, `engine_consensus`, `engine_rbac`, `engine_admin`, plus the 19
normal domains) are each a standalone MCP tool with a `POST /engine/<domain>` REST
twin (`mcp/tools/engine_tools.py::register_engine_tools`, mounted automatically —
every tool gets its REST route registered in the *same* call that registers the MCP
tool, so the two can't drift). `GraphSession`/native-Cypher/fencing/tenant-RLS are
**not** separately surfaced tools — they are cross-cutting currency consumed *by*
every existing `graph_*`/`engine_*` tool, so their "surface" is every entrypoint that
already exists, not a new one.

---

## 2. Phase 1 — one Agent-OS work/identity/ingestion model

### 2.1 `WorkItem` — the one engine-native state machine

**What it is.** Before this workstream, "a unit of work advancing toward done" had
**four** independently-evolved status vocabularies: the `:Task` ingestion queue, the
`:AgentTask` dispatch DAG, Loop/Goal, and the dispatch envelope. `WorkItem` is now the
one versioned lifecycle all of them route through (fully or as a read-only
projection):

```
submitted -> ready -> leased(fencing_token) -> running(heartbeat, attempt)
    -> succeeded(result_ref) | failed(error_ref) | cancelled | dead_letter
```

**Code anchor.** `orchestration/work_item.py::WorkItemStatus` (the `StrEnum`),
`TERMINAL_WORK_ITEM_STATUSES`, `submit_work_item`/`claim_next`/`commit_result`/
`reap_expired_leases`. Every transition is a single
`backend.compare_and_set_node_fields(...)` CAS call — the *same* primitive
`TaskManagerMixin._claim_next_task` and `research.loops.claim_loop` already used, not
a new atomicity mechanism.

**What is migrated vs. shimmed vs. follow-up** (stated explicitly in the module
docstring, not left implicit):

| Subsystem | Status | Bridge |
|---|---|---|
| `:AgentTask` dispatch (`agent_dispatch_worker.py`) | **Migrated**, opt-in | `ensure_agent_task_work_item`/`claim_agent_task_via_work_item`, `AGENT_CLAIM_BACKEND=workitem` |
| `:Task` ingestion queue (`engine_tasks.TaskManagerMixin`) | **Migrated** (closed this cycle, AU-P1-CL) | `ensure_ingest_task_work_item`/`claim_ingest_task_work_item` — lane/admission/fair-rotation *selection* stays on the `:Task` index; the WorkItem shadow is the actual claim/lease/backoff/DLQ authority, created lazily at claim time |
| Team `:TaskNode` (`capabilities.teams.TeamCapability`) | **Migrated** (AU-P1-CL) | `team_task_work_item_id`/`start_team_task_work_item` |
| Loop/Goal (`research/loops.py`) | **Shimmed (read-only)** | `work_item_view_of_loop` projects Loop's native status onto `WorkItemStatus` for observability; `submit_loop`/`claim_loop`/`run_goal_loop` write paths are **unchanged** — a full storage migration is a stated follow-up, not silently done |

**A missing/no-op executor never fakes success.** Continuing the Phase-0 discipline:
a task with no bound executor resolves to `unroutable`/`failed`, never
`completed`/`reward=1.0` — the `WorkItem` state machine carries the same discipline
forward via `dead_letter` once retries are exhausted (never a silent "done").

**Two surfaces.** `WorkItem` itself has **no** dedicated `graph_workitem` MCP
tool/REST route — by design, it is the shared backing state machine *behind*
existing entrypoints: `graph_orchestrate` (`action=dispatch`/`execute_agent`, REST
`/graph/orchestrate`) drives the AgentTask/team paths, `graph_ingest` drives the
ingestion-queue path, and `graph_goals` (REST `/graph/goals`) exposes the Loop
read-only projection. A raw `WorkItem` row is queryable like any other node via
`graph_query action=cypher` (`MATCH (w:WorkItem {id: $id}) RETURN w`, REST
`/graph/query`) — there is no typed getter tool for it.

### 2.2 Partitioned-log `AgentBus` delivery plane

**What it is.** `AgentBus` (`messaging/bus.py`) kept its semantic registry
(`:Agent`/`:Topic`/`:BusSubscription` presence and membership — small, low-churn) as
KG nodes, which is correct — but it *also* wrote one `:BusMessage` graph node **per
recipient** on every `send()` and read the mailbox via a property-scoped `MATCH` scan
on every `receive()`: O(agents) writes, O(history) reads, on a store that is not a
queue. `messaging/bus_log.py` replaces the delivery/wakeup plane with a durable
partitioned log carrying message bodies — real offsets/consumer cursors, a DLQ, and
backpressure via queue depth — while the KG keeps only the registry.

**Code anchor.** `messaging/bus_log.py::resolve_bus_log_backend` (the 3-tier
resolver), `AgentBus.send`/`_send_via_log`/`receive` (`messaging/bus.py`).

**Backend preference order** (`BUS_LOG_BACKENDS = ("engine", "kafka", "graph")`):

1. **engine** — the epistemic-graph engine's native AMQP-style broker (the exact
   surface `graph_broker`/`engine_broker` expose: `declare_exchange`/`declare_queue`/
   `bind`/`publish`/`consume`), a direct exchange + one durable queue per recipient,
   broker owns fan-out — no per-recipient application writes.
2. **kafka** — two keyed topics (`agent_bus_direct`/`agent_bus_topic`), tenant-
   qualified partition keys, one consumer per subscriber with its own committed
   offset. One stated trade-off: a Kafka subscriber's consumer reads the whole keyed
   topic and filters client-side to its own recipient/topic (bounded by traffic
   volume, not by registered-agent count — so it is *not* the O(agents) fan-out this
   workstream removes, just a different bound).
3. **graph** (`None` from the resolver) — the *original* `:BusMessage` graph-node
   model, unchanged, as the zero-infra dev fallback.

Store-and-forward topic replay (a late subscriber gets the backlog via a
per-`(agent,topic)` cursor) and federation relay are unaffected — they sit on top of
whichever delivery backend is active.

**Default posture.** Backend resolution is automatic/best-available — no env flag
required to opt in; `graph` is what you get with nothing configured (byte-identical
to pre-AU-P1-2 behavior).

**Two surfaces.** MCP tool `graph_bus` (`mcp/tools/bus_tools.py::register_bus_tools`)
↔ REST `/graph/bus` (`kg_server.py` `ACTION_TOOL_ROUTES["graph_bus"]`). Note this is
a **different** tool from `graph_message` (which is the curated cross-process
context-handoff store, `/graph/context`) — `graph_bus` is specifically the
agent-to-agent messaging surface this workstream hardened.

### 2.3 Engine-native capability index — filtered ANN + CDC

**What it is.** Capability-aware `designate()` (which entity should handle a task) is
now authoritative in the **engine**, not the in-process Python index.

**Code anchor.** `knowledge_graph/retrieval/engine_capability_search.py::
engine_filtered_search`/`build_capability_filters`. Two engine-native tiers, in
preference order:

1. **Unified filtered plan** — `Scan(label) |> Filter(caps/tenant/policy) |>
   Rank(query) |> Limit` in one costed round-trip (the engine composes the
   capability/tenant/policy restriction with the vector `Rank` leg itself).
2. **Native ANN + bounded post-filter** — when the connected engine has no `query`
   feature (a lean build), falls back to unseeded `semantic_search` kNN, over-fetch,
   and restrict to the filter-matching ids over the bounded candidate pool (never a
   full-graph scan).

Both tiers return `None` when no engine vector surface is reachable, signalling the
caller to fall back further. `knowledge_graph/retrieval/capability_index.py`'s
`CapabilityIndex` (the in-process hnswlib/numpy structure) is now, **by design**, a
*bounded, non-authoritative cache* — LRU-evicted, kept fresh by CDC deltas — used
only as the dev/lean-engine fallback and for reward write-back
(`record_outcome`/`record_capability_outcome`).

**X-4 extends the same file** (see §4.3) — `capability_hierarchy` is an optional
parameter to `engine_filtered_search` that makes the `capabilities` filter
subsumption-aware; passing nothing is byte-identical to pre-X-4 behavior.

**Two surfaces.** Consumed by `graph_orchestrate`'s designation/dispatch path
(`/graph/orchestrate`) and by `graph_search action=discover`/hybrid retrieval
(`/graph/search`) — no standalone `graph_capability_index` tool; this is
infrastructure behind existing routing/retrieval calls, same posture as
`GraphSession`.

### 2.4 `AssetOccurrence` identity over deduped `Blob`

**The bug this fixes.** `MediaStore` used to derive **both** the blob id and the
media-asset id from the content digest — so identical bytes arriving from a second
message or a second tenant silently collapsed onto the **same** node and overwrote
its provenance (source/tenant/owner/ACL/retention/legal-hold all belonged to
whichever write landed last).

**Code anchor.** `knowledge_graph/memory/media_store.py::MediaStore.store_media`
(mints a fresh uuid-keyed `:AssetOccurrence` every call — never digest-derived),
`store_rendition` (derived forms), `migrate_legacy_assets_bulk` (bulk migration CLI
for pre-existing `MediaAsset` nodes, closed this cycle: L29).

**Identity chain:** `Blob(digest)` ← `Rendition` ← `AssetOccurrence` ←
`Message`/`Document`. Only immutable bytes dedup (`:Blob`, content-addressed); every
`store_media()` call mints a **distinct** `:AssetOccurrence` owning its own
source/tenant/owner/acl/event_time/retention/legal_hold/provenance. One cross-modal
ACID txn writes the node + its blob-ref + the occurrence edge together (a reader
never sees a half-written occurrence).

**Two surfaces.** `graph_ingest action=ingest`/`document_process` are the entry
points that call `store_media` under the hood (`/graph/ingest`); there is no
standalone `graph_media` tool — media identity is infrastructure behind the ingest
path, same posture as the capability index above.

**Honest gap (accepted, not silent — L30):** an async OCR/ASR/embedding worker that
would call `record_extraction`/`store_rendition` off the hot path is a **stated
ACCEPT**, not shipped — the seam exists (`store_rendition`), the worker itself is a
Codex-boundary "heavy-model plugin/remote" concern, deliberately out of scope for a
small-footprint framework.

### 2.5 `ChangeEnvelope` — one canonical unit-of-change

**What it is.** Every connector shape (push, MCP pull, fleet-package pull,
CDC/webhook, bulk snapshot) now emits — or is bridged into, via
`from_connector_record` — **one** typed `ChangeEnvelope` carrying:

- **identity/idempotency** — `envelope_id`, `idempotency_key` (deterministic from
  connector+instance+object+version+operation — redelivery is provably a no-op),
  `tenant`.
- **provenance/lineage** — `connector`, `source_instance`, `source_object_id`,
  `source_version`, `schema_version`, `ontology_mapping_version`.
- **bitemporal timestamps** — `event_time`, `valid_time`, `observed_time`.
- **payload** — `operation` (`upsert`/`delete`/`snapshot_complete` — the last is the
  reconcile pass's "authoritative snapshot ends here" marker), typed payload or blob
  ref (exactly one set).
- **governance** — `source_acl` (reuses `ExternalAccess`), `classification` (reuses
  `DataClassification`), `retention`, `legal_hold`.

**Code anchor.** `knowledge_graph/ingestion/change_envelope.py::ChangeEnvelope`,
`Operation`; the atomic consumer is
`knowledge_graph/ingestion/envelope_ingest.py::ingest_envelope(engine, envelope)` —
**one** transaction: validate → resolve identity → write → lineage+checkpoint → CDC →
watermark advance, crash-resume safe. Returns `status` in
`{"success","skipped","rejected","failed"}` and `watermark_advanced: bool` so a
resumed connector can trust `False` to mean "retry this envelope."

**Migration scope, stated honestly.** A first wave of `source_sync` handlers
(`leanix`, `claude_memory`, plus 14 more this release — **22 native total**) is
migrated onto `ingest_envelope`. **7 handlers are a documented ACCEPT, not a silent
gap** (L32): `gitlab`/`archivebox`/`freshrss`/`rss`/`confluence`/`fleet_connectors`/
`fleet` use chunk/embed/gate pipelines where `ingest_envelope`'s one-node-per-call
model would **regress** their multi-node chunking — migrating them would make things
worse, so they stay on the legacy path by design. `source_sync.py`'s own module
docstring enumerates which handlers are which.

**Two surfaces.** No standalone `graph_change_envelope` tool — it is the internal
currency behind `source_connector`/`graph_ingest`'s connector-sync path (`source_sync
action=sync`, REST via `source_connector` → `/connector/source`).

### 2.6 The 12 mandatory signed connector manifests

**What it is.** `agents/<pkg>/connector_manifest.yml` — a per-connector manifest
whose `provenance.integrity.hash` is a **real HMAC-SHA256 signature**
(`knowledge_graph/ontology/ontology_integrity.py::canonical_hash`/`sign`, keyed off
the same signing secret `security/run_token.py` already resolves — not a second crypto
stack), re-verified before every sync. `connector_manifest_gate.py` wires this into
`source_sync` as a **fail-closed** precheck.

**The 12 (`MANDATORY_NAMED_CONNECTOR_SOURCES`, `knowledge_graph/ontology/
connector_manifest_gate.py`):** `jira`, `confluence` (share the `atlassian-agent`
package), `gitlab`, `servicenow`, `leanix`, `langfuse`, `tunnel_manager`,
`microsoft-agent`, `container-manager-mcp`, `documentdb-mcp`, `repository-manager`,
`systems-manager`, `vector-mcp`.

**Unconditional, not opt-in.** Unlike the pre-existing (1.20.0) opt-in
`CONNECTOR_MANIFEST_REQUIRE_ENTERPRISE` allowlist env var, these 12 are baked in —
`enterprise_required_sources()` returns
`MANDATORY_NAMED_CONNECTOR_SOURCES | opted_in`, so an operator does **not** need to
name them for the fail-closed policy to apply. Each of the 12 now has a live,
dispatchable `source_sync` code path — the 5 that previously had none
(`microsoft-agent`/`container-manager-mcp`/`documentdb-mcp`/`repository-manager`/
`systems-manager`, plus `vector-mcp`) got one shared handler,
`source_sync._sync_ops_mcp_connector`, this cycle.

**Honest scope note.** The manifest **files themselves** are committed in each
connector's *own* repo (`agents/<pkg>/connector_manifest.yml`), not in
`agent-utilities` — this repo only ships the **gate** that discovers, compiles, and
verifies them (`resolve_agents_root`/`find_connector_manifest`/`check_manifest_bytes`).
A deployment whose `AGENTS_ROOT`/`WORKSPACE_PATH` doesn't resolve to a real fleet
checkout will not find any manifest, and (per the fail-closed policy) that means
these 12 sources refuse to sync rather than silently proceeding unverified.

**Two surfaces.** The gate is not a user-facing tool — it is a precondition inside
`source_connector`/`source_sync` (`/connector/source`). The CLI sweep companion
(`scripts/check_connector_manifests.py`) is a repo-hygiene script, not a runtime
surface.

---

## 3. Phase 2 — engine-authoritative placement + analytics, measured

### 3.1 AU consumes the engine's placement catalog (DIST-P2-2b)

**What it is.** The engine (`epistemic-graph src/raft/placement.rs`) now owns an
authoritative, versioned `PlacementCatalog` (routing epochs, online move,
virtual/multi-group partitions). AU is a **consumer**, never a second authority.

**Code anchor.** `knowledge_graph/core/placement_catalog.py::resolve_placement` —
three tiers: (1) a short-TTL `(endpoint, epoch)` cache per `(tenant, sub_key)`; (2) on
miss, ask the engine's placement-route op, trying every configured endpoint in
HRW-preference order (the catalog is cluster-wide, any endpoint can answer); (3) the
**static HRW ring** (`shard_topology.shard_endpoint_for`) as the bootstrap/fallback.

**The honest caveat, verbatim from the code and repeated in
`docs/architecture/engine_sharding.md`:** *the currently-shipped engine has no wire
`Method` for `PlacementRoute` yet* — `PlacementCatalog::route` is presently consumed
only **inside** the engine by `MultiRaft`'s own cross-group dispatch. So every real
call from this module fails today, and the designed fallback kicks in exactly as
built: **every current deployment still falls back to the static HRW ring
byte-for-byte.** This ships the AU-side consumer half of a two-sided contract; the
engine-side wire RPC is the stated follow-up (tracked as L44 in the program ledger).
The moment the engine adds the method, AU starts consuming it with **no further
AU-side change** — a catalog-aware `client.placement.route(...)` namespace is tried
first, ahead of a raw `_send` fallback, for a nicer typed client later.

**Default posture.** `PLACEMENT_CATALOG_ENABLED` (default `True`,
`core/config.py:1782`), `PLACEMENT_CATALOG_TTL_S` (default `5.0` seconds,
`core/config.py:1794`). Hermetic under the unit suite: `AGENT_UTILITIES_TESTING`
skips the real network round-trip straight to HRW.

**Two surfaces.** Wired into the *one* engine resolver
(`engine_resolver.resolve_engine`), so every entrypoint gets it "for free" — there is
no dedicated placement tool to call; you observe its effect (or, today, its
no-effect) through whichever graph operation you already run.

### 3.2 Analytics feature/model/experiment registries (L41/INT-P2-1b)

**What it is.** The engine's durable analytics-job plane (`epistemic-graph`
`eg-jobs`) commits every job result as a provenance'd `Claim`/`Evidence` pair stamped
with full `AlgoVersion` lineage (`family`/`algorithm`/`params_digest`/
`code_version`/`env_version`) plus an immutable input-snapshot handle. This module is
the AU-side **queryable registry** that groups committed claims by that lineage —
answering "which jobs produced model X" / "runs of experiment Z" — over the engine's
existing store of record. **It never writes a claim/evidence node** — read-only,
rebuildable, same authority split as the capability-index cache above.

**Code anchor.**
`knowledge_graph/retrieval/analytics_job_registry.py::AnalyticsJobRegistry`
(`refresh_from_engine`, `features()`/`models()`/`experiments()` — three views over
one `AlgoVersionLineage` grouping key). Deliberately **not** folded into
`models.model_registry.ModelRegistry` (the LLM-routing registry) — a different
domain (a trained analytics artifact vs. a chat-model routing entry) that would
conflate two unrelated concepts under one name if merged.

**Two surfaces: none — and this is the officially accepted posture, not an
oversight.** `analytics_job_registry.py` is one of four modules named explicitly in
`scripts/surface_parity_baseline.txt` (the surface-parity gate's accepted-exception
list) with the comment: *"consumed by the eg-jobs durable job plane (INT-P2-1)"* —
i.e. it is internal machinery behind an already-surfaced capability (the engine's own
analytics-job plane), not a standalone feature that needs its own tool.

### 3.3 Workload contract + soak/chaos harness (SCALE-P2-1)

**What it is.** Replaces `docs/scaling/capacity_model.md`'s prior linear-arithmetic
"1M residents" claim with a **machine-readable contract**:
`docs/scaling/workload_contract.yml` — registered agents (1,000,000), concurrent
active sessions (20,000, `= capacity_model.active_agents(1e6, 0.02)`), 5 independent
rate axes (turns/sec, tool-calls/sec, graph-mutations/sec, messages/sec,
token-throughput), tenant count + Zipf skew + one elephant tenant, per-agent
footprint, interactive/background mix, availability + RPO/RTO, and
p50/p95/p99/p99.9 SLO targets. Every numeric axis is anchored to an **existing**
`capacity_model.py` constant (commented `# anchor:` in the YAML) rather than a fresh,
unverified number.

**Code anchor.** `docs/scaling/workload_contract.yml` (the contract),
`docs/scaling/workload_contract.py` (typed loader + `ScaledWorkload.for_scale` — the
SLO targets and per-unit sizes do **not** scale down for a smaller run; population/
rate axes do), `scripts/scale/loadgen.py` (the load generator that drives the
contract against the real `WorkItem` path), `tests/scale/test_workload_contract.py`
(cross-checks the YAML against `capacity_model.py` so the two can't drift).

**Honest limitation — stated in the evolution roadmap, not hidden here:** the
contract, loader, and harness are **built**; the actual **sustained 1M-resident
soak has not been run** to ratify the SLOs. This is the standing
`feedback-test-migrations-at-production-scale` discipline (prove at scale by
running the workload, not by modeling it) — named as Seam 5 of the follow-on
roadmap, not yet closed.

**Two surfaces.** None — this is an offline capacity-planning/testing artifact
(a YAML contract + a `scripts/` load generator + a `tests/scale/` harness), not a
runtime capability with an operator surface.

### 3.4 `ActionPolicy` per-engine caching (L42)

**What it is.** A real ~300ms/`decide()` cost — `get_action_policy()` was rebuilding
`ActionPolicy` (re-parsing the policy YAML from disk) on **every** autonomous-action
decision — fixed by caching one instance per distinct engine identity.

**Code anchor.** `orchestration/action_policy.py::get_action_policy` — `id(engine)`
keyed cache (`_POLICY_CACHE`, bounded at `_POLICY_CACHE_MAX_SIZE = 64`,
LRU-evicted), with an `engine is cached[0]` identity guard so a reused `id()` after
garbage collection can never hand back the wrong engine's policy. Measured **67×**
faster (13.5ms → 0.2ms); decisions are provably identical (the cache changes cost,
not behavior).

**Two surfaces.** Not applicable — this is a pure performance fix inside the
existing `ActionPolicy.decide()` gate every autonomous action already passes
through (`graph_orchestrate`'s dispatch/approval actions, the fleet reconciler,
remediation playbooks — see `docs/architecture/fleet_autonomy.md`).

---

## 4. Exceed track — the Codex X-series

### 4.1 X-2 — Enterprise operations causal graph (`graph_ops_causal`)

**What it is.** Joins entities the connector fleet **already ingests** — Langfuse
trace/generation, GitLab/repository-manager commit/MR, ServiceNow/Atlassian
incident/change, LeanIX capability/owner, container-manager-mcp deployment — into one
causal chain: `trace/generation -> agent/tool/model -> service -> deployment ->
commit/merge-request -> incident/change -> capability/owner -> policy/control/
evidence`. The four analyses (`root_cause_rank`, `blast_radius_analysis`,
`change_risk_score`, `control_evidence_chain`) are **thin compositions over the
causal-reasoning engine already shipped** (`StructuralCausalModel.
get_causal_ancestors`/`get_causal_descendants`, `CausalVerifier`,
`SpuriousnessDetector`, `knowledge_graph/core/formal_reasoning_core.py`) — no new
traversal algorithm.

**Code anchor.**
`knowledge_graph/enrichment/ops_causal_graph.py::root_cause_rank`/
`blast_radius_analysis`/`change_risk_score`/`control_evidence_chain`/
`build_causal_model`/`load_ops_causal_neighborhood`/`materialize_ops_causal_links`;
crosswalk types in `knowledge_graph/ontology/ops_causal_crosswalk.py`. CONCEPT:
`AU-KG.enrichment.ops-causal-graph`.

**Two surfaces.** MCP tool `graph_ops_causal`
(`mcp/tools/ops_causal_tools.py::register_ops_causal_tools`, actions `root_cause` \|
`blast_radius` \| `change_risk` \| `control_evidence` \| `join`) ↔ REST
`POST /ops/causal` (`kg_server.ACTION_TOOL_ROUTES["graph_ops_causal"] =
"/ops/causal"`). The `join` action is the only mutating one — it *materializes*
caller-supplied `links_json` as real graph edges between **already-existing** node
ids via the shared enrichment writer (no new nodes). Every other action can run
offline against explicit `links_json` (test/CI-friendly) or, given an active engine
and a `node_id`, load the causal neighborhood live from the KG. Ships with a
dedicated skill, `agent_utilities/skills/kg-ops-causal/`.

**Limitation.** Two new ontology relationship types were added to support this
(`CHANGE_REQUEST`/`USED_MODEL`, per the program ledger) — the causal chain is only as
complete as what the connector fleet has actually ingested for a given
service/commit/incident; a gap in upstream connector coverage is a gap in the causal
graph, not something this module can infer around.

### 4.2 X-3 — Epistemic mining flywheel (`ClaimFlywheel`)

**What it is.** A governed five-state lifecycle over mining-produced `Claim`s:
`proposed -> validated -> accepted -> deprecated -> retracted` (any pre-terminal
state may be retracted directly). `RETRACTED` is **terminal and sticky** — `propose()`
refuses to re-open a retracted claim, so a rejected mined finding is never silently
re-proposed on a later mining pass over the same (content-addressed) finding id.

**Code anchor.** `knowledge_graph/research/claim_flywheel.py::ClaimFlywheel`
(`propose`/`validate`/`accept`/`reject`/`deprecate`/`retract`/`record_outcome`),
`ClaimLifecycleState`, `LifecycleTransition` (persisted as an append-only
`ClaimLifecycleEvent` node — never a silent mutation of the `Claim` node's own
`status`/`is_verified` fields, which stay exactly what the existing mining pipeline
already set). CONCEPT: `AU-KG.evolution.mining-flywheel`.

**Deliberately a thin overlay, not a second governance stack.** A claim only
reaches `VALIDATED` because `promotion_governance.PromotionGovernanceValidator`
said so, and only reaches `ACCEPTED` because
`orchestration.action_policy.get_action_policy()` independently allowed it — the
**same** gates `loop_controller._run_insight_validation`/`_run_trace_mining`
already used. Outcome feedback persists through
`graph.routing.enrichers.capability_designation.record_capability_outcome` — the
**same** durable contextual-bandit spine AU-P1-3 already uses, never a parallel
reward store. **Two loops closed this cycle:** an accepted ontology-gap claim now
materializes as a real KG edge, and an accepted routing-quality claim's outcome
survives a process restart via that durable bandit.

**Two surfaces: none directly.** `ClaimFlywheel` has no standalone
`graph_claim_flywheel` tool — it is invoked internally by
`knowledge_graph/research/loop_controller.py` (`_run_insight_validation`/
`_run_trace_mining`), which is itself reached through the loop engine
(`graph_loops`, MCP+REST) or `graph_orchestrate action=execute_workflow`'s
loop-cycle path. A caller cannot directly call `flywheel.propose()`/`.accept()`
through either surface today — only *trigger a mining pass* that exercises the
whole state machine internally. This module is **not** listed in
`scripts/surface_parity_baseline.txt`, but that is because the static
surface-parity checker's reachability scan considers it reachable (its importer,
`loop_controller.py`, is itself reachable from the `graph_loops` surface root) —
not because its individual lifecycle methods are each independently invocable.

### 4.3 X-4 — Ontology-driven tool/agent routing

**What it is.** Extends AU-P1-3's engine-native filtered ANN with **ontology
subsumption**: a tool/agent declaring a *narrower* capability now satisfies a request
for the *broader* one (`rdfs:subClassOf`-aware), a versioned `CapabilityDescriptor`
(typed I/O schema, side effects, cost/latency/locality, policy/approval class), and
full eligibility explainability.

**Code anchor.**
`knowledge_graph/ontology/capability_hierarchy.py::CapabilityHierarchy` (the
subsumption index), `knowledge_graph/retrieval/capability_descriptor.py` (the typed
descriptor), `graph/routing/enrichers/capability_routing.py::
route_capability_request`/`explain_routing_eligibility` (the WHY-eligible dict,
computed engine-native-first, falling back to the in-process cache only when the
engine is unreachable). `knowledge_graph/retrieval/engine_capability_search.py`'s
`build_capability_filters`/`engine_filtered_search` take an optional
`capability_hierarchy` parameter — **default `None` is byte-identical to pre-X-4
behavior** (the exact-string `array_contains` filter the engine already used).

**Two surfaces.** No dedicated tool. `capability_hierarchy.py` and
`capability_descriptor.py` are named explicitly in
`scripts/surface_parity_baseline.txt` with the comment *"internal routing structures
behind ontology-driven tool/agent routing (X-4), which IS surfaced via
`graph_orchestrate`"* — i.e. the accepted posture is that X-4 changes *how*
`graph_orchestrate`'s existing dispatch/designation call picks a candidate, not that
it adds a new callable. `explain_routing_eligibility` itself is a plain library
function with **no MCP/REST caller anywhere in this codebase** (confirmed by
searching every `mcp/`/`gateway/` module) — it is reachable only via a direct Python
import today; the "surfaced via `graph_orchestrate`" claim covers the
subsumption-aware *filtering*, not this specific explainability function.

### 4.4 X-5 — Workload-aware placement mining

**What it is.** Mines agent-trace co-occurrence (tenant/tool/entity/modality access
skew, over the real `Episode -[:USED_TOOL]-> ToolCall -[:AFFECTS]-> Entity`
provenance chain — the same schema `trace_pattern_miner` already mines) into typed
`PlacementProposal`s (`shard_split`/`replica`/`cache_prewarm`/`materialized_join`/
`embedding_refresh`/`index_change`), each carrying real mined evidence and an
expected-benefit statement — never fabricated.

**Code anchor.**
`knowledge_graph/research/placement_mining.py::mine_placement_patterns`/
`placement_proposals_from_mining`/`run_canary`/`apply_placement_change`/
`run_placement_mining_cycle`. CONCEPT: `AU-KG.evolution.placement-mining-canary-loop`.

**Pipeline (mirrors `loop_controller._run_trace_mining` exactly — reuses the same
governance spine, not a fourth authority):**

```
mine (associate/anomaly/sequence over trace co-occurrence)
    -> PlacementProposal (typed, evidenced)
    -> Claim (status="proposal", ALWAYS persisted, is_verified=False)
    -> PromotionGovernanceValidator.validate()          (reused as-is)
    -> action_policy.decide(kind="apply_placement_change")   (shipped tier:
       approval_required — deploy/action-policy.default.yml — so nothing
       auto-applies out of the box)
    -> only if allowed: a MEASURED CANARY (apply small, measure SLO delta,
       promote or roll back)
    -> promote reaches the engine's PlacementCatalog admin path
       (ReshardingClient.catalog_assign/catalog_remove via engine_resharding —
       no second placement authority, no new engine RPC)
```

**Honest limitation, verified by grep, not assumed.** `run_placement_mining_cycle`
is exercised by its own unit test suite
(`tests/unit/knowledge_graph/test_placement_mining.py`) and referenced by
`action_policy.py` (for the `apply_placement_change` policy kind), but **nothing in
this codebase calls it automatically** — there is no scheduled loop, cron, or
`loop_controller` wiring that runs a placement-mining pass on its own. It must be
invoked programmatically (or by a future scheduler) today; this is a real gap
against "the mining flywheel is closed-loop," and matches the evolution roadmap's
own Seam 4 ("close the placement control loop") being named as **future**, not
already-done, work.

**Two surfaces.** None — `placement_mining.py` is named in
`scripts/surface_parity_baseline.txt` with the comment *"feeds the already-exposed
ActionPolicy/reconciler surface (X-5) — proposals are reviewed/applied through
action_policy, not a standalone tool."* Given the limitation above, there is
currently no way to *trigger* a placement-mining cycle from either the MCP or REST
surface at all — only to review/approve a proposal that some other process already
produced, via the existing approval-queue surface
(`/api/fleet/approvals`).

### 4.5 X-7 — Policy-aware context compiler

**What it is.** Replaces the ad-hoc "flatten retrieval hits into a text block"
pattern with one selection/assembly layer scoring six axes: relevance (engine's own
ANN score), diversity (greedy MMR over embedding cosine, falling back to lexical
Jaccard), evidence quality (reads the `KnowledgeBatch`-shaped epistemic columns —
`confidence`/`source_refs`/`evidence_refs`/`proof_ids`/`contradiction_ids`/
`policy_labels` — when a result carries them; the `"epistemic:contested"` policy
label flags a disputed claim), bi-temporal freshness (recency decay against
`event_time`/`valid_from`), token cost (`RetrievalBudgetManager`, every drop
logged), and policy (every candidate passes the **same** fine-grained
`ontology.permissioning.enforce` gate the live read path uses — row-level drop,
column-level redaction, no bypass).

**Code anchor.** `knowledge_graph/retrieval/context_compiler.py::ContextCompiler`,
`ContextBundle`, `ContextItem`, `Citation`. CONCEPT:
`AU-KG.retrieval.context-compiler` (this consumes EG's `EPI-P3-1` "universal
epistemic columns" contract — the columns are **populated by the engine**, not
by AU; AU degrades to a neutral prior when a result doesn't carry them, additive
not breaking).

**Output.** A `ContextBundle`: the selected `ContextItem`s (each with its per-axis
scores), a flat `citations` list, a `proof_graph` of supports/contradicts/
alternative-to edges, and a `decisions` log recording every selection/rejection
with its scores — same candidates + same session ⇒ same bundle (a benchmark/audit
can diff two runs deterministically).

**Two surfaces.** MCP tool `graph_search`, `mode="compiled"`
(`mcp/tools/query_tools.py:996` documents the mode; construction at
`query_tools.py:1137-1142`) ↔ REST `/graph/search` (`kg_server.
ACTION_TOOL_ROUTES["graph_search"] = "/graph/search"`). This is the one X-series
item with the cleanest, most direct two-surface exposure — it rides an *existing*,
already-surfaced tool's mode enum rather than needing a new registration at all.

**Limitation, stated in the evolution roadmap as Seam 6:** the compiled bundle is
not yet routed through the KV-cache layering path (LMCache), so "epistemic quality"
and "serving latency" are not yet a single optimized path — that wiring is future
work, not shipped here.

### 4.6 X-8 — Agent digital twin + deterministic replay

**What it is.** A durable, queryable projection over a run's existing `WorkItem`
DAG, `:ToolCall` provenance, and `AgentPolicyDecision` audit — pinning the exact
model/prompt/tool/skill/policy versions + catalog epoch a run executed under.
`replay_twin()` deterministically replays a recorded run (tool calls/model
responses mocked from the record, never re-executed); `counterfactual_replay()`
swaps a policy version (genuinely re-invokes the pure `ActionPolicy.decide()`) or a
model/prompt version (via caller-supplied alternate responses) and reports the
delta; `twin_incident_steps()` is a read-only step-through for incident
investigation.

**Code anchor.** `orchestration/agent_digital_twin.py::AgentDigitalTwin`,
`VersionPins`, `capture_twin`/`capture_twin_from_kg` (build one), `replay_twin`,
`counterfactual_replay`, `twin_incident_steps`, `persist_twin` (best-effort write
of a `:AgentDigitalTwin` KG node + `TWIN_OF`/`REFERENCES` edges — mirrors
`agent_runner._record_execution_trace`'s pattern). CONCEPT:
`AU-ORCH.twin.agent-digital-twin`. Deliberately thin: every piece of provenance it
touches (the WorkItem DAG, `:ToolCall` shape, `AgentPolicyDecisionNode`, the
`run_vcs` event kernel/replay machinery) is **reused, not duplicated**.

**Two surfaces: genuinely none — the most significant honest gap in this
program.** Confirmed by grepping every `agent_utilities/mcp/*` and
`agent_utilities/gateway/*` module for any reference to `agent_digital_twin`,
`capture_twin`, `AgentDigitalTwin`, `replay_twin`, or `counterfactual_replay`: **zero
hits outside the module itself and its own test file.** The pre-existing
`graph_runvcs` MCP tool (`mcp/tools/state_tools.py::graph_runvcs`, action
`replay`) calls the **generic** `run_vcs.replay.replay_run` directly — it does
**not** go through `agent_digital_twin.replay_twin`/`counterfactual_replay`, so a
caller cannot reach the twin's version-pinning, counterfactual-swap, or
incident-step-through behavior from either the MCP or REST surface today. The only
way to exercise X-8 is a direct Python import (or by querying a `persist_twin()`-
written `:AgentDigitalTwin` node with a generic `graph_query` Cypher call, which
surfaces the *data* but none of the replay/counterfactual *behavior*).

This gap is also **invisible to the automated "Two surfaces by default" gate**:
`scripts/check_surface_parity.py`'s `CAPABILITY_PREFIXES` list (the set of packages
the gate scans for unreachable modules) does **not** include
`agent_utilities/orchestration/` at all — only `knowledge_graph/*`,
`protocols/source_connectors/`, `harness/`, `rlm/`, `workflows/`, and `domains/`.
`agent_digital_twin.py` lives in `orchestration/`, so the gate structurally cannot
flag it, whether or not it has a surface. This is a real, verified coverage gap in
the audit tooling itself, worth surfacing to whoever owns the surface-parity gate
next — not just a missing baseline entry.

---

## 5. The two-surfaces map (quick reference)

| Capability | MCP tool (action) | REST route | Standalone or piggybacked? |
|---|---|---|---|
| Low-level engine surface (19 normal + 5 admin domains) | `engine_<domain>` | `POST /engine/<domain>` | Standalone, one per domain |
| `WorkItem` (query only) | `graph_query` (`cypher`) | `/graph/query` | Piggybacked (generic Cypher) |
| `WorkItem` (AgentTask/team dispatch) | `graph_orchestrate` (`dispatch`/`execute_agent`) | `/graph/orchestrate` | Piggybacked |
| `WorkItem` (ingestion queue) | `graph_ingest` | `/graph/ingest` | Piggybacked |
| `WorkItem` (Loop/Goal read view) | `graph_goals` | `/graph/goals` | Piggybacked |
| `AgentBus` / partitioned log | `graph_bus` | `/graph/bus` | Standalone |
| Engine-native capability index | `graph_orchestrate` (designation), `graph_search` (`discover`) | `/graph/orchestrate`, `/graph/search` | Piggybacked |
| `AssetOccurrence` / media identity | `graph_ingest`, `document_process` | `/graph/ingest` | Piggybacked |
| `ChangeEnvelope` ingest | `source_connector` (`sync`) | `/connector/source` | Piggybacked |
| 12 mandatory manifests | (precondition inside `source_connector`) | `/connector/source` | Piggybacked (gate, not a callable) |
| Placement-catalog consumer | (transparent to every entrypoint) | — | None (infra) |
| Analytics job registries | **none** | **none** | Internal machinery (baselined) |
| Workload contract + soak harness | **none** | **none** | Offline artifact, not runtime |
| `ActionPolicy` caching | (transparent) | — | None (perf fix) |
| X-2 ops-causal graph | `graph_ops_causal` (5 actions) | `POST /ops/causal` | **Standalone** |
| X-3 claim flywheel | `graph_loops` (indirect, whole-pass only) | `/graph/loops`-family | Piggybacked, coarse-grained |
| X-4 ontology routing | `graph_orchestrate` (designation, filtering only) | `/graph/orchestrate` | Piggybacked; `explain_routing_eligibility` itself has **no** caller |
| X-5 placement mining | **none** | **none** | Internal machinery (baselined); **no automatic trigger either** |
| X-6 (`ContextCompiler` epistemic columns — EG-side TMS/recompute, consumed not built here) | `graph_search` (`compiled`) | `/graph/search` | Piggybacked |
| X-7 context compiler | `graph_search` (`compiled`) | `/graph/search` | **Standalone mode on an existing tool** |
| X-8 agent digital twin | **none** | **none** | **No surface at all — confirmed gap** |

---

## 6. What this doc deliberately does NOT claim

- **Not AU work this cycle:** calibrated causal do-calculus (do-intervene/
  counterfactual), proof/belief redaction (`explain_belief_redacted`), bitemporal
  epistemic-status ops — all `epistemic-graph` `eg-epistemic` crate, see EG's own
  `docs/architecture/epistemic-os-hardening.md` and its 1.21.0-cycle CHANGELOG.
- **Not new this cycle:** the Claim/Evidence/BeliefState substrate and
  `belief_revision` itself — shipped in the 1.12.0 epistemic-substrate release.
- **Not in this codebase, and the 1.21.0 CHANGELOG already says so:** an
  Arrow-Flight external heavy-compute path, and an analytics-job **scheduler**
  beyond the feature/model/experiment registries in §3.2 (a "which jobs ran"
  index is not a "run more jobs on a schedule" service).
- **Not run yet:** the sustained 1M-resident soak (contract + harness exist, §3.3);
  the placement-mining control loop is not closed (§4.4); the KV-cache × context-
  compiler wiring (§4.5) is not built. All three are named as explicit next steps
  in `plans/epistemic-os-evolution-roadmap-2026-07-11.md`, not silently dropped
  here.

## 7. Other Phase-1 reliability fixes (named for completeness)

Smaller but real fixes shipped alongside the headline items above, from the
CHANGELOG's `### Fixed` section — verified, briefly noted rather than given a full
section since they are not new *capabilities*:

- **`_fence_still_valid` fails closed on the engine-native claim path (L15)** — see
  §1's Phase-0 recap table; this is the specific behavior change, not just a
  cross-reference.
- **Self-ingest telemetry is durable (OBS-P1-1)** — `observability/self_ingest.py`'s
  bounded queue used to silently drop records on backpressure; failed sends now
  requeue in-process (bounded retries) and spill to a durable SQLite-WAL
  `SpillBuffer` (mirrors the existing `GraphOutbox` pattern) instead of vanishing.
  The one remaining loss case (the spill buffer itself saturated) is counted
  (`dropped`) and logged at `ERROR` — never silent.
- **Real OpenTelemetry wiring (L24)** — `observability/__init__.py`'s
  `TelemetryEngine` was a placeholder; now a real Tracer/Meter provider exporting
  via OTLP to the engine collector, opt-in, instrumentation failures never break
  the business path.
- **`USES_SKILL` provenance edge was silently dropped (F8)** — the
  `(RunTrace)-[:USES_SKILL]->(:CallableResource)` edge matched a skill by `name`,
  which the engine can't resolve on a write (only `id` works) — see
  `orchestration/agent_runner.py:2046`. Now matches by resolved id, same as the
  `EXECUTED_ON` edge beside it.

---

*This catalog was produced by direct code inspection (grep + read) against the
`docs/au-catalog` worktree at commit `89da1265` + `93270411` + `071f6d86`, cross-
checked against `CHANGELOG.md`, `docs/capabilities.md`,
`scripts/surface_parity_baseline.txt`, and `scripts/check_surface_parity.py`. Where
a claim could not be verified in code, it is either omitted or explicitly flagged as
unverified/absent above — see §6.*
