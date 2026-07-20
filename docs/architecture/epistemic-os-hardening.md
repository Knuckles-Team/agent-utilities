# Epistemic OS Hardening — historical program audit

> **Historical, non-authoritative reference.** This page preserves the audit trail
> for an earlier hardening increment. Version numbers, release-step language,
> status labels, gaps, and defaults below describe that increment and must not be
> used as current operating or source contracts. Current authority lives in
> [Graph Authority Convergence](graph-authority-convergence.md),
> [Mandatory ContextCompiler](mandatory-context-compiler.md),
> [Universal External Graph Connectors](universal-external-graph-connectors.md),
> [Privacy-safe External Ingestion](privacy-safe-external-ingestion.md), and the
> generated [runtime configuration](../reference/runtime-configuration.md). Live
> release and connector results are recorded only by the corresponding
> certification harnesses.

This is the **historical audit catalog** for what `agent-utilities` (AU) shipped in the
"Epistemic OS Hardening" program — the AU half of a coordinated two-repo release with
`epistemic-graph` (EG). It exists so a future static audit does not have to re-derive
"what changed" from commit archaeology: every capability below is verified directly
against `agent_utilities/` source at the commit this doc was written against
(`89da1265` + the AU-owned pre-commit-gate closeout commits `93270411`/`071f6d86` —
the version bump itself is the one remaining release step, so `pyproject.toml` still
reads `1.20.0` at the time of writing; every capability described here is already in
the tree).

**Companion documents** (read these first for the summary; this doc goes deeper):

- [`CHANGELOG.md`](https://github.com/Knuckles-Team/agent-utilities/blob/main/CHANGELOG.md) `## [1.21.0]` — the shipped-reality summary,
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
At the time of the original increment, neither its proposed Arrow Flight compute
plane nor an analytics-job scheduler had landed. The current architecture does not
add Arrow Flight as a second authority: authenticated standalone workers claim
engine-owned `eg-jobs` work instead. Native `ProgramOptimize` scheduling and its
on-demand surface have since shipped; §3.2 records that current-source closure.

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
| `GraphSession` — one explicit actor/tenant/scope/graph/trace/policy-version currency | `knowledge_graph/core/session.py::GraphSession` (`from_ambient`, `use_session`, `require_scope`) | — (module docstring: AU-P0-1) | Served boundaries require a verified middleware- or process-minted session; caller fields cannot construct or override authority |
| Native-Cypher authority, never silent `[]` | `knowledge_graph/backends/epistemic_graph_backend.py::CypherEngineError`; 11 `NotImplementedError` sites (grep `ABSTRACT-OK.*AU-P0-2`) | `AU-KG.query.vendor-agnostic-traversal`, `AU-KG.query.object-graph-mapper` | Always on — an unsupported Cypher shape raises, it never degrades to an empty result |
| Engine-native `WorkItem`/claim, fencing that **fails closed** | `orchestration/agent_dispatch_worker.py::_fence_still_valid`; `orchestration/work_item.py` | (AU-P0-3, hardened this cycle: L15) | WorkItem is the sole writable task authority; claims, renewals, and commits are native and fenced |
| Fail-closed connectors (quarantine unknown ACL, universal activation manifest gate, tombstone guard) | `protocols/source_connectors/base.py::ExternalAccess.quarantined`/`default_external_access`; `protocols/source_connectors/registry.py::build_connector`; `knowledge_graph/ontology/connector_manifest_gate.py`; `knowledge_graph/core/source_sync.py::_reconcile` | `AU-P0-4` | Missing ACL is always quarantined, and a signed connector bundle is required for every non-test activation; `SOURCE_SYNC_ALLOW_EMPTY_TOMBSTONE` gates authoritative empty snapshots |
| Tenant RLS end-to-end, fail-**closed** cross-tenant GUC fix | `knowledge_graph/backends/postgresql_backend.py::set_request_tenant` (raises on a failed `SET LOCAL` for a *non-empty* tenant rather than serving a stale GUC) | (AU-P0-5) | Always on for a non-empty tenant; fail-open only for the unscoped/system baseline (`tenant_id == ""`) |
| Scoped low-level engine tools (`engine_<domain>`) + one process client | `mcp/tools/engine_tools.py::action_policy`/`_enforce_action_scope`/`_client_for`; `knowledge_graph/core/graph_compute.py::get_or_create` | `AU-P0-6` | Every action requires its calculated `kg:read`, `kg:write`, or `kg:admin` GraphSession scope; `KG_ENGINE_TOOL_POOL_SIZE` bounds only zero-connection graph views over the process transport |

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

**What it is.** `WorkItem` is the sole writable lifecycle and ownership record
for ingestion, goal loops, teams, orchestrator work, and dispatched turns:

```
submitted -> ready -> leased(fencing_token) -> running(heartbeat, attempt)
    -> succeeded(result_ref) | failed(error_ref) | cancelled | dead_letter
```

**Code anchor.** `orchestration/work_item.py::WorkItemStatus` (the `StrEnum`),
`TERMINAL_WORK_ITEM_STATUSES`, `submit_work_item`/`claim_next`/`commit_result`/
`reap_expired_leases`. Production claim/renew/commit/cancel/defer transitions
use dedicated epistemic-graph transactions. The graph CAS implementation is a
local dependency-injected test harness, not a second production authority.

**Current producers and consumers:**

| Subsystem | Status | Bridge |
|---|---|---|
| Agent dispatch (`agent_dispatch_worker.py`) | **Native** | claims and commits the submitted execution WorkItem |
| Ingestion (`engine_tasks.TaskManagerMixin`) | **Native authority** | WorkItem owns selection, lease, backoff, and dead-letter state |
| Teams (`capabilities.teams.TeamCapability`) | **Native** | team DAG edges connect WorkItems directly |
| Loop/Goal (`research/loops.py`) | **Native** | definitions are read-only; WorkItem owns lifecycle |

**A missing/no-op executor never fakes success.** Continuing the Phase-0 discipline:
a task with no bound executor resolves to `unroutable`/`failed`, never
`completed`/`reward=1.0` — the `WorkItem` state machine carries the same discipline
forward via `dead_letter` once retries are exhausted (never a silent "done").

**Two surfaces.** `WorkItem` itself has **no** dedicated `graph_workitem` MCP
tool/REST route — by design, it is the shared backing state machine *behind*
existing entrypoints: `graph_jobs` (`action=dispatch`/`status`, REST
`/graph/jobs`) drives durable execution, `graph_agents` drives agent/team paths,
`graph_ingest` drives the
ingestion-queue path, and `graph_goals` (REST `/graph/goals`) exposes the Loop
read-only projection. A raw `WorkItem` row is queryable like any other node via
`graph_query action=cypher` (`MATCH (w:WorkItem {id: $id}) RETURN w`, REST
`/graph/query`) — there is no typed getter tool for it.

### 2.2 Partitioned-log `AgentBus` delivery plane

**What it is.** `AgentBus` stores its low-churn semantic registry and transactional
inbox/outbox records in the KG. `messaging/bus_log.py` provides the required bounded
partitioned delivery plane with explicit acknowledgements, a DLQ, and backpressure.

**Code anchor.** `messaging/bus_log.py::resolve_bus_log_backend` (the required
resolver), `AgentBus.send`/`_send_via_log`/`receive` (`messaging/bus.py`).

**Backends** (`BUS_LOG_BACKENDS = ("engine", "kafka")`):

1. **engine** — the epistemic-graph engine's native AMQP-style broker (the exact
   surface `graph_broker`/`engine_broker` expose: `declare_exchange`/`declare_queue`/
   `bind`/`publish`/`consume`), with a fixed number of tenant partition queues.
2. **kafka** — two keyed topics (`agent_bus_direct`/`agent_bus_topic`), tenant-
   qualified partition keys, and one shared materializer group per tenant/topic
   pair. Consumer count is independent of subscriber cardinality.

**Default posture.** The engine backend is the default. An explicitly selected
backend is a hard contract and an unavailable backend fails closed.

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
`CapabilityIndex` (the in-process hnswlib/numpy structure) is a finite,
non-authoritative learner — LRU-evicted with a 4,096-entity default and kept fresh
by CDC deltas — used for object-index consumers and reward write-back
(`record_outcome`/`record_capability_outcome`).

**X-4 extends the same file** (see §4.3) — `engine_filtered_search` always
applies ontology subsumption. An injected `capability_hierarchy` can isolate a
schema; otherwise the bundled current hierarchy is resolved automatically.

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
`store_rendition` (derived forms) and canonical `AssetOccurrence` retrieval
(closed this cycle: L29).

**Identity chain:** `Blob(digest)` ← `Rendition` ← `AssetOccurrence` ←
`Message`/`Document`. Only immutable bytes dedup (`:Blob`, content-addressed); every
`store_media()` call mints a **distinct** `:AssetOccurrence` owning its own
source/tenant/owner/acl/event_time/retention/legal_hold/provenance. One cross-modal
ACID txn writes the node + its blob-ref + the occurrence edge together (a reader
never sees a half-written occurrence).

**Two surfaces and background extraction.** `graph_ingest action=ingest` queues
document and media inputs as durable WorkItems and exposes job status through
`/graph/ingest`. The worker routes audio and image inputs through the shared lazy
ASR/OCR reader registry before the normal Document/Chunk enrichment path.
`document_process` and `POST /document/process` accept already-extracted text for a
bounded synchronous call. Messaging attachments are persisted by
`messaging.router._persist_media` on the background persistence pass, off the reply
path. There is no standalone `graph_media` authority: `record_extraction` and
`store_rendition` remain output callbacks for optional or external models, while
identity, bytes, and provenance stay in the same engine-backed `MediaStore`.

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
**one engine-native `ApplyChangeEnvelope` transaction** containing graph rows,
blob/feature/evidence material, policy, lineage, typed content version, source
cursor, and the projection/CDC outbox. The verified `GraphSession` supplies
tenant/policy/trace authority; catalog epoch and fencing token are route-bound,
and graph OCC retries only on an explicit pre-commit stale response with the
same idempotency key. Returns `status` in
`{"success","skipped","rejected","failed"}` and `watermark_advanced: bool` so a
resumed connector can trust `False` to mean "retry this envelope."

There is no downgrade to sequential Python writes. A non-redb authority, a
missing verified session, or a missing native capability fails closed before
materialization.

**Migration scope.** Every durable `source_sync` handler is envelope-native.
GitLab's resolver is read-only and its complete project graph is one native
multi-node envelope; document/chunk/section slices from ArchiveBox, Confluence,
and the connector fleet are one envelope each; RSS/FreshRSS preserve their
relevance gate and commit decisions/cursors natively; fleet capability slices
and LeanIX relationship projections are native multi-node envelopes. Chunked
full-corpus drains retain their carried page cursor on any page failure and commit
the terminal source cursor only with a native review-marker envelope after verified
exhaustion. The sole
allowlisted entry, `package_install`, is a write-free dispatcher in
`source_sync.py`; the architecture gate verifies it cannot call a graph writer.

**Two surfaces.** No standalone `graph_change_envelope` tool — it is the internal
currency behind `source_connector`/`graph_ingest`'s connector-sync path (`source_sync
action=sync`, REST via `source_connector` → `/connector/source`).

### 2.6 The mandatory signed connector capability fleet

**What it is.** `agents/<pkg>/connector_manifest.yml` — a per-connector manifest
whose `provenance.integrity.hash` is bound to an **Ed25519 release signature**
(`knowledge_graph/ontology/ontology_integrity.py::canonical_hash` and
`ReleaseSigner`), using an explicitly referenced release key and independently pinned
public key, re-verified before every sync. `connector_manifest_gate.py` wires this into
`source_sync` as a **fail-closed** precheck.

**The fleet.** All 65 connector packages declared by the repository-manager
workspace own a versioned capability bundle: manifest, ontology, SHACL shapes,
mappings, source presets, exact tool-schema fingerprints, fixtures, migrations,
and certification metadata. `MANDATORY_NAMED_CONNECTOR_SOURCES` retains the
13 source-key aliases used by the original 12-package rollout, but
`mandatory_connector_packages()` discovers the full packaged fleet and makes it
unconditional.

**Unconditional.** The required-source registry combines every canonical source
key and all bundled package names. Any additional registered connector becomes
subject to the same gate; operators do not maintain a security allowlist.

**Portable and connector-owned.** The authoritative bundle is committed in
each connector repository. Agent Utilities also carries a generated, pinned
manifest fallback so a standalone wheel or GraphOS binary enforces the same
minimum contract without a sibling checkout. A live connector-owned bundle wins
when present. Missing bundles, invalid signatures, ontology/tool-schema drift,
or provider startup/schema failures stop ingestion before any record is applied.

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

**Code anchor.** `knowledge_graph/core/placement_catalog.py::resolve_placement`:
(1) a short-TTL `(endpoint, group, epoch, fence)` cache per `(tenant, sub_key)`;
(2) on miss, ask configured coordinator contacts in their declared order;
(3) require an authoritative complete response; and (4) map the returned group
through explicit deployment topology when groups use distinct endpoints. Any
unreachable authority, invalid fence, placed epoch zero, or ambiguous topology
fails closed. There is no hash or raw-method fallback.

Current Epistemic Graph releases expose `PlacementRoute` and the typed
`client.placement.route(...)` namespace. A one-endpoint production cell treats that
endpoint as its stable coordinator and preserves the returned group and epoch with no
additional configuration. A multi-endpoint deployment whose catalog returns only a
group must define `GRAPH_RAFT_GROUP_ENDPOINTS`; missing topology fails closed rather
than discarding the authoritative route. Structured `STALE_ROUTE` errors refresh the
catalog and retry once with the same idempotency key and the new fence.

**Default posture.** Placement authority is mandatory.
`PLACEMENT_CATALOG_TTL_S` defaults to `5.0` seconds. Tests inject the typed
placement client; runtime lookups require an authenticated `GraphSession`.

**Two surfaces.** Construction selects only a coordinator contact through the one
engine resolver. Every authenticated graph operation then consumes the engine route;
there is no dedicated placement tool or construction-time identity bypass.

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

**Execution and two surfaces.** The registry remains a read-only internal view; it
does not need a second write authority. The capability it indexes is current and
reachable: `graph_evolution action=optimize_component` submits the engine-owned
`ProgramOptimize` job and has the REST twin `POST /graph/evolution`. The scheduled
optimization sweep calls the same `run_component_optimization` core, and
`graph-os-analytics-worker` claims, renews, stages, and publishes jobs through the
authenticated engine job protocol. `KG_OPTIMIZATION_ENABLED` and
`KG_OPTIMIZATION_INTERVAL` are the typed scheduling controls. There is no Python
optimizer or Arrow Flight fallback.

### 3.3 Workload contract + soak/chaos harness (SCALE-P2-1)

**What it is.** Replaces `docs/scaling/capacity_model.md`'s prior linear-arithmetic
"1M residents" claim with a **machine-readable contract**:
`scripts/scale/workload_contract.yml` — registered agents (1,000,000), concurrent
active sessions (20,000, `= capacity_model.active_agents(1e6, 0.02)`), 5 independent
rate axes (turns/sec, tool-calls/sec, graph-mutations/sec, messages/sec,
token-throughput), tenant count + Zipf skew + one elephant tenant, per-agent
footprint, interactive/background mix, availability + RPO/RTO, and
p50/p95/p99/p99.9 SLO targets. Every numeric axis is anchored to an **existing**
`capacity_model.py` constant (commented `# anchor:` in the YAML) rather than a fresh,
unverified number.

**Code anchor.** `scripts/scale/workload_contract.yml` (the packaged contract),
`scripts/scale/workload_contract.py` (typed loader + `ScaledWorkload.for_scale` — the
SLO targets and per-unit sizes do **not** scale down for a smaller run; population/
rate axes do), `scripts/scale/loadgen.py` (the load generator that drives the
contract against the real `WorkItem` path), `tests/scale/test_workload_contract.py`
(cross-checks the YAML against `capacity_model.py` so the two can't drift).

**Recovery contract.** The dispatch lease now satisfies the declared five-minute
RTO by construction: `AGENT_DISPATCH_CLAIM_TTL_S` is typed, defaults to 120 seconds,
and is capped at 300; `AGENT_DISPATCH_RENEW_INTERVAL_S` defaults to 30 seconds and
is always shorter than a valid lease. The config doctor reports the
`dispatch_lease_recovery` invariant instead of requiring an undocumented deployment
override.

**Measured boundary.** The contract, loader, and harness are built; the actual
**sustained 1M-resident soak has not been run** to ratify the SLOs. This is the standing
`feedback-test-migrations-at-production-scale` discipline (prove at scale by
running the workload, not by modeling it). The document therefore distinguishes
implemented recovery invariants from production-scale certification evidence.

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
through (`graph_governance` approval/policy actions, the fleet reconciler,
remediation playbooks — see `docs/architecture/fleet_autonomy.md`).

### 3.5 Bounded lazy graph lifecycle and maintained-index correctness (P0-5)

**What it is.** Agent Utilities carries the engine's recovery policy as typed
configuration instead of relying on an operator to discover three independent daemon
settings. Generated `single-node-prod` and `enterprise` profiles enable catalog-only
startup, cap resident graphs at 1,024, and page durable recovery in batches of 4,096.
Development may use eager startup, while graph residency and recovery remain
bounded. The production guard rejects eager startup or either zero bound, and the autostart launcher projects the
resolved values into the child environment.

**Correctness boundary.** The engine owns the hard guarantees: immutable graph
incarnations, delete/evict cancellation, per-graph lifecycle locks, durable source-version
fences, explicit `PARTIAL_MATERIALIZATION` responses, and maintained-index manifests
(source snapshot, build version, completeness cursor, validity). A recovering graph is
not available until every page and maintained index covers the same durable snapshot.
Engine `Health`/`ListGraphs` responses carry completeness and freshness; they do not
persist workstation identities, endpoints, or filesystem locations.

**Code anchors.** `core/config.py`, `core/profile_guard.py`,
`deployment/config_generator.py`, and
`knowledge_graph/core/graph_compute.py::_autostart_engine`; engine-side evidence is
cataloged in epistemic-graph's `docs/architecture/epistemic-os-hardening.md` and guarded
by `scripts/check_lazy_lifecycle_architecture.py`.

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
`graph-research-and-analysis` workflow skill.

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
(`graph_loops`, MCP+REST) or `graph_workflows action=execute`'s
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
`build_capability_filters`/`engine_filtered_search` accept an injected
`capability_hierarchy` for isolated schemas; `None` resolves the bundled current
hierarchy and never selects a flat exact-string mode.

**Two surfaces.** Subsumption-aware candidate filtering remains infrastructure
behind `graph_orchestrate`'s designation/dispatch path. The operator-facing WHY
view is now the `ontology_interface action=explain_routing_eligibility` MCP action;
the shared dispatcher exposes the same core at `POST /ontology/interface`. It
accepts the candidate entity, requested capability, and optional tenant/policy
constraints, then calls `explain_routing_eligibility` directly. No second routing
or policy authority was introduced.

### 4.4 X-5 — Workload-aware placement mining

**What it is.** Mines agent-trace co-occurrence (tenant/tool/entity/modality access
skew, over the canonical `RunTrace -[:USED_TOOL]-> ToolCall -[:ACTED_ON]-> Entity`
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

**Triggers and default posture.** `graph_loops action=placement_control` is the
explicit one-pass trigger. Its shared REST dispatcher exposes the same action at
`POST /graph/loops`; both call `placement_control_loop`, which delegates to the one
`run_placement_mining_cycle` governance/canary spine. Automatic execution is opt-in
through typed `AgentConfig.placement_control_loop_enabled`
(`PLACEMENT_CONTROL_LOOP_ENABLED=false` by default). When enabled, the existing
`LoopController.run_one_cycle` invokes the same controller stage; there is no second
scheduler or apply implementation. Whether manually or automatically triggered,
the shipped `apply_placement_change` policy remains `approval_required`, so enabling
the mining stage does not silently authorize a move.

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

**Serving seam.** The compiled bundle uses the shared, content-addressed
epistemic-graph KV surface through `ContextCompiler.compile(..., kv_backend=...)`.
Its key binds the evidence set, policy version, and token budget; a cache hit restores
citations, calibrated scores, and proof edges without repeating selection. The
provider boundary in `context_compiler_serving.py` then carries the same governed
bundle into the per-request LMCache policy, so epistemic selection and serving reuse
share one current path. `tests/retrieval/test_context_compiler_kv_seam.py` guards
stable reuse and rejects cross-evidence or cross-policy collisions.

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

**Two surfaces.** `graph_runvcs` now carries four twin actions:
`twin_capture`, `twin_replay`, `twin_counterfactual`, and `twin_incident`.
`twin_capture` hydrates from the authoritative WorkItem/ToolCall graph and may
persist the projection; the other actions consume its serialized, version-pinned
twin without re-executing recorded tools. The shared dispatcher provides the REST
twin at `POST /graph/runvcs`. These actions call the functions above directly; the
pre-existing generic live-session `replay` action remains a distinct run-VCS
operation rather than a duplicate twin implementation.

The surface-parity scanner now treats `orchestration/agent_digital_twin.py` as an
explicit capability module. That targeted coverage prevents this cross-package
feature from becoming unreachable without classifying orchestration protocols and
other infrastructure as standalone tools.

---

## 5. The two-surfaces map (quick reference)

| Capability | MCP tool (action) | REST route | Standalone or piggybacked? |
|---|---|---|---|
| Low-level engine surface (19 normal + 5 admin domains) | `engine_<domain>` | `POST /engine/<domain>` | Standalone, one per domain |
| `WorkItem` (query only) | `graph_query` (`cypher`) | `/graph/query` | Piggybacked (generic Cypher) |
| `WorkItem` (execution/team dispatch) | `graph_orchestrate` (`dispatch`/`execute_agent`) | `/graph/orchestrate` | Piggybacked |
| `WorkItem` (ingestion queue) | `graph_ingest` | `/graph/ingest` | Piggybacked |
| `WorkItem` (Loop/Goal read view) | `graph_goals` | `/graph/goals` | Piggybacked |
| `AgentBus` / partitioned log | `graph_bus` | `/graph/bus` | Standalone |
| Engine-native capability index | `graph_orchestrate` (designation), `graph_search` (`discover`) | `/graph/orchestrate`, `/graph/search` | Piggybacked |
| `AssetOccurrence` / media identity and extraction | `graph_ingest`, `document_process` | `/graph/ingest`, `/document/process` | Piggybacked |
| `ChangeEnvelope` ingest | `source_connector` (`sync`) | `/connector/source` | Piggybacked |
| 12 mandatory manifests | (precondition inside `source_connector`) | `/connector/source` | Piggybacked (gate, not a callable) |
| Placement-catalog consumer | (transparent to every entrypoint) | — | None (infra) |
| Analytics jobs + registry views | `graph_evolution` (`optimize_component`) | `/graph/evolution` | Job action surfaced; registries are read-only views |
| Workload contract + soak harness | **none** | **none** | Offline artifact, not runtime |
| `ActionPolicy` caching | (transparent) | — | None (perf fix) |
| X-2 ops-causal graph | `graph_ops_causal` (5 actions) | `POST /ops/causal` | **Standalone** |
| X-3 claim flywheel | `graph_loops` (indirect, whole-pass only) | `/graph/loops`-family | Piggybacked, coarse-grained |
| X-4 ontology routing | `graph_orchestrate` (designation/filtering), `ontology_interface` (`explain_routing_eligibility`) | `/graph/orchestrate`, `/ontology/interface` | Piggybacked; same explanation core on both operator surfaces |
| X-5 placement mining | `graph_loops` (`placement_control`) | `/graph/loops` | Piggybacked; optional same-core automatic loop stage |
| X-6 (`ContextCompiler` epistemic columns — EG-side TMS/recompute, consumed not built here) | `graph_search` (`compiled`) | `/graph/search` | Piggybacked |
| X-7 context compiler | `graph_search` (`compiled`) | `/graph/search` | **Standalone mode on an existing tool** |
| X-8 agent digital twin | `graph_runvcs` (`twin_capture`/`twin_replay`/`twin_counterfactual`/`twin_incident`) | `/graph/runvcs` | Piggybacked on run-VCS |

---

## 6. What this doc deliberately does NOT claim

- **Not AU work this cycle:** calibrated causal do-calculus (do-intervene/
  counterfactual), proof/belief redaction (`explain_belief_redacted`), bitemporal
  epistemic-status ops — all `epistemic-graph` `eg-epistemic` crate, see EG's own
  `docs/architecture/epistemic-os-hardening.md` and its 1.21.0-cycle CHANGELOG.
- **Not new this cycle:** the Claim/Evidence/BeliefState substrate and
  `belief_revision` itself — shipped in the 1.12.0 epistemic-substrate release.
- **No second heavy-compute authority:** external analytics workers use the
  authenticated engine job protocol; there is no Arrow Flight or Python optimizer
  fallback. Native scheduling and on-demand submission are described in §3.2.
- **Certification evidence is distinct from implementation:** the sustained
  1M-resident production soak is not claimed here. The lease/RTO mismatch,
  placement-control trigger, and ContextCompiler KV serving seam are implemented;
  a full-scale run remains a deployment certification activity, not missing source.

## 7. Other Phase-1 reliability fixes (named for completeness)

Smaller but real fixes shipped alongside the headline items above, from the
CHANGELOG's `### Fixed` section — verified, briefly noted rather than given a full
section since they are not new *capabilities*:

- **`_fence_still_valid` fails closed on the engine-native claim path (L15)** — see
  §1's Phase-0 recap table; this is the specific behavior change, not just a
  cross-reference.
- **Self-ingest telemetry is durable (OBS-P1-1)** — `observability/self_ingest.py`'s
  sanitized record enters a SQLite WAL with `synchronous=FULL` before it becomes
  eligible for network delivery. The delivery queue is only an accelerator;
  backpressure and exhausted retries leave the durable row pending. Rows are
  deleted only after the remote batch acknowledgment, so restart/failure is
  at-least-once. Raw tenant/actor identity is replaced by opaque references,
  local paths/endpoints/personal fields are stripped at the WAL boundary, and
  sink failures use a non-recursive emergency channel. The one loss case (the
  bounded WAL unavailable or saturated) is counted and emitted without payload.
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
