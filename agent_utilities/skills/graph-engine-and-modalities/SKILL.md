---
name: graph-engine-and-modalities
skill_type: skill
description: >-
  Use low-level native epistemic-graph engine primitives for storage,
  transactions, SQL, SPARQL/RDF, analytics, reasoning, time series, streams, blobs,
  channels, ledger, finance, mining, tenancy, consensus, resharding, RBAC, and
  administration. Use when work explicitly belongs on the engine data, compute,
  or cluster plane. For canonical ontology design or governed Graph-OS object,
  memory, node, or edge changes, use graph-modeling-and-mutation.
---

# Graph engine and modalities

Select the native engine domain, keep Python orchestration thin, and verify the
result through the served Graph-OS contract.

Keep canonical ontology and governed object mutations in
`graph-modeling-and-mutation`; use engine primitives only when the native plane
owns the operation.

## Choose the engine domain

| Domain | Operations |
|---|---|
| Core graph and lifecycle | `engine_nodes`, `engine_edges`, `engine_graph`, `engine_lifecycle` |
| Transactions | `engine_txn` |
| SQL and relational query or mutation | `engine_query` |
| SPARQL/RDF query or update | `engine_rdf` |
| OWL and rule reasoning | `engine_reasoning` |
| Analytics, graph learning, and data science | `engine_analytics`, `engine_graphlearn`, `engine_datascience` |
| Mining and finance | `engine_mining`, `engine_finance`, `quant` |
| Persisted enterprise, regime, and RLM actor operations | `graph_domain_ops` |
| Time series and streaming | `engine_timeseries`, `engine_streaming` |
| Blobs, ledger, and messaging | `engine_blob`, `engine_ledger`, `engine_channels`, `engine_broker` |
| Cluster control | `engine_consensus`, `engine_resharding`, `engine_tenants` |
| Administration | `engine_rbac`, `engine_admin` |

Use a direct engine operation for a bounded, well-specified task. Delegate one
specialist via `graph_orchestrate`, a collective via `graph_agents`, or a
dependency DAG via `graph_workflows`.

## Action reference

Every modality tool below shares ONE shape: action-routed 1:1 over the corresponding
`epistemic_graph` client (`NodeClient`, `BlobClient`, `FinanceClient`, …), so its action
set is discovered from the live client and never drifts in this doc — call any of them
with an empty `action` to list what the connected engine build actually exposes.
Common invocation: `load_tools(tools=["engine_<domain>"])` →
`engine_<domain>(action="", params_json="{}")` to list → `engine_<domain>(action=
"<method>", params_json="{...}", graph="")` to invoke → `unload_tools(...)`. REST
twin: `POST /engine/<domain>` with the same body shape.

| Domain tool(s) | Fronts | Notable actions / gotchas |
|---|---|---|
| `engine_nodes` / `engine_edges` / `engine_graph` / `engine_lifecycle` | core graph CRUD at the wire level | batch/union reads, degree/neighbour queries (`nodes`); temporal invalidate/supersede (`edges`); AST parse/index + semantic/embedding compute (`graph`); prune/decay/evict, `batch_update`, context view, (de)serialize (`lifecycle`) — the low-level primitives beneath the curated `graph_write`/`graph_query`; prefer those for everyday reads/writes |
| `engine_txn` | server-side ACID transactions, optimistic concurrency control (OCC) | `begin` → stage ops → `commit`/`rollback`; a transaction can span modalities (graph/tabular/timeseries/blob) and commits all-or-nothing; an OCC conflict rejects the commit — retry the transaction, never partially apply it |
| `engine_analytics` / `engine_datascience` / `engine_graphlearn` | centrality + (personalized) PageRank; estimators/numeric primitives/training kernels; a pure-Rust KAN (Kolmogorov-Arnold) link predictor | the raw modality-tier routers; for the friendlier fixed-action `graph_learn` wrapper over the same link predictor, see `graph-research-and-analysis` |
| `engine_blob` | content-addressed binary object store for media/files/large payloads | put/store, get/fetch, stat, delete + streamed variants; objects are keyed by content hash (dedupe-by-content); binary results return base64-wrapped as `{"__bytes_b64__": "..."}` |
| `engine_channels` / `engine_broker` | dynamic pub/sub agent-communication channels; the engine's native RabbitMQ/Kafka-class broker (exchange/queue/stream admin, routed publish incl. confirmed/idempotent, consumer-group ack/nack) | the raw engine-native fabric; for the federated, durable, cross-host agent-to-agent bus, use `graph_bus` (`graph-orchestration-and-automation`) instead |
| `engine_finance` | portfolio optimization, risk metrics, regime detection, signal generation, HFT primitives, derivatives pricing | computed in-engine over the same substrate that holds the market/entity graph and time-series data; pair with `engine_timeseries` for price history and `engine_analytics`/`engine_datascience` for cross-modal statistics; the separate `quant` (emerald-exchange) tool fronts a different domain, not this one |
| `engine_ledger` | append-only audit ledger recording graph mutations | `get`/`apply`/`clear` (admin) — a durable "who changed what, when" distinct from live graph state; for higher-level KG-native audit records (ExecutionSummary, action outcomes) see `graph-runtime-and-governance`'s audit/compliance tools instead |
| `engine_mining` | association-rule mining, clustering, and anomaly detection running inside the engine, compute-near-data | the raw modality-tier router (empty `action` lists what the live engine supports); the friendlier fixed-action `graph_mine`/`graph_mine_deep` wrappers (including the deep-learning family delegated to `agents/data-science-mcp`) and the GOVERNED mining→claim-flywheel are documented in `graph-research-and-analysis` |
| `engine_timeseries` | a native time-series database co-located with the graph | append, range and window scans, as-of (point-in-time) lookups, gap-filling — temporal metrics queryable cross-modally alongside graph and tabular data |
| `engine_streaming` | change-data-capture streams, continuous (standing) queries, watches, and triggers | register interest and the engine pushes matching changes instead of polling — a reactive substrate for event-driven agents/pipelines |

## Workflow

### 1. Specify the contract

State the graph or tenant, input shape, operation, bounds, consistency needs,
expected output, and verification. Confirm that the selected domain owns the
work instead of implementing an equivalent loop in Python.

### 2. Check capabilities and cost

- Confirm the operation is served by the current engine build.
- Bound rows, neighbors, time range, batch size, iterations, and result size.
- Use streaming or jobs for work that cannot fit within an interactive call.
- Choose an economical model for deterministic result interpretation; use a
  stronger model only when synthesis or uncertainty warrants it.

### 3. Execute safely

- Keep reads and writes explicitly separated.
- Use transactions or compare-and-set for concurrent mutations.
- Preserve tenant and authorization context.
- For resharding, consensus, RBAC, or administration, require the relevant
  governance decision before execution.

Use the native modality contract instead of treating every engine call alike:

- For SQL, identify read, DML, or DDL intent; parameterize values; bound returned
  rows; and keep transaction, tenant, and authorization context explicit. Wire
  listeners and authentication remain external deployment configuration.
- For SPARQL/RDF, distinguish query from update, name the graph scope, parameterize
  or structurally validate untrusted terms, and preserve asserted versus inferred
  provenance.
- For reasoning, select classification, consistency checking, rule evaluation, or
  bounded materialization. State the ontology/rule set and verify inferred facts
  separately from asserted facts.
- For consensus and resharding, inspect membership, leadership, health, and active
  transactions before proposing a change. Preserve quorum and verify ownership after
  a move.
- For tenancy, create or route only the named tenant and verify isolation. For RBAC,
  make the smallest role or grant change and read it back. For administration,
  authorize maintenance explicitly and validate backup or restore artifacts through
  the served engine contract.

### 4. Verify

- Read back mutations through the normal graph surface.
- Check job completion and result artifacts, not only job acceptance.
- Compare analytic output with a small known case or invariant.
- Confirm retries do not duplicate ledger entries, blobs, or streamed events.
- Confirm SQL/SPARQL writes and reasoned materializations are visible through an
  independent bounded read in the same tenant and graph scope.

## Guardrails

- Do not reproduce native graph, vector, or numerical loops in Python.
- Do not run unbounded whole-graph analytics on an interactive path.
- Do not move data across tenants or graphs implicitly.
- Do not expose engine administration to an unprivileged actor.
- Report unsupported operations explicitly; do not fabricate a fallback result.
