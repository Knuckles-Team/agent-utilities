---
name: graph-engine-and-modalities
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
