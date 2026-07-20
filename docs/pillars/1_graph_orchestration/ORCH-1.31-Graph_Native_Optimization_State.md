# Graph-native program optimization state

Program revisions, training-corpus references, optimizer artifacts, plans,
candidates, typed results, evaluation evidence, and promotion decisions are durable
epistemic-graph state. The distributed analytics-job lifecycle supplies leases,
checkpointing, cancellation, retry fencing, result publication, and restart recovery.

Agent Utilities submits and polls the native job through `GraphComputeEngine`; it
does not serialize a Python optimizer frontier. A failed, cancelled, malformed, or
timed-out job remains failed, and no alternate optimizer runs. Successful result rows
are validated against one frozen typed schema before they can become proposals.

The durable state contains opaque references and numerical coordinates only. Raw
training material is resolved in memory for execution and is never written into the
job, graph, trace, or review report.

Current implementation:

- `agent_utilities/harness/optimization_backend.py`
- `agent_utilities/knowledge_graph/core/graph_compute.py`
- `epistemic-graph/crates/eg-program`
- `epistemic-graph/crates/eg-jobs`
