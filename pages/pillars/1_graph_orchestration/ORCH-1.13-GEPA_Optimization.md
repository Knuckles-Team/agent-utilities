# Native GEPA program optimization

GEPA is one of the thirteen optimizer families compiled by the Rust
`epistemic-graph` `eg-program` authority. Agent Utilities does not carry a Python
GEPA implementation or a second provider stack. It maps ephemeral training data to
opaque program, corpus, evidence, budget, and promotion contracts, then submits one
durable `ProgramOptimize` job.

The native compiler emits a bounded model-transport plan for GEPA reflection. The
plan contains only opaque input/output references, explicit dependencies, a maximum
operation count, policy scope, and all observed modalities. Existing governed model
transport executes the plan; the optimizer never creates a duplicate LLM client.

Every candidate remains propose-only until typed evaluation evidence proves the
promotion policy. No raw prompt, response, identity, endpoint, credential, or local
path is representable in the durable contract.

Current implementation:

- `agent_utilities/harness/optimization_backend.py` — privacy-safe request mapping
  for all fourteen modalities and all thirteen optimizer families.
- `agent_utilities/harness/program_optimization.py` — target registry, evaluation,
  and promotion boundary.
- `epistemic-graph/crates/eg-program` — optimizer compiler, plans, candidates, and
  promotion rules.
- `graph_evolution action=optimize_component` — the sole GraphOS on-demand surface.
