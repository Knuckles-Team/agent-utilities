# Delivery plan

1. Pin and test the generated EG contract, including schema inspect, validation, graph/session, work, retrieval and memory methods required by active AU callers.
2. Record a caller inventory by module and public entry point. Migrate ontology and shape sources to EG or certified packs, prove semantic parity, then delete AU semantic files and imports.
3. Replace each graph facade and durable writer with typed client calls, removing legacy fallback in the same PR. Keep model-backed extraction as candidate-claim generation only.
4. Migrate untrusted historical data through an attested tenant-bound path or quarantine it. Never infer tenant from a mutable payload field.
5. Run static, contract, served, privacy and whole-repo gates; register exact merged-head evidence and update each item independently.

## Design

```mermaid
sequenceDiagram
  participant Host as graph-os
  participant AU as AU application
  participant Client as Generated EG client
  participant EG as EG schema and state
  Host->>AU: execute(verified context, RunSpec)
  AU->>Client: inspect/query/validate typed request
  Client->>EG: pinned contract digest + scoped request
  EG-->>Client: typed result + receipt/provenance
  Client-->>AU: exact generated result
  AU-->>Host: agent outcome + emitted events
```

#### Existing components to reuse

`agent_utilities/knowledge_graph/backends/epistemic_graph_backend.py` shows the client-adapter boundary. `agent_utilities/knowledge_graph/retrieval/context_compiler.py` is the retained context composer. `tests/gates/test_epistemic_graph_schema_authority_cutover.py` is the existing static authority check. Extend these rather than introduce a second graph service. The EG client is the sole schema and wire implementation; the generated release digest is checked during composition.

#### Typed request contract

Every AU→EG call carries a verified principal and tenant, graph, purpose, policy version, operation, bounded budget, idempotency key for writes, and an exact contract digest. Schema inspection returns source identity, composed digest, classes/properties and provenance. Validation returns findings and source digest; a failed validation cannot advance a cursor or publish a pack. Candidate-claim submission includes RunSpec and source evidence; EG issues the durable receipt. Unknown enum fields and digest mismatch fail before side effects.

#### Semantic migration

Inventory each AU TTL and parsing call. Associate it with an EG core source or a certified connector pack source. Publish the destination first. Verify semantic graph parity and composed digest. Then switch AU callers to IRI/digest requests and remove the AU file/parser. The cutover cannot leave two writable ontology authorities. Dynamic schema conflicts are atomic and typed. Schema drift classification runs in the SDK sync runner, while EG owns shape rendering, candidate validation and activation. AU only asks for an agent review when policy requires one.

#### Failure handling

Reject absent tenant/graph, stale digest, schema conflict, malformed claim, denied read, untrusted historical memory, unsupported method and owner outage. A transport retry reuses the same key only for idempotent operations. No branch falls back to an AU RDF graph or SQLite store.

#### Simplicity and quality

Delete adapters that only rename the same EG method. Keep one DTO conversion at the AU API/composition edge. CCCC highlights oversized client facades to split by use case. Jscpd and dupehound catch copied wire logic. KISS review rejects a second registry or optional backend. Use Ruff/mypy/Pytest plus normal repo hooks.

## Stage model allocation (AU-SEMANTIC-R029)

Data flow: `_analysis_action_evolve_variants (mcp/tools/analysis_tools.py:976) -> run_evolution_cycle (harness/agentic_evolution_engine.py:316) -> stage allocation -> route_model (decide/consumers/routing.py:58) -> served policy -> run trace`.

Types: `StageAllocation` and `RunBudget` are new. `route_model`, `route_model_by_observed_cost` (`routing.py:135`, no production caller today) and `select_model_for_step` (`core/resource_optimizer.py:232`, called from `graph/executor.py:258`) are reused. The policy decision stays served; this client only asks.

Failure: an exhausted budget refuses. An unreachable policy abstains and never picks the strongest model.

Decision: ask the served policy per stage. The rejected alternative is a fixed local tier table, and that table duplicates the policy.

Search terms used: select_model, route_model, model_router, RunBudget, stage_model, squeeze.
Existing code: `agent_utilities/decide/consumers/routing.py:58`, `:135`; `agent_utilities/core/resource_optimizer.py:232`; `agent_utilities/core/model_router.py:111`; `agent_utilities/harness/agentic_evolution_engine.py:316`. Reachable today: per-role routing yes, per-stage allocation in evolution runs no.

### References

- arXiv:2604.07725 — Maheswaran et al., "Squeeze Evolve: Unified Multi-Model Orchestration for Verifier-Free Evolution" (2026)
