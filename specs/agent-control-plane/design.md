# Design: agent control plane

## Reuse and live wiring

The checkout already contains `agent_utilities/graph/builder.py`, `graph/routing/`, `graph/subagent_patterns.py`, `graph/topology_engine.py`, `graph/team_composer.py`, `rlm/sandboxes/`, and `knowledge_graph/research/loops.py`. Those are migration inputs, not evidence that the target contract is wired. The current tree does **not** contain `agent_utilities/layers/` or `agent_utilities/decide/`; do not claim the proposed port and consumer implementations are landed. Preserve working execution behavior while replacing selection and success-rate authority at a single cutover. EG and graph-os contracts must be available as public tagged/generated dependencies, with in-memory test doubles for unit tests.

## Data flow and interfaces

```text
task + tenant/policy → typed mapping claim → EG AgentAssemble / Decide
 → DecisionCommit(snapshot checked) → graph-os capacity admission
 → AU L3 node scheduler → HarnessPort.negotiate(RunSpec, descriptor, policy)
 → SandboxPort.lease (if local) → HarnessPort.start/events/result
 → EG durable run events + independent outcome → release leases
```

`RunSpec` is immutable and digest-bound: task/context handles, L1/L2/L3 component digests, required/optional capabilities, toolset and skill digests, minimum trace fidelity, policy/auth references, account and environment modes, usage units and hard/advisory budgets, deadline/cancel semantics, artifacts, completion predicate, side-effect/idempotency and reconciliation policy. `HarnessDescriptor` is a claim until startup inspection and conformance observe it. `negotiate` returns a typed refusal on any unmet **required** field; optional omissions are recorded. `HarnessPort` exposes `describe`, `negotiate`, `start`, `events`, `cancel`, `result`. `SandboxPort` exposes `lease`, `execute`, `release`, `health` and is only used for `local-sandbox`; caller-managed and provider-managed environments are explicitly identified.

Five adapter families share one conformance contract: in-process pydantic-ai(+harness), Claude Code, Codex, Devin and Grok. Each must prove MCP context, a planted permitted tool, a planted skill, denied tool absence, cancellation, event fidelity, usage quality and outcome reconciliation. API-key and subscription modes are supported only when the descriptor and current provider terms permit automation. A remote provider session is not represented as a local sandbox.

For swarm work, a committed `TopologyPlan` names width/rounds, `StopRule`, lease plan and per-node `SubagentAllowance`. AU constructs `ElasticTopologyAdmission` from that plan and stores the decision record digest in every spawned child. Continuation may stop early or release capacity. Re-admission is required to increase any cap. Outcome labels come from an independent evaluator. Remove `TopologyEngine.record_outcome` and `KGTeamComposer` success-rate selection at cutover; EG owns policy-backed topology facts and learned heads. AU never imports RDF/OWL/SHACL engines or authors a parallel ontology store.

## Invariants and failure handling

- Graph selection is a proposal until EG commits the decision and graph-os acquires every required capacity lease. Partial lease acquisition rolls back; one fresh re-decision is allowed on denial.
- Server policy is checked at run creation and privileged calls. Tool allowlists and CLI flags are defense in depth, not authorization.
- A timeout after possible side effect is `outcome-uncertain`; reconcile provider/session IDs, idempotency keys and EG receipts before retry or failover. No effect proven is a prerequisite for a new attempt.
- Missing events create `trace-incomplete` with high-watermark and gap cause. Usage quality is `measured`, `estimated`, or `unavailable`; `unavailable` never becomes zero or satisfies a strict budget.
- Release lease on terminal, stop, cancellation and verified reconciliation. Fence loss blocks additional effects.

## Contracts and migration

Consume only the public generated EG client and wire versions pinned in `pyproject.toml`/lock. Add contract golden vectors for RunSpec and event normalization; retain old-record decode support where the public contract requires it. Introduce adapters behind `HarnessPort`, route one production path through them, then delete obsolete selectors and local success writes in the same accepted tree. The [harness-evolution](../harness-evolution/spec.md) spec consumes terminal, independently evaluated runs and must not create a second scheduler.
