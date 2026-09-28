# Agent control plane

**Owner:** agent-utilities (AU) · **Stable ID:** `agent-control-plane`
**Delivery:** PARTIAL · **Acceptance:** NOT VERIFIED · **Scope IDs:** RF-029, EH-035–EH-039, EH-048, EH-206, EH-453–EH-464 (AU portions)

## Outcome and actors

An operator or API caller submits a task. AU turns it into a typed, authorized, digest-bound execution plan, executes its nodes through a capability-negotiated harness and sandbox, and emits complete run events. The caller can explain why an agent, topology, model, skill, tool, harness and environment were selected. A contributor can implement this using this repository and the public contract owners listed below.

## Scope and ownership

AU owns task mapping proposals, L3 agent-graph orchestration, L4 `HarnessPort`/`SandboxPort`, routing consumers, per-node subagent allowance enforcement, and normalized L5 event emission. [epistemic-graph](https://github.com/Knuckles-Team/epistemic-graph) owns ontology, component catalog, `AgentAssemble`, `Decide`, `DecisionCommit`, capacity, durable run/outcome records and generated clients. [graph-os](https://github.com/Knuckles-Team/graph-os) owns ingress authorization, capacity admission, delegated execution and service transport. AU must not write ontology/RDF/SHACL, substitute its own decision or lease authority, or treat a model's choice as authorization.

## User stories and acceptance

1. **P0: safe plan.** Given typed task classes, component constraints, tenant and policy, AU asks the generated EG client for assembly/Decide, commits a record before side effects, and exposes selected and rejected candidates with premises and certificate. Unknown or unverified premises cause typed abstention.
2. **P0: executable run.** AU negotiates required tools, skills, trace fidelity, account mode, usage budget, environment and subagent allowance against a real harness descriptor. A missing required capability refuses launch. The run's immutable digest appears on each normalized event.
3. **P0: bounded swarm.** A committed topology plan determines width, rounds, stop rule and lease-bound subagent caps. AU may stop or narrow it; widening requires a fresh decision and capacity admission. A harness unable to enforce children starts with children disabled unless policy explicitly allows a measured token/cost cap.
4. **P1: learning with evidence.** Independent outcomes can inform future routing through promoted EG decision heads. Self-reported success, uncalibrated EMAs and incomplete traces cannot authorize a route.

## Requirements

| ID | Requirement | Scope IDs | Proof |
|---|---|---|---|
| AC-01 | Preserve task IRI and evidence class; free-text mapping is a labelled claim, never a proof. | EH-206 | mapping and refusal tests |
| AC-02 | Route model, prompt, skill, tool, harness, account mode and sandbox through one committed EG decision; bind selected component digests. | EH-035–EH-039 | client contract and served integration |
| AC-03 | Use `AgentAssemble` for the L3 graph and `Decide` for legal-option selection; consume generated clients and no Python shadow solver. | RF-029 | deterministic fixture and source gate |
| AC-04 | Negotiate the entire RunSpec before start; verify actual startup tool and skill exposure, strict budgets, policy, isolation and fidelity. | RF-029 | five-adapter conformance kit |
| AC-05 | Carry topology plan, stop rule and `SubagentAllowance` into node execution; enforce narrow-only continuation and fence loss. | EH-048, EH-459, EH-462, EH-463 | topology and cancellation tests |
| AC-06 | Emit launch, step, tool, usage, artifact, receipt and terminal events through the EG durable contract; mark gaps and uncertain outcomes explicitly. | RF-029 | replay and fault tests |
| AC-07 | Use independently evaluated outcomes for any learned routing term; exploration defaults off and is forbidden for sensitive or irreversible work. | EH-464 | calibration and negative fixtures |

## Completion measure

All five supported harness families pass one shared conformance kit; a served end-to-end fixture proves `AgentAssemble → Decide → DecisionCommit → admission → run → terminal receipt`; no alternate AU topology success store or unauthorized fallback remains. Each requirement has exact merged-head CI evidence in `evidence.md`. Source presence alone is not acceptance.
