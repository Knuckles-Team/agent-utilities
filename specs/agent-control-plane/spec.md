# Agent control plane

**Owner:** agent-utilities (AU) · **Stable ID:** `AU-CONTROL-001`
**Delivery:** PARTIAL · **Acceptance:** NOT VERIFIED · **Scope IDs:** AU-CONTROL-R001, AU-CONTROL-R002–AU-CONTROL-R006, AU-CONTROL-R007, AU-CONTROL-R008, AU-CONTROL-R009–AU-CONTROL-R020 (AU portions), AU-CONTROL-R024–AU-CONTROL-R026, AU-CONTROL-R027–AU-CONTROL-R030. See [requirements.md](requirements.md) for the definition of every requirement ID and [status.json](status.json) for its delivery state and evidence.

## Outcome and actors

An operator or API caller submits a task. AU turns it into a typed, authorized, digest-bound execution plan, executes its nodes through a capability-negotiated harness and sandbox, and emits complete run events. The caller can explain why an agent, topology, model, skill, tool, harness and environment were selected. A contributor can implement this using this repository and the public contract owners listed below.

## Scope and ownership

AU owns task mapping proposals, L3 agent-graph orchestration, L4 `HarnessPort`/`SandboxPort`, routing consumers, per-node subagent allowance enforcement, and normalized L5 event emission. [epistemic-graph](https://github.com/Knuckles-Team/epistemic-graph) owns ontology, component catalog, `AgentAssemble`, `Decide`, `DecisionCommit`, capacity, durable run/outcome records and generated clients. [graph-os](https://github.com/Knuckles-Team/graph-os) owns ingress authorization, capacity admission, delegated execution and service transport. AU must not write ontology/RDF/SHACL, substitute its own decision or lease authority, or treat a model's choice as authorization.

## User stories and acceptance

1. **P0: safe plan.** Given typed task classes, component constraints, tenant and policy, AU asks the generated EG client for assembly/Decide, commits a record before side effects, and exposes selected and rejected candidates with premises and certificate. Unknown or unverified premises cause typed abstention.
2. **P0: executable run.** AU negotiates required tools, skills, trace fidelity, account mode, usage budget, environment and subagent allowance against a real harness descriptor. A missing required capability refuses launch. The run's immutable digest appears on each normalized event.
3. **P0: bounded swarm.** A committed topology plan determines width, rounds, stop rule and lease-bound subagent caps. AU may stop or narrow it; widening requires a fresh decision and capacity admission. A harness unable to enforce children starts with children disabled unless policy explicitly allows a measured token/cost cap.
4. **P1: reusable agents.** An operator saves an agent with its prompt, tools, skills, model profile, context policy and role. The web UI and the intent router list the same record. Each fleet package prompt with a role becomes a pre-built role agent. Each solved assembly becomes an `agent_graph` record for reuse.
5. **P0: task plan on request.** An operator asks "How can I execute task X?" through the `ask` intent. AU returns one plan with steps, agents, reuse, guardrails, workflow and DAG. Each element cites the EG record, catalog hit, policy rule or claim behind it. A missing authority appears as a named gap.
6. **P1: cross-source report.** An operator asks a question that spans systems, such as "How does the supply chain affect our services?". The ontology selects the sources, and AU reads them live through approved virtual mappings. Only metadata enters the graph.
7. **P1: learning with evidence.** Independent outcomes can inform future routing through promoted EG decision heads. Self-reported success, uncalibrated EMAs and incomplete traces cannot authorize a route.

## Requirements

| ID | Requirement | Scope IDs | Proof |
|---|---|---|---|
| AC-01 | Preserve task IRI and evidence class; free-text mapping is a labelled claim, never a proof. | AU-CONTROL-R008 | mapping and refusal tests |
| AC-02 | Route model, prompt, skill, tool, harness, account mode and sandbox through one committed EG decision; bind selected component digests. | AU-CONTROL-R002–AU-CONTROL-R006 | client contract and served integration |
| AC-03 | Use `AgentAssemble` for the L3 graph and `Decide` for legal-option selection; consume generated clients and no Python shadow solver. | AU-CONTROL-R001 | deterministic fixture and source gate |
| AC-04 | Negotiate the entire RunSpec before start; verify actual startup tool and skill exposure, strict budgets, policy, isolation and fidelity. | AU-CONTROL-R001 | five-adapter conformance kit |
| AC-05 | Carry topology plan, stop rule and `SubagentAllowance` into node execution; enforce narrow-only continuation and fence loss. | AU-CONTROL-R007, AU-CONTROL-R015, AU-CONTROL-R018, AU-CONTROL-R019 | topology and cancellation tests |
| AC-06 | Emit launch, step, tool, usage, artifact, receipt and terminal events through the EG durable contract; mark gaps and uncertain outcomes explicitly. | AU-CONTROL-R001 | replay and fault tests |
| AC-07 | Use independently evaluated outcomes for any learned routing term; exploration defaults off and is forbidden for sensitive or irreversible work. | AU-CONTROL-R020 | calibration and negative fixtures |
| AC-08 | Store built, role and assembled agents in one Agent Library. Expose list, get and save through the web API and the intent router. Save every solved assembly for reuse, and publish it after its decision commits. | AU-CONTROL-R024–AU-CONTROL-R026 | library, intent-router and assembly-binding tests |
| AC-11 | Compose one structured task plan from EG assembly, the EG topology decision, capability search, guardrails and workflow lookup. Cite provenance per element. Name every missing port as a gap. | AU-CONTROL-R027 | `tests/unit/decide/test_task_planner.py` |
| AC-12 | Route a bare planning `ask` to the task planner. Keep a hinted `ask` on its declared route. | AU-CONTROL-R028 | `tests/unit/knowledge_graph/test_virtual_graph.py` routing cases |
| AC-13 | Select sources for a cross-source question by ontology. Join live rows over approved virtual mappings. Copy no entity row. | AU-CONTROL-R029, AU-CONTROL-R030 | `tests/unit/knowledge_graph/test_virtual_graph.py` |

## Completion measure

All five supported harness families pass one shared conformance kit; a served end-to-end fixture proves `AgentAssemble → Decide → DecisionCommit → admission → run → terminal receipt`; no alternate AU topology success store or unauthorized fallback remains. Each requirement has exact merged-head CI evidence in `evidence.md`. Source presence alone is not acceptance.

Requirement IDs are defined in [requirements.md](requirements.md); delivery state per ID is in `status.json`.
