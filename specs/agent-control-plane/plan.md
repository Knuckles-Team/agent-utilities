# Implementation plan: agent control plane

1. Pin the public EG generated-client version and add golden request/response fixtures for assembly, decisions, topology, capacity and run events. Implement typed refusals rather than compatibility guesses.
2. Define immutable `RunSpec`, `HarnessDescriptor`, `NegotiatedRunSpec`, `RunEvent`, `HarnessPort` and `SandboxPort`; add five adapters behind one conformance kit. Dependency-inject provider launchers so unit and PR tests use fixtures without accounts, GPU, live graph or subscription sessions.
3. Wire task mapping → assembly/Decide → commit → graph-os admission into an actual AU entry point. Include unknown-premise abstention and a single re-decision on capacity denial.
4. Convert topology consumers to use a committed plan. Build `ElasticTopologyAdmission` from plan caps, enforce `SubagentAllowance`, stop rules and narrow-only continuation. Remove local success-rate writes and selectors.
5. Emit durable run events with usage quality and reconciliation state. Add replay tests, independent evaluation, then enable learned terms only with promoted EG evidence.
6. Run focused, full and served fixtures. Update `evidence.md` with exact commit, CI URL, input digest, outcome and any limitation before declaring acceptance.

**Dependency order:** public EG contract → graph-os admission → AU port/consumer → end-to-end evidence. A stub client, a mock-only happy path, or a green source build alone does not close AC-02/03/06.

## Agent Library and assembly binding (AU-CONTROL-R024–R026)

- **Store.** `agent_utilities/orchestration/agent_library.py` owns `AgentLibrary` and `AgentRecord`. A record is the existing `CallableResource` node. Local and role agents use `resource_type=AGENT_SKILL` with the runnable-skill contract. Assembled graphs use `resource_type=AGENT_GRAPH` and are not runnable. No new node label exists.
- **EG record reuse.** The record fields mirror EG's `AgentLibraryEntryDraft`. EG's `AgentLibrary` method has no list operation and needs a mutation context from graph-os. A later lane publishes records to it when graph-os binds that context.
- **Surfaces.** The `agent_library` tool (verbs `manage`, `find`, `ask`; read-only actions `list`, `get`) and the agent-webui routes call `AgentLibrary`. The webui passes its tenant and commons union reader.
- **Role agents.** `seed_role_agents` reads the `:Prompt` corpus. It reads no workspace path.
- **Assembly.** `Assembler` gains `publish_context` and `library`. `install_library_assembler` binds both with the session clients. graph-os owns the call that supplies the context providers.

## Task planner and cross-source reports (AU-CONTROL-R027–AU-CONTROL-R030)

**Reuse.** The planner composes existing parts only. `decide/consumers/assembly.py` supplies `assemble_mapped` and `spec_fields`. `decide/consumers/topology.py` supplies `ask_topology` and `plan_of`. `decide/topology/templates.py` supplies the reference templates and their DAG edges. The planner adds no solver, selector or success store. Ports cover the remaining facts: a capability search, a guardrail source and a compiled-workflow lookup. graph-os binds those ports to the EG coverage and guardrail queries (EG-DECISION-ENGINE-R126, EG-DECISION-ENGINE-R127). `graph_workflows compile` and `process_plan_compiler.py` back the workflow lookup.

**Module.** `agent_utilities/decide/consumers/task_planner.py` holds `TaskPlanner`, `TaskPlan`, `is_planning_question` and `install_task_planner`. `process_task_planner()` returns the installed planner, or one over the installed EG assembler and its sync driver. The driver runs EG coroutines off the event loop.

**Routing.** `dispatch_intent` answers a bare `ask` before capability ranking. A planning question returns the plan. A question that spans installed virtual sources returns the cross-source report. A hinted `ask` keeps its declared route.

**Virtual graphs.** `agent_utilities/knowledge_graph/virtual_graph/` holds the flow. `contracts.py` defines `SourceConnection`, `MetadataContract`, `VirtualMapping`, `metadata_triples` and `MaterializationPolicy`. `ontology.py` reads labels, ancestry and relations; `tbox_from_sparql` asks EG three bounded SPARQL questions. `federation.py` selects sources and runs live bind joins. `adapters.py` reads API, MCP, A2A and GraphQL sources through their discovered operations. SQL, Iceberg and Teradata-style sources bind through EG OBDA named virtual graphs (EG-UNIFIED-DATA-PLANE-R005).

**Interim boundary.** AU links concept classes over the relation facts EG returns. EG-FEDERATED-QUERY-R072 moves source selection into EG. AU then consumes that answer and retires its local path search.

**Live wiring.** graph-os installs the planner ports and the cross-source catalog at startup. Until then, the planner runs over the installed assembler, and the cross-source route stays inactive.

## Harness configuration on the envelope (AU-CONTROL-R035.3 to AU-CONTROL-R036)

Data flow: `committed HarnessConfig decision -> decide_harness_config (decide/consumers/harness_config.py) -> producer (core/sessions.py:1231, mcp/tools/job_tools.py:89, integrations/git_resolver.py:234) -> AgentTurnEnvelope.harness_config (agent_dispatch.py:471) -> DispatchCarrier signature (agent_dispatch.py:180) -> enqueue_agent_turn (:737) -> worker verify (:350) -> run_agent`.

Model selection today runs `route_model` (`decide/consumers/routing.py:58`), `decide.choose` (`decide/__init__.py:67`), `DecisionRunner.choose` (`decide/runner.py:218`) and the transport (`decide/transport.py:28`). The new consumer uses the same runner and transport. It differs in one respect: the answer is a structured `HarnessConfigAnswer`, not one option id. AU applies it unchanged.

Already present: `allowed_tool_subset` is on the carrier (`agent_dispatch.py:210`) and on the envelope (`:509`), signed and checked at `:445`. New: the `harness_config` field, its signature binding, the verify argument, the absent-field refusal and the decision consumer.

Decisions: the field is typed, not a free map, so the signature digest is canonical. Rejected: AU computing the config from model facts. That would be a second selector beside the engine. An abstention stops the enqueue, because a guess would bypass the decision ladder.

Depends on the `EG` row for the `HarnessConfig` decision kind and the `GRAPHOS-A2A` admission row that supplies the context budget. Unblocks `AU-HARNESS-R012`.

## Design

### Reuse and live wiring

The checkout already contains `agent_utilities/graph/builder.py`, `graph/routing/`, `graph/subagent_patterns.py`, `graph/topology_engine.py`, `graph/team_composer.py`, `rlm/sandboxes/`, and `knowledge_graph/research/loops.py`. Those are migration inputs, not evidence that the target contract is wired. The current tree does **not** contain `agent_utilities/layers/` or `agent_utilities/decide/`; do not claim the proposed port and consumer implementations are landed. Preserve working execution behavior while replacing selection and success-rate authority at a single cutover. EG and graph-os contracts must be available as public tagged/generated dependencies, with in-memory test doubles for unit tests.

### Data flow and interfaces

```text
task + tenant/policy → typed mapping claim → EG AgentAssemble / Decide
 → DecisionCommit(snapshot checked) → graph-os capacity admission
 → AU L3 node scheduler → HarnessPort.negotiate(RunSpec, descriptor, policy)
 → SandboxPort.lease (if local) → HarnessPort.start/events/result
 → EG durable run events + independent outcome → release leases
```

`RunSpec` is immutable and digest-bound: task/context handles, L1/L2/L3 component digests, required/optional capabilities, toolset and skill digests, minimum trace fidelity, policy/auth references, account and environment modes, usage units and hard/advisory budgets, deadline/cancel semantics, artifacts, completion predicate, side-effect/idempotency and reconciliation policy. `HarnessDescriptor` is a claim until startup inspection and conformance observe it. `negotiate` returns a typed refusal on any unmet **required** field; optional omissions are recorded. `HarnessPort` exposes `describe`, `negotiate`, `start`, `events`, `cancel`, `result`. `SandboxPort` exposes `lease`, `execute`, `release`, `health` and is only used for `local-sandbox`; caller-managed and provider-managed environments are explicitly identified.

Five adapter families share one conformance contract: in-process pydantic-ai(+harness), Claude Code, Codex, Devin and Grok. Each must prove MCP context, a planted permitted tool, a planted skill and denied tool absence. Each must also prove cancellation, event fidelity, usage quality and outcome reconciliation. API-key and subscription modes are supported only when the descriptor and current provider terms permit automation. A remote provider session is not represented as a local sandbox.

For swarm work, a committed `TopologyPlan` names width/rounds, `StopRule`, lease plan and per-node `SubagentAllowance`. AU constructs `ElasticTopologyAdmission` from that plan and stores the decision record digest in every spawned child. Continuation may stop early or release capacity. Re-admission is required to increase any cap. Outcome labels come from an independent evaluator. Remove `TopologyEngine.record_outcome` and `KGTeamComposer` success-rate selection at cutover; EG owns policy-backed topology facts and learned heads. AU never imports RDF/OWL/SHACL engines or authors a parallel ontology store.

### Invariants and failure handling

- Graph selection is a proposal until EG commits the decision and graph-os acquires every required capacity lease. Partial lease acquisition rolls back; one fresh re-decision is allowed on denial.
- Server policy is checked at run creation and privileged calls. Tool allowlists and CLI flags are defense in depth, not authorization.
- A timeout after possible side effect is `outcome-uncertain`; reconcile provider/session IDs, idempotency keys and EG receipts before retry or failover. No effect proven is a prerequisite for a new attempt.
- Missing events create `trace-incomplete` with high-watermark and gap cause. Usage quality is `measured`, `estimated`, or `unavailable`; `unavailable` never becomes zero or satisfies a strict budget.
- Release lease on terminal, stop, cancellation and verified reconciliation. Fence loss blocks additional effects.

### Contracts and migration

Consume only the public generated EG client and wire versions pinned in `pyproject.toml`/lock. Add contract golden vectors for RunSpec and event normalization; retain old-record decode support where the public contract requires it. Introduce adapters behind `HarnessPort` and route one production path through them. Delete obsolete selectors and local success writes in the same accepted tree. The [harness-evolution](../harness-evolution/spec.md) spec consumes terminal, independently evaluated runs and must not create a second scheduler.

## Skill workflow tool-gap filling (AU-CONTROL-R037)

Data flow: `workflow template slots -> find_workflow_gaps / fill_workflow_gaps (decide/consumers/workflow_gaps.py) -> execute_workflow (orchestration/engine.py:2407) -> assembled agent graph`.

Attach point: `graph_orchestrate` (`mcp/tools/analysis_tools.py:4223`) reaches `execute_workflow`. The first child row locates the exact call and wires it.

Reuse: `assemble_mapped` (`decide/consumers/assembly.py:220`) builds the graph. The `find` ranking supplies candidates.

Decision: the engine's committed choice fills each slot. The rejected alternative is an LLM that proposes a substitute tool, which creates a second selector.

## Restricted interpreter for model-written code (AU-CONTROL-R038)

Data flow: `metric-scoring run loop -> _run_metric_isolated (agent_utilities/harness/online_scoring.py:70) -> Router.select (rlm/sandboxes/router.py:59) -> Sandbox.execute (rlm/sandboxes/base.py:147) -> SandboxResult or SandboxRejected -> next tier`.

Types: `SandboxCapabilities` (`base.py:44`) gains `isolation`; `ResourceLimits` holds wall time, memory and call count. The backends and the router already exist and are reused. The new work is the closed isolation value, the limit record and the trace fields.

Attach point: the chain loop at `online_scoring.py:106`, and the same loop at `change_synthesis.py:276`.

Failure: a rejected construct escalates to the next tier. An exhausted chain returns a typed refusal and never runs the code unrestricted.

Decision: host-controlled helpers are the only door out of the interpreter. The rejected alternative is a deny-list over a general interpreter, which fails open on a name the list misses.

## Fan-out sandbox branching (AU-CONTROL-R039)

Data flow: `served fork fan-out -> _fork_fanout (mcp/tools/engine_surface_tools.py:1694) -> ForkableSandbox.branch (rlm/sandboxes/base.py:222) -> warm registry -> new parent handle -> audit and metric`.

Types: `BranchRequest`, `BranchResult` and `BranchRefusal` with a closed reason set. `ForkableSandbox.execute` and the warm registry are reused. The call from `_fork_fanout` to `branch` is new.

Failure: a rung without snapshot support returns `BranchRefusal`. A host that cannot be reached returns `host_unavailable`. Neither path silently cold-starts.

Decision: the warm registry owns every parent, so a branch is one more registered parent. The rejected alternative is a per-call snapshot cache, which duplicates parent lifetime rules.

## The harness holds run output uncommitted and applies the verdict it receives (AU-CONTROL-R040)

Data flow: `execute_agent -> hold layer (new) -> run tools write to the layer -> FsCarrier.snapshot (carrier.py:121) -> RunEventLog.append (kernel.py:160) -> HeldItemVerdict from graph-os -> apply | drop local layer | keep held -> base tree`.

Existing code: `agent_utilities/runtime/run_vcs/kernel.py:145`, `agent_utilities/runtime/run_vcs/carrier.py:101`, `agent_utilities/runtime/run_vcs/run_session.py:39`, `agent_utilities/runtime/run_vcs/run_commit.py:30`. Reachable today: no for the carrier; yes for replay only.

Moves to GRAPHOS-A2A: the admission outcome of a proposal. Moves to EG-DECISION-ENGINE: the run's durable record. Existing code that implements a moved part: `agent_utilities/knowledge_graph/enrichment/writeback/approval.py:47` (approval queue).

Cross-repository rows: GRAPHOS-A2A-R006 — return the admitted, rejected or abstained outcome for a proposal; EG-DECISION-ENGINE-R133 — store the run's durable record.

The adapter preserves `HeldItemVerdict.outcome` and `verdict_ref`. `admitted` applies the layer; `rejected` discards only the local layer. `abstained`, unavailable responses and typed refusals preserve the held proposal with no filesystem effect. The engine retains the authoritative held-item record. The harness makes no admission decision. The rejected alternative is a harness-side review queue, which decides admission outside graph-os.

## The harness runs agent runtimes isolated and enforces the quota it receives (AU-CONTROL-R041)

Data flow: `served fork fan-out -> _fork_fanout (engine_surface_tools.py:1694) -> backend start with received RuntimeQuota (container_fork_backend.py) -> signal handler -> grace timer -> kill group -> StopReason -> trace events`.

Existing code: `agent_utilities/rlm/sandboxes/container_fork_backend.py:54`, `agent_utilities/core/cgroup_resources.py:52`, `agent_utilities/rlm/sandboxes/_bridge.py:287`, `agent_utilities/deployment/skill_validation.py:590`. Reachable today: no for quotas and signal handling; the PID 1 expiry is reached through the backend.

Moves to EG-DECISION-ENGINE: the quota values. No agent-utilities code implements them today.

Cross-repository rows: EG-DECISION-ENGINE-R133 — produce the quota values for a run.

The decision: the runtime owns its own stop. The rejected alternative is a supervisor that only kills on timeout, which loses in-flight results. `ResourceLimits` of AU-CONTROL-R038.1 is reused for the call-count limit.

## The harness loads only the tools it is given during a run (AU-CONTROL-R042)

Data flow: `execute_agent -> GraphOS find -> EG selected ids -> GraphOS authorized schemas -> factory.py:973 -> DisclosureSet -> gate -> GraphOS act -> tool`.

Existing code: `agent_utilities/capabilities/fleet_tool_search.py:54`, `agent_utilities/capabilities/fleet_tool_search.py:84`, `agent_utilities/capabilities/fleet_tool_search.py:198`, `agent_utilities/agent/factory.py:529`. Reachable today: no for the fleet capability; yes for deferred local toolsets.

Moves to GRAPHOS-FLEET: the tool registry and on-demand disclosure. Moves to EG-DECISION-ENGINE: tool choice. Existing code that implements a moved part: `agent_utilities/capabilities/fleet_tool_search.py:54` (relevance search) and `:84`.

Cross-repository rows: GRAPHOS-FLEET-R011 owns canonical registry digests and authorized `find`. GRAPHOS-FLEET-R013 projects typed schemas, `graphos://` resources and the active digest. GRAPHOS-FLEET-R029 owns the intent surface: `find` discovers, `manage` mounts/releases and `act` dispatches. AU must not restore retired fleet meta-tools or per-session visibility middleware. EG-DECISION-ENGINE-R134.2 owns relevance filtering followed by R045 coverage-per-token selection; AU preserves that result.

CONFIRMED: `fleet_relevance_search` in `fleet_tool_search.py:54` is the selection code named by EG-R134.2. `FleetToolset` at `:84` probes and mounts through the local multiplexer. That registry/disclosure responsibility moves to GraphOS-R011/R013/R029. Retain the AU adapter and call gate; remove local ranking and registry authority only after consumer cutover evidence.

BLOCKED for runtime cutover: require served R134.2 selection and generated consumer contracts for selected ids, authorized schemas and registry generation. Pin their run/generation association before loading. A missing or inconsistent handoff must cause no tool call. GraphOS producer tests cannot close AU factory wiring or the served disclosure gate.

The decision: the gate checks the given set at call time. The rejected alternative is a prompt instruction not to call unseen tools, which a model can ignore. R031 and R032 stay unchanged.

## The harness runs the given fan-out in parallel and returns results (AU-CONTROL-R043)

Data flow: `served fork fan-out -> _fork_fanout (engine_surface_tools.py:1694) -> received SubtaskSet check -> one sandbox per subtask -> FanoutResult -> aggregation -> trace events`.

Existing code: `agent_utilities/graph/coordination.py:51`, `agent_utilities/graph/coordination.py:255`, `agent_utilities/graph/hierarchical_coordination.py:65`, `agent_utilities/mcp/tools/engine_surface_tools.py:1694`. Reachable today: no for aggregation of fan-out results.

Moves to EG-DECISION-ENGINE: decomposition into independent subtasks and routing to specialists. Existing code that implements a moved part: `agent_utilities/graph/planning/__init__.py:165` (decomposition).

Cross-repository rows: EG-DECISION-ENGINE-R134 — decompose the task, route the subtasks and set the independence flag.

The decision: subtasks share nothing, so aggregation is the only join. The rejected alternative is a shared scratchpad between agents, which couples failures. Width stays with AU-CONTROL-R007. References: [arXiv:2605.15132] Rose et al., APWA.

References:
- arXiv:2605.15132 — Rose et al., "APWA: A Distributed Architecture for Parallelizable Agentic Workflows" (2026)

## The harness exposes file and shell tools that forward to the client (AU-CONTROL-R044)

Data flow: `ACP session -> session_config (protocols/acp_adapter.py:273) -> GraphACPDeps.transport -> run loop -> ClientCapabilityToolset tool -> ClientCapabilityTransport.call -> graph-os session -> client`.

Types: `ClientCapabilityToolset`, `ClientCapabilityTransport`, `client_capability_refused`. The toolset reads the tool selection and limits from `HarnessConfig`.

Attach point: `session_config` (`protocols/acp_adapter.py:273`), called by `PydanticAIACPAgent` for each new session. `build_swe_agent` (`orchestration/swe_agent.py:44`) is the second attach point for non-ACP runs.

Failure behavior: a missing client, an unselected tool or a client refusal returns a typed refusal. No local fallback exists.

Existing code: `agent_utilities/tools/swe_workspace_tools.py:51`, `agent_utilities/tools/swe_workspace_tools.py:69`, `agent_utilities/tools/swe_workspace_tools.py:87`, `agent_utilities/tools/swe_workspace_tools.py:100`, `agent_utilities/tools/swe_workspace_tools.py:143`, `agent_utilities/tools/tool_registry.py:247`, `agent_utilities/tools/tool_registry.py:323`, `agent_utilities/runtime/workspace.py:108`, `agent_utilities/orchestration/swe_agent.py:44`, `agent_utilities/orchestration/engine.py:1709`, `agent_utilities/protocols/acp_adapter.py:160`, `agent_utilities/protocols/acp_adapter.py:273`, `agent_utilities/core/execution/adapters/base.py:34`. Reachable today: no for client-capability forwarding. The local workspace tools are reachable only through `run_swe_task`. The default-off `DEFAULT_SWE_TOOLS` also reaches them. Search terms: read_file, write_file, apply_patch, run_command. More terms: search_files, edit_file, shell_tool. Protocol terms: client capability, fs/read_text_file, terminal/create, request_permission. The adapter file `core/execution/adapters/base.py` names only the `acp-json-rpc` stream format at `:34`.

Reused: the `trace` decorator, `tool_version`, `PersistencePrivacyGuard` and the typed action names. New: the transport protocol, the five client-method tools and `apply_patch` and `search_files`.

Cross-repository rows: GRAPHOS-ACP `CodingSessionAdmission` admits the workspace root, records the grant and forwards each request to the client. EG-DECISION-ENGINE `HarnessConfig` supplies tool selection and limits. TUI-RUNTIME `CodingSessionView` implements the client capabilities with path containment and approval.

The decision: execution is a client call because the files live on the person's host. The rejected alternative is a cluster service that mounts or syncs the host checkout. It copies data and needs host credentials on the cluster.

## Amendment 2026-10-17: reference-only single-tool worker

R035.8 extends `AgentTurnEnvelope` with a tool-task kind using the existing
signed `payload_ref` and one-element `allowed_tool_subset`. The payload reference is the digest of the complete immutable owner request.
The request includes caller authority, tenant, session, operation, parameters,
policy revision and registry digest. Parameters are canonical JSON data, never a prompt.
The queue and WorkItem retain only the reference. WorkItem metadata sanitization
must not rewrite exact operation inputs.

`api/tool_task.py` defines the request, bound result and owner dispatcher port.
`agent_dispatch_worker` receives that port from composition and invokes it only
inside the existing WorkItem lease. The owner resolves immutable inputs and
reauthorizes the original caller immediately before effects, with durable
idempotency on the signed dispatch job ID. It must refuse uncertain replay.
The owner persists and retrieves the full immutable bound result receipt.
Indeterminate outcomes leave the WorkItem nonterminal and withhold broker
acknowledgment until owner reconciliation returns a definitive bound result.
Unexpected execution exceptions and malformed post-effect receipts are also
indeterminate. Only a proven pre-effect refusal can terminally refuse execution.
Expired tool-task deadlines admit no new effect and cannot prove cancellation
of a prior effect; nonterminal work remains unacknowledged for owner reconciliation.
Rejected tool-task carriers also grant no execution and receive no broker ACK;
untrusted kind can only conservatively withhold acknowledgment. Authenticated
renewal/reconciliation requires a fresh owner-authenticated carrier and is not
implemented, so poison deliveries may remain held. Expired authority is never reused.
The worker checks input binding before execution and result binding before a
successful terminal commit. An absent port performs no effect.

The actual consumer loop and pool pass this dependency through; no process-global
registry or local tool executor is introduced. Tests qualify worker wiring with
controlled ports only. GraphOS must still implement the real admitted-request
store, current-caller resolver and shared invocation dispatcher before enabling
one-shot submission. Signed approval and prompt suppression remain unavailable.

Native retry exhaustion remains a production-owner prerequisite. The native claim
operation can atomically mark scheduling as dead-letter after exhausting its attempt budget. That state does not prove that an uncertain tool effect
failed, nor does it resolve an unknown receipt. A production owner must durably
coordinate scheduling terminal states with effect reconciliation before this
port is composed. The controlled retry-budget tests qualify worker boundaries
only; a pre-read check cannot repair that native authority contract atomically.
