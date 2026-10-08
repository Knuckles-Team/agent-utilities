# Implementation plan: agent control plane

1. Pin the public EG generated-client version and add golden request/response fixtures for assembly, decisions, topology, capacity and run events. Implement typed refusals rather than compatibility guesses.
2. Define immutable `RunSpec`, `HarnessDescriptor`, `NegotiatedRunSpec`, `RunEvent`, `HarnessPort` and `SandboxPort`; add five adapters behind one conformance kit. Dependency-inject provider launchers so unit and PR tests use fixtures without accounts, GPU, live graph or subscription sessions.
3. Wire task mapping → assembly/Decide → commit → graph-os admission into an actual AU entry point. Include unknown-premise abstention and a single re-decision on capacity denial.
4. Convert topology consumers to use a committed plan. Build `ElasticTopologyAdmission` from plan caps, enforce `SubagentAllowance`, stop rules and narrow-only continuation. Remove local success-rate writes and selectors.
5. Emit durable run events with usage quality and reconciliation state. Add replay tests, independent evaluation, then enable learned terms only with promoted EG evidence.
6. Run focused, full and served fixtures. Update `evidence.md` with exact commit, CI URL, input digest, outcome and any limitation before declaring acceptance.

**Dependency order:** public EG contract → graph-os admission → AU port/consumer → end-to-end evidence. A stub client, a mock-only happy path, or a green source build alone does not close AC-02/03/06.

## Task planner and cross-source reports (AU-CONTROL-R027–AU-CONTROL-R030)

**Reuse.** The planner composes existing parts only. `decide/consumers/assembly.py` supplies `assemble_mapped` and `spec_fields`. `decide/consumers/topology.py` supplies `ask_topology` and `plan_of`. `decide/topology/templates.py` supplies the reference templates and their DAG edges. The planner adds no solver, selector or success store. Ports cover the remaining facts: a capability search, a guardrail source and a compiled-workflow lookup. graph-os binds those ports to the EG coverage and guardrail queries (EG-DECISION-ENGINE-R126, EG-DECISION-ENGINE-R127). `graph_workflows compile` and `process_plan_compiler.py` back the workflow lookup.

**Module.** `agent_utilities/decide/consumers/task_planner.py` holds `TaskPlanner`, `TaskPlan`, `is_planning_question` and `install_task_planner`. `process_task_planner()` returns the installed planner, or one over the installed EG assembler and its sync driver. The driver runs EG coroutines off the event loop.

**Routing.** `dispatch_intent` answers a bare `ask` before capability ranking. A planning question returns the plan. A question that spans installed virtual sources returns the cross-source report. A hinted `ask` keeps its declared route.

**Virtual graphs.** `agent_utilities/knowledge_graph/virtual_graph/` holds the flow. `contracts.py` defines `SourceConnection`, `MetadataContract`, `VirtualMapping`, `metadata_triples` and `MaterializationPolicy`. `ontology.py` reads labels, ancestry and relations; `tbox_from_sparql` asks EG three bounded SPARQL questions. `federation.py` selects sources and runs live bind joins. `adapters.py` reads API, MCP, A2A and GraphQL sources through their discovered operations. SQL, Iceberg and Teradata-style sources bind through EG OBDA named virtual graphs (EG-UNIFIED-DATA-PLANE-R005).

**Interim boundary.** AU links concept classes over the relation facts EG returns. EG-FEDERATED-QUERY-R072 moves source selection into EG. AU then consumes that answer and retires its local path search.

**Live wiring.** graph-os installs the planner ports and the cross-source catalog at startup. Until then, the planner runs over the installed assembler, and the cross-source route stays inactive.
