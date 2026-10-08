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

