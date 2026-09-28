# Implementation plan: agent control plane

1. Pin the public EG generated-client version and add golden request/response fixtures for assembly, decisions, topology, capacity and run events. Implement typed refusals rather than compatibility guesses.
2. Define immutable `RunSpec`, `HarnessDescriptor`, `NegotiatedRunSpec`, `RunEvent`, `HarnessPort` and `SandboxPort`; add five adapters behind one conformance kit. Dependency-inject provider launchers so unit and PR tests use fixtures without accounts, GPU, live graph or subscription sessions.
3. Wire task mapping → assembly/Decide → commit → graph-os admission into an actual AU entry point. Include unknown-premise abstention and a single re-decision on capacity denial.
4. Convert topology consumers to use a committed plan. Build `ElasticTopologyAdmission` from plan caps, enforce `SubagentAllowance`, stop rules and narrow-only continuation. Remove local success-rate writes and selectors.
5. Emit durable run events with usage quality and reconciliation state. Add replay tests, independent evaluation, then enable learned terms only with promoted EG evidence.
6. Run focused, full and served fixtures. Update `evidence.md` with exact commit, CI URL, input digest, outcome and any limitation before declaring acceptance.

**Dependency order:** public EG contract → graph-os admission → AU port/consumer → end-to-end evidence. A stub client, a mock-only happy path, or a green source build alone does not close AC-02/03/06.
