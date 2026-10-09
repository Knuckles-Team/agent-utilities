# Tasks: agent control plane

- [ ] AC-01: Map free text to typed task IRIs as labelled claims; test unknown and conflicting mappings.
- [ ] AC-02/03: Add generated-client assembly and Decide calls in the live entry path; commit the exact record and pin digests.
- [ ] AC-04: Implement immutable RunSpec, ports, five adapters and common conformance fixtures.
- [ ] AC-04: Prove tools **and** skills available at startup, strict budget refusal and environment-mode honesty.
- [ ] AC-05: Carry topology and subagent caps to execution; delete local EMA selection and generic topology writes.
- [x] AC-05: Add the typed sub-agent allowance to `HarnessRequest` (`max_subagents`); the native adapter fails closed above zero (no native sub-agent tool to grant it through yet), the `claude-code` adapter denies its own `Task` tool below the allowance. Closes AU-CONTROL-R018.
- [ ] AC-05 follow-up: Bind `AgentSpec`/`request_for_spec` to the committed topology plan node's allowance so `max_subagents` carries a real decided value instead of the port's conservative default.
- [ ] AC-05: Implement stop, narrow, fence-loss and capacity-release behavior.
- [ ] AC-06: Stream normalized events, terminal receipts and incomplete/uncertain states; reconcile before retry.
- [ ] AC-07: Gate learned routing on independent outcome, calibration, promotion and exploration policy.
- [x] AC-08: Publish the swarm-topology ontology/shapes, ship the reference topology templates, and consume the generated topology contract via a pinned dependency, validating templates at publish. Closes AU-CONTROL-R009, AU-CONTROL-R010, AU-CONTROL-R011, AU-CONTROL-R012.
- [x] AC-09: Filter topology options by admissibility derivation, remove any local topology solver, respect capacity headroom at request priority, and allow at most one re-decision on capacity denial. Closes AU-CONTROL-R013, AU-CONTROL-R014, AU-CONTROL-R016, AU-CONTROL-R017.
- [x] AC-10: Access the harness and sandbox through direct typed ports, route model selection through the committed decision, and attribute catalog entries to their originating server. Closes AU-CONTROL-R021, AU-CONTROL-R022, AU-CONTROL-R023.
- [x] AC-08: Add the `AgentLibrary` store over `CallableResource` records, the `agent_library` intent tool, role-agent seeding from the `:Prompt` corpus, and the assembly commit, publish and library binding. Closes AU-CONTROL-R024, AU-CONTROL-R025, AU-CONTROL-R026.
- [ ] AC-08: graph-os calls `install_library_assembler` with its commit and publish context providers at serving start.
- [x] AC-11/AC-12: Ship the task planner and the `ask` planning route with provenance and named gaps. Closes AU-CONTROL-R027, AU-CONTROL-R028.
- [x] AC-13: Ship virtual-graph contracts and the cross-source report over live bind joins. Closes AU-CONTROL-R029, AU-CONTROL-R030.
- [ ] AC-11 follow-up: Bind the planner capability-search and guardrail ports to EG-DECISION-ENGINE-R126 and EG-DECISION-ENGINE-R127 in graph-os.
- [ ] AC-13 follow-up: Consume EG-FEDERATED-QUERY-R072 source selection and retire the local path search; bind SQL sources through EG OBDA.
- [ ] AC-14: Rank fleet discovery hits with native capabilities in `find`, with a `fleet.call` `how_to_call` and a reported probe failure. Closes AU-CONTROL-R031.
- [x] AC-14 follow-up: Accept `act(action='<server>.<tool>')`, map it to the prefixed fleet name, mount lazily and call through the governed `fleet.call` path. Closes AU-CONTROL-R032.
- [ ] AC-14 follow-up: Add server descriptions to the fleet catalog in graph-os so `find` ranks a server before its first probe.
- [x] AC-15: Run the declared default on a read-only action tie and send an unmatched `ask` to the NL planner. Closes AU-CONTROL-R033, AU-CONTROL-R034.
- [ ] Run the quality and served tests in `test-spec.md`; record merged-head results in `evidence.md`.

Checkboxes represent accepted deliverables, not merely files authored. Keep them open until their stated proofs pass.
