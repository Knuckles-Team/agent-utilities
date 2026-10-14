# Test contract: agent control plane

| Scenario | Layer and fixture | Expected result |
|---|---|---|
| Typed task with published components | generated-client contract + served in-memory EG | committed graph/decision; digest and why-not recorded |
| Unknown or free-text-only task | unit and served negative | labelled claim or typed abstention; no execution |
| Missing required tool, skill, trace fidelity, budget meter, or account mode | each adapter conformance | refusal before launch; no side effect |
| Planted permitted and forbidden tools plus planted skill | all five adapter fixture launches | permitted tool/skill observed, forbidden tool absent |
| Sandbox and remote mode claims | port contract | no local lease for provider-managed remote; untrusted code refused on weak mode |
| Capacity denial during multi-cell admission | graph-os/AU integration | partial leases released, at most one new decision, no heuristic execution |
| Harness without enforceable child cap | adapter negative | children disabled or launch refused by policy |
| Continuation asks for greater width/depth/rounds | topology unit | refuse; new decision and lease needed |
| Fence loss, timeout after dispatch, missing event, unavailable usage | fault injection | stop/reconcile; `outcome-uncertain`/`trace-incomplete`; no blind retry or zero cost |
| Self-reported success or unpromoted model | routing negative | no selection authority or promotion |
| Agent saved through the library | `tests/unit/orchestration/test_agent_library.py` | listed and read back with every field; runnable-skill contract kept |
| Library agent through the intent router | `tests/unit/orchestration/test_agent_library_intent.py` | `find` ranks `agent_library`; `ask` lists it; `ask` cannot save; `manage` saves after a previewed plan |
| Packaged prompt with and without a role | `tests/unit/orchestration/test_agent_library.py` | role blueprint becomes a role agent with package tools; no role, no agent |
| Solved, uncommitted and failed-publish assemblies | `tests/unit/decide/test_assembly_consumer.py` | committed graph is published and saved. Uncommitted graph is saved only. Publish failure keeps the agent |
| Planning question with scripted solved EG answers | unit, `tests/unit/decide/test_task_planner.py` | decided topology, per-agent components, reuse sources, policy guardrails, compiled workflow. Decision provenance per element. Goal text absent from EG requests |
| Planning question with EG abstention or no ports | unit | named gaps; no model composition; fallback template labelled `fallback` |
| Bare, non-planning and hinted `ask` | unit routing | plan for bare planning text; no route for code questions or hinted calls |
| Cross-source question over two fake sources | unit, `tests/unit/knowledge_graph/test_virtual_graph.py` | ontology selects both sources. Live bind joins. Zero materialized rows. Key filter pushed only with `filter:in` |
| Unapproved mapping, row budget, credential ref, undiscovered field | unit negative | uncovered class with no reads. Incomplete report. Refused connection. Refused mapping |
| Fleet hit for "list Docker containers" | unit, `tests/unit/test_intent_fleet_ranking_and_ties.py` | fleet row ranks first; server named; `how_to_call` uses `fleet.call`; failed probe reported as `fleet_error` |
| Read-only action tie and unmatched `ask` | unit, `tests/unit/test_intent_fleet_ranking_and_ties.py` | tie runs the declared default. No read default refuses. Unmatched `ask` routes to `nl_query`. Pinned calls never fall back |
| R035.3 Typed `harness_config` envelope field. | unit, fake engine | `tests/unit/test_agent_dispatch.py::test_envelope_carries_harness_config`. |
| R035.4 Sign the field. | unit, fake engine | `tests/unit/test_agent_dispatch.py::test_signature_covers_harness_config`. |
| R035.5 Reject a changed field. | unit, fake engine | `tests/unit/test_agent_dispatch.py::test_verify_rejects_tampered_harness_config`. |
| R035.6 Refuse an absent field. | unit, fake engine | `tests/unit/test_agent_dispatch_worker.py::test_absent_harness_config_refused`. |
| R035.7 Producers set the field. | unit, fake engine | `tests/unit/test_agent_dispatch.py::test_producers_enqueue_with_committed_config`. |
| R036.1 Name the shared path. | unit, fake engine | `tests/unit/test_decide_harness_config.py::test_consumer_uses_shared_runner_and_transport`. |
| R036.2 Typed answer model. | unit, fake engine | `tests/unit/test_decide_harness_config.py::test_answer_model_matches_contract_fixture`. |
| R036.3 Request function. | unit, fake engine | `tests/unit/test_decide_harness_config.py::test_request_inputs_and_answer_mapping`. |
| R036.4 Abstention is not a guess. | unit, fake engine | `tests/unit/test_decide_harness_config.py::test_abstention_carries_missing_fact`. |
| R036.5 Wire into the producers. | unit, fake engine | `tests/unit/test_agent_dispatch.py::test_producer_calls_decision_before_envelope`. |
| R036.6 Trace the decision. | unit, fake engine | `tests/unit/test_decide_harness_config.py::test_decision_id_in_envelope_and_trace`. |
| R036.7 End-to-end proof. | unit, fake engine | `tests/harness/test_runloop_envelope_e2e.py::test_decision_to_signed_envelope_to_run`. |
| R037.1 Typed gap model. | unit, fake engine | `tests/unit/test_workflow_gaps.py::test_gap_model_fields`. |
| R037.2 Detect unmet slots. | unit, fake engine | `tests/unit/test_workflow_gaps.py::test_unmet_slot_becomes_gap`. |
| R037.3 Fill from the committed ranking. | unit, fake engine | `tests/unit/test_workflow_gaps.py::test_fill_applies_committed_choice`. |
| R037.4 Refuse instead of inventing a tool. | unit, fake engine | `tests/unit/test_workflow_gaps.py::test_no_candidate_refuses_with_slot`. |
| R037.5 Wire into workflow execution. | unit, fake engine | `tests/unit/test_workflow_gaps.py::test_execute_workflow_fills_gaps_first`. |
| R037.6 End-to-end proof. | unit, fake engine | `tests/harness/test_workflow_gap_e2e.py::test_gap_filled_then_refused`. |

Fresh checkout: `python3 scripts/uv_workspace.py doctor`. Install locked extras using the repository helper, then run `python3 scripts/uv_workspace.py run --all-extras pytest tests/unit tests/orchestration -q` and the focused new conformance tests. Tests must provide fake EG/gateway/provider adapters and temporary directories. Ordinary PR checks must need no pre-existing deployment, vendor account or GPU. Live certification is a separately labelled optional environment test with captured receipts.

Quality: run `python3 scripts/safe_precommit_all_files.py` through the repo lease wrapper when the shared runner is available, plus the configured CCCC, jscpd and Dupehound gates. `pyproject.toml` pins Dupehound 0.1.2 (similarity 0.85, 40-token minimum) and jscpd 5.0.16 (50-token/5-line minimum, mild mode). Use configured differential exclusions and no new duplicate findings. CCCC uses the repository's current hook configuration. Do not invent a numeric threshold. Apply KISS: reuse EG decision/capacity, existing graph entry points and one adapter conformance suite. Reject a second optimizer, event store or scheduler. A local missing tool is reported as not run, never pass.
| R040.1 Model. | contract, fake producer | `tests/unit/runtime/test_workspace_proposal.py::test_holds_paths_and_digests` asserts the exact producer outcome set and preserves `verdict_ref`. |
| R040.2 Behavior. | contract, fake producer | `tests/unit/runtime/test_workspace_proposal.py::test_admission_outcome_adapter` covers admitted/apply, rejected/local-drop and abstained/hold. It asserts that unavailable and refused responses preserve the held layer and base tree. |
| R040.3 Wiring. | unit, fake engine | `tests/unit/test_execute_agent_proposal.py::test_run_holds_layer`. |
| R040.4 Refusal. | unit, fake engine | `tests/unit/runtime/test_workspace_proposal.py::test_base_moved_refused`. |
| R040.5 Evidence. | unit, fake engine | `tests/unit/runtime/test_workspace_proposal.py::test_apply_events_emitted`. |
| R040.6 Hot-path proof. | served, fake producer; all three outcomes plus unavailable and refusal | `tests/integration/test_execute_agent_proposal_served.py::test_no_change_before_verdict`. |
| R041.1 Model. | unit, fake engine | `tests/unit/rlm/test_runtime_quota.py::test_stop_reasons_closed`. |
| R041.2 Behavior. | unit, fake engine | `tests/unit/rlm/test_runtime_quota.py::test_term_then_kill_after_grace`. |
| R041.3 Wiring. | unit, fake engine | `tests/unit/test_fork_fanout_quota.py::test_quota_reaches_backend`. |
| R041.4 Refusal. | unit, fake engine | `tests/unit/rlm/test_runtime_quota.py::test_over_host_limit_refused`. |
| R041.5 Evidence. | unit, fake engine | `tests/unit/rlm/test_runtime_quota.py::test_stop_evidence_recorded`. |
| R041.6 Hot-path proof. | unit, fake engine | `tests/integration/test_fork_fanout_served.py::test_signal_ignored_is_killed`. |
| R042.1 Model. | unit, fake engine | `tests/unit/capabilities/test_disclosure_set.py::test_only_received_ids_added`. |
| R042.2 Behavior. | unit, fake engine | `tests/unit/capabilities/test_disclosure_set.py::test_loads_given_tools`. |
| R042.3 Wiring. | unit, fake engine | `tests/unit/test_agent_factory_disclosure.py::test_given_tools_loaded`. |
| R042.4 Refusal. | unit, fake engine | `tests/unit/capabilities/test_disclosure_set.py::test_undisclosed_call_refused`. |
| R042.5 Evidence. | unit, fake engine | `tests/unit/capabilities/test_disclosure_set.py::test_disclosure_counts_traced`. |
| R042.6 Hot-path proof. | unit, fake engine | `tests/integration/test_execute_agent_disclosure_served.py::test_undisclosed_tool_refused`. |
| R043.1 Model. | unit, fake engine | `tests/unit/graph/test_subtask_set.py::test_received_set_holds_flag`. |
| R043.2 Behavior. | unit, fake engine | `tests/unit/graph/test_subtask_set.py::test_no_cross_agent_messages`. |
| R043.3 Wiring. | unit, fake engine | `tests/unit/test_fork_fanout_aggregate.py::test_results_returned`. |
| R043.4 Refusal. | unit, fake engine | `tests/unit/graph/test_subtask_set.py::test_unflagged_set_refused`. |
| R043.5 Evidence. | unit, fake engine | `tests/unit/graph/test_subtask_set.py::test_result_provenance`. |
| R043.6 Hot-path proof. | unit, fake engine | `tests/integration/test_fork_fanout_served.py::test_parallel_subtasks_returned`. |

## Producer contract coverage

AU-CONTROL-R040 and R041 consume EG-DECISION-ENGINE-R133; admission belongs to GRAPHOS-A2A-R006. AU-CONTROL-R042 and R043 consume EG-DECISION-ENGINE-R134. R042 consumes EG-DECISION-ENGINE-R134.2 without reranking. Registry, schema projection and disclosure contracts belong to GRAPHOS-FLEET-R011, GRAPHOS-FLEET-R013 and GRAPHOS-FLEET-R029.

Harness tests assert receipt and application of these contracts. They do not prove producer decisions, storage or admission.


### Planned disclosure handoff acceptance

Extend `tests/integration/test_execute_agent_disclosure_served.py::test_undisclosed_tool_refused` through the served `execute_agent` operation and real factory. Script GraphOS `find` with two authorized tool schemas and a registry digest. Script EG-R134.2 to select one id, even when local relevance prefers the other. Assert only the selected schema enters the run and its call reaches GraphOS `act` once.

Call the catalogued but undisclosed id; assert `tool_not_disclosed` and zero downstream calls. Exercise denied discovery, missing schema, inconsistent registry generation and engine refusal; assert no tool effect. Inspect loaded/refused counts on the run trace. These planned consumer tests do not prove producer ranking, authorization or registry storage.
