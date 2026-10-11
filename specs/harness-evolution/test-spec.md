# Test contract: harness evolution

| Scenario | Fixture | Expected result |
|---|---|---|
| All controls omitted. Capture-only enabled | unit + provider fake | capture/train/promote default false. Capture alone emits no trainer job or pointer change |
| Closed/unversioned provider, no chosen-token probabilities | capability negatives | typed refusal; no fabricated training row |
| Terminal valid capture and replay. | fixed token ids, masks, frozen `log_q`, verifier, held blobs. | exact digest match, masked non-policy tokens, immutable sampler identity. |
| Incomplete/mismatched arrays, missing reward, dead blob holder | capture negatives | trace may persist, trainer eligibility false |
| External trainer absent/failed/cancelled | injected dispatch and A2A fake | no serving pointer movement; recoverable receipt, no duplicate job |
| Held-out regression or unsupported replay mass | promotion negative | promotion denied; old immutable version remains served |
| Duplicate evidence signal | served EG in-memory contract | one Gap and one WorkItem, stable idempotency key |
| Offer with open dependency, cooldown, over-budget or other tenant | decision contract | candidate excluded with reason; no AU fallback sort |
| Decide unavailable, commit replay mismatch, claim fence lost | integration fault | no execution and no second scheduler |
| Proposal missing approval, stale base, dirty tree, path escape, gate failure, uncertain remote result. | graph-os/RM contract fakes. | no unauthorized commit; pending proposal with typed reason. |
| L4 conformance | native adapter with injected runner. Claude Code adapter with fake executable | both satisfy `HarnessPort`. Outcome names the port, keeps the run identifier and uses a known status |
| Native envelope | injected `run_agent` returning the run-summary envelope | summary requested; JSON output parsed; degraded failure text kept; outcome marked recorded |
| Native timeout | runner slower than `timeout_s` | typed `timeout` outcome |
| Claude Code launch | fake executable records arguments, directory, input and environment | pinned flags first. MCP configuration last. Prompt on input only. Directory is the worktree. Parent secrets absent |
| Claude Code result | fake success, error, unparseable and sleeping modes | typed usage, cost, model and transcript reference. `failed` with exit code; `timeout` with the child killed |
| Claude Code refusal | no worktree, no configuration file, no executable | `refused` outcome; child never starts |
| Diff stat | temporary Git repository with one changed file | exact files, insertions and deletions; no repository write |
| Selection | registry with fake ports | `native` by default; field selects the port; unknown name raises |
| L5 recording | patched RunTrace writer and usage recorder | non-native outcome written once with `harness:<name>` mode; native outcome not written again |
| Parallel engine dispatch | `AgentSpec(harness="claude-code")` with a fake port | node runs through the port and returns its result |
| Successful approved proposal | public contract integration | exact commit/gate/receipt digests; Gap resolves only after receipt re-ingest |
| Agent with 10 new outcomes, 3 failures (AU-HARNESS-R011) | graph double + fake optimizer | One `PromptVersion` proposal. Parent hash of the baseline. 10 `was_derived_from` edges. Prompt file unchanged. Cursor advanced. |
| Pending proposal, too few outcomes, no failures, other agent's traces | graph double | no optimizer call and no new version |
| Optimizer returns no candidate; graph write refused | graph double | status `error`; cursor and versions unchanged |
| Default optimizer | fake engine serving `optimize_program` | native `bootstrap_few_shot` compiled state recorded on the proposal |
| Scheduled sweep | `run_optimization_sweep(targets=["prompt_evolution"])` | dispatches to the run-outcome sweep; reported as optimized, propose-only |
| R012.1 `HarnessConfig` model. | scripted model, fake engine | `tests/harness/test_harness_config.py::test_config_frozen_and_thresholds_ordered`. |
| R012.2 Carry the config on `HarnessRequest`. | scripted model, fake engine | `tests/harness/test_harness_native.py::test_request_without_config_is_refused`. |
| R012.3 `run_agent` accepts and applies the config. | scripted model, fake engine | `tests/harness/test_harness_runtime.py::test_run_agent_builds_runtime_from_config`. |
| R012.4 Worker reads the envelope field. | scripted model, fake engine | `tests/unit/test_agent_dispatch_worker.py::test_worker_passes_envelope_harness_config`. |
| R012.5 End-to-end proof of the envelope path. | scripted model, fake engine | `tests/harness/test_runloop_envelope_e2e.py::test_envelope_config_reaches_run_loop`. |
| R013.1 Reuse decision: one implementation. | scripted model, fake engine | `tests/unit/test_composition.py::test_default_capabilities_hold_one_context_manager`. |
| R013.2 Token counter against the budget. | scripted model, fake engine | `tests/harness/test_context_manager.py::test_peak_tokens_tracked`. |
| R013.3 Region split. | scripted model, fake engine | `tests/harness/test_context_manager.py::test_regions_keep_tool_pairs_together`. |
| R013.4 Elision at the soft threshold. | scripted model, fake engine | `tests/harness/test_context_manager.py::test_elision_stubs_middle_observations_only`. |
| R013.5 Summarization at the hard threshold. | scripted model, fake engine | `tests/harness/test_context_manager.py::test_summarization_folds_oldest_middle`. |
| R013.6 Policy gating. | scripted model, fake engine | `tests/harness/test_context_manager.py::test_policy_gates_each_mechanism`. |
| R013.7 Typed `context_overflow` termination. | scripted model, fake engine | `tests/harness/test_context_manager.py::test_overflow_is_typed_termination`. |
| R013.8 Mechanism counters. | scripted model, fake engine | `tests/harness/test_context_manager.py::test_counters_increment`. |
| R013.9 End-to-end proof. | scripted model, fake engine | `tests/harness/test_context_manager_e2e.py::test_live_run_elides_summarizes_and_overflows`. |
| R014.1 `PlanState` model. | scripted model, fake engine | `tests/harness/test_harness_plan.py::test_plan_state_replaces_list`. |
| R014.2 `update_plan` tool. | scripted model, fake engine | `tests/harness/test_harness_plan.py::test_update_plan_tool_gated_by_config`. |
| R014.3 Per-turn injection outside history. | scripted model, fake engine | `tests/harness/test_harness_plan.py::test_plan_injected_not_stored`. |
| R014.4 Planner boundary. | scripted model, fake engine | `tests/harness/test_harness_plan.py::test_scaffold_imports_no_planner`. |
| R014.5 End-to-end proof. | scripted model, fake engine | `tests/harness/test_harness_plan_e2e.py::test_plan_in_every_request_never_in_history`. |
| R015.1 Registration filter. | scripted model, fake engine | `tests/harness/test_action_space.py::test_registration_filters_to_subset`. |
| R015.2 `shell_only` mode. | scripted model, fake engine | `tests/harness/test_action_space.py::test_shell_only_registers_one_tool`. |
| R015.3 Execution-time refusal. | scripted model, fake engine | `tests/harness/test_action_space.py::test_out_of_set_call_returns_refusal_observation`. |
| R015.4 End-to-end proof. | scripted model, fake engine | `tests/harness/test_action_space_e2e.py::test_live_registration_and_refusal`. |
| R016.1 Stuck reminder. | scripted model, fake engine | `tests/harness/test_run_guards.py::test_reminder_after_identical_calls`. |
| R016.2 Stuck termination. | scripted model, fake engine | `tests/harness/test_run_guards.py::test_stuck_termination_after_failing_repeats`. |
| R016.3 Post-edit diagnostics. | scripted model, fake engine | `tests/harness/test_run_guards.py::test_edit_result_carries_diagnostics`. |
| R016.4 Read-before-write. | scripted model, fake engine | `tests/harness/test_run_guards.py::test_write_requires_fresh_read_hash`. |
| R016.5 Tool errors as observations. | scripted model, fake engine | `tests/harness/test_run_guards.py::test_tool_error_becomes_observation`. |
| R016.6 Result truncation. | scripted model, fake engine | `tests/harness/test_run_guards.py::test_long_result_truncated_with_marker`. |
| R016.7 Parallel read-only tools. | scripted model, fake engine | `tests/harness/test_run_guards.py::test_read_only_tools_run_concurrently`. |
| R016.8 End-to-end proof. | scripted model, fake engine | `tests/harness/test_run_guards_e2e.py::test_all_guards_fire_in_live_loop`. |
| R017.1 `TerminationReason` enum. | scripted model, fake engine | `tests/harness/test_harness_runtime.py::test_single_termination_reason`. |
| R017.2 Map existing exits. | scripted model, fake engine | `tests/harness/test_harness_runtime.py::test_every_exit_sets_reason`. |
| R017.3 Counters. | scripted model, fake engine | `tests/harness/test_harness_runtime.py::test_counters_cover_brief`. |
| R017.4 Write through `_record_execution_trace`. | scripted model, fake engine | `tests/harness/test_run_trace_fields.py::test_trace_write_carries_reason_and_counters`. |
| R017.5 Non-native path. | scripted model, fake engine | `tests/harness/test_run_trace_fields.py::test_non_native_outcome_has_reason`. |
| R017.6 End-to-end proof. | scripted model, fake engine | `tests/harness/test_run_trace_e2e.py::test_each_reason_persisted`. |
| R018.1 Arm and task-set types. | scripted model, fake engine | `tests/harness/test_ablation.py::test_arms_differ_in_one_component`. |
| R018.2 Paired execution. | scripted model, fake engine | `tests/harness/test_ablation.py::test_paired_outcomes_recorded`. |
| R018.3 Exact McNemar test. | scripted model, fake engine | `tests/harness/test_ablation.py::test_exact_mcnemar_known_values`. |
| R018.4 Benjamini-Hochberg correction. | scripted model, fake engine | `tests/harness/test_ablation.py::test_bh_adjustment`. |
| R018.5 Report. | scripted model, fake engine | `tests/harness/test_ablation.py::test_report_fields_and_insufficient`. |
| R018.6 Governed proposal. | scripted model, fake engine | `tests/harness/test_ablation.py::test_facts_submitted_as_proposals_only`. |
| R018.7 End-to-end proof. | scripted model, fake engine | `tests/harness/test_ablation_e2e.py::test_two_arms_to_proposal`. |

Fresh checkout: `python3 scripts/uv_workspace.py doctor`, then focused `python3 scripts/uv_workspace.py run --all-extras pytest tests/harness tests/unit -q`. New tests must inject fake EG, graph-os, provider, trainer and repository-manager boundaries. Normal PR CI needs no live graph, vendor credentials, model server, GPU or private network. Optional live certification records exact receipts and is not a prerequisite for contribution PRs unless the changed contract requires it.

Quality: run the repository's configured CCCC, jscpd and Dupehound gates and `python3 scripts/safe_precommit_all_files.py` through the lease wrapper when available. Configured Dupehound is 0.1.2 with similarity 0.85 and 40-token minimum; jscpd is 5.0.16 with 50-token/5-line minimum and mild mode. Honor actual configured differential filters; require zero **new** duplicate findings. CCCC uses the configured hook threshold, not an invented score. Apply KISS by extending `SubstrateTrainer`, native WorkItem, EG Decide and existing Loop controller, and deleting replacement authority. Missing optional tooling is recorded as not run, never as a pass.
| Studied sources become compared, admitted proposals. | tests/harness/test_study_pipeline.py fake sources | AU must not write code, specs or prompts from a study directly. |
| Locate and wire the study attach point. | test_study_pipeline.py fake sources | The first child locates the entry beside `agent_utilities/harness/assimilation_benchmark.py` and wires `study_pipeline.run_study` into the e |
| Typed `StudySource` and `FeatureRecord`. | test_study_models.py fake sources | Records without an evidence span are refused. |
| Study intake deduplicates by content digest. | test_study_intake.py fake sources | It must not copy the source body into the graph; it stores the digest and locator. |
| Batch study scales to ten times the source count. | test_study_batch.py fake sources | One failed source must not stop the batch and yields a typed failure row. |
| Capability comparison matrix against the platform. | test_study_compare.py fake sources | AU must not keep a private capability inventory. |
| Matrix report lists every feature with its verdict. | test_study_compare.py fake sources | A study with no extractable feature reports `no_features`, not an empty success. |
| Gap features become proposals only. | test_study_propose.py fake sources | AU edits no repository, spec or prompt. AU promotes no proposal. |
| One open proposal per feature blocks duplicates. | test_study_propose.py fake sources | `study_propose.propose` must skip a gap whose feature digest has an open or rejected proposal. It must return the existing proposal ID. |
| Admission requires a measured lift. | test_study_admit.py fake sources | A refusal names the missing measurement. |
| Admission is a governed decision, not an AU choice. | test_study_admit.py fake sources | AU must not rank or accept proposals itself. |
| Research goals drive source discovery. | test_study_discover.py fake sources | It must stop at a per-run source cap of 50. |
| Background study yields to interactive work. | test_study_batch.py fake sources | `run_batch` pauses under interactive load. It resumes from its cursor without repeats |
| Summary quality, lossless summary nodes and oversize output variables | scripted run, fake judge | A poor summary regenerates once and a second failure keeps the raw block. Oversize output never enters a request raw. |
| Parallel workers share a typed workspace | four fake workers | A schema violation is refused. A conflict is recorded. The breaker stops new workers after three failures. |
| Evaluation gates separate safety from accuracy | scripted round | Accuracy gain with a safety regression is held. Divergent judge and outcome trends are refused as evidence. |
| Run experience becomes reusable skills | three scripted runs | Three successes scaffold one skill. A candidate with too few rollouts is refused. |
| Replay of a received run program | scripted engine program, then a changed state | The replay uses no model call. A changed state returns control to the agent. |
| A span for every agent, tool and orchestration step | scripted served run, in-memory exporter | One connected trace holds agent, tool, step and diagram spans. |
| Harness edits are revertible and paired with predictions | two scripted edits, scripted verdicts | A falsified verdict reverts only its recorded files. |
| Parallel-map primitives and recursive decomposition | prompt of 3 times the window, scripted limits | The answer fits the received limits. A model-written loop is refused. |
