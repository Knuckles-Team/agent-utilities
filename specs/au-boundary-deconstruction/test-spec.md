# Test contract

| Level | Positive proof | Negative proof |
|---|---|---|
| Unit | Generated EG request/result is converted once at AU API boundary. Model adapter receives immutable scoped context | Unknown contract digest, missing graph/tenant, malformed result and unsupported enum fail before execution |
| Wiring | Real arbora route invokes `agent_utilities.api` and EG/SDK owner method once; all old operations are mapped or approved drops | A forbidden AU internal import, duplicate console script or orphaned route fails the boundary check |
| Contract | Pinned generated client and served method agree on schema digest, error code, retryability and receipt | An older incompatible generation and changed idempotency payload are rejected with no side effect |
| Migration | Valid tenant-bound usage/memory record is replayed once and remains visible after process restart | Spoofed tenant, missing source attestation, corrupt row and cross-tenant read remain quarantined |
| Live path | Public endpoint returns scoped result from the new owner and leaves a durable trace/receipt | Denied principal, expired token, timeout and owner outage do not invoke fallback authority |

Run focused tests for each cut, then `python3 scripts/uv_workspace.py run --all-extras pytest -q`, `python3 scripts/check_current_only_contract.py`, `python3 scripts/check_tracked_privacy.py`, `python3 scripts/check_version_consistency.py`, and `agent-utilities lane lease --resource precommit-all-files --operation gate -- python3 scripts/safe_precommit_all_files.py`. Contract tests that need EG use a disposable public container/fixture with a declared image and pinned contract version. Pure static gates run on a fresh checkout without credentials. Capture command, exit code, exact commit, date and artifact URL in `evidence.md`. Tests written or passed on an unmerged branch are not landing evidence.

Quality acceptance includes zero newly duplicated code under jscpd and dupehound, no unexplained CCCC complexity regression, KISS review of every added layer, type/lint clean, and no bypassed normal hook. Scanner findings must be compared against the exact base and final head rather than waived because the repository already has debt.

## Relocation inventory and owner evidence

The R036.1, R037.1, R040.1, R044.1, R045.1 and R047.1 tests check the
current recorded path inventory and detect additions or removals. Their
respective `.2` rows retain the parent requirements' real owner, recovery,
parity and caller-census evidence. An absent AU path is not a replacement
receipt.

## R028 importer and consumer evidence

The four R028 importer-set tests fail when a new current-harness importer is
added and refuse unresolved dynamic imports. They do not establish the
replacement capabilities. R028.2.2, R028.6.2, R028.7.2 and R028.8.2 each
retain their parent consumer or EG CI-gate proof as separate acceptance.

## F-J and T-Z migration acceptance

`AU-BOUNDARY-R006.2` and `AU-BOUNDARY-R009.2` require a complete,
revision-pinned fleet inventory, a resolved census for each package, and real
SDK/graph read and write receipts for each migrated package. R009.2 also
requires the retargeted generated templates to satisfy the same checks. A
fixture census or an indeterminate dynamic import is not a migration receipt.

## AU-BOUNDARY-R049 governed write-back

| Proof | Test |
|---|---|
| A dry run routes through the SDK, returns change-set evidence and writes no durable ledger | `test_dry_run_routes_through_sdk_without_a_durable_ledger` |
| A live write records the change set, the audit reservation and one `applied` receipt | `test_live_write_records_change_set_audit_and_applied_receipt` |
| A live write without the enable flag is refused before the SDK and the sink | `test_live_write_stays_refused_without_the_enable_flag` |
| A sink failure records `outcome_uncertain` and returns the original error | `test_sink_failure_is_recorded_as_uncertain_and_reported` |
| The SDK refuses a denied authorization before the sink runs | `test_sdk_refuses_an_unauthorized_live_call_before_the_sink` |
| The SDK refuses a stale source version before the sink runs | `test_sdk_refuses_a_stale_source_version_before_the_sink` |
| An approved replay uses the `proposal_approval` mode | `test_approved_replay_uses_proposal_approval_mode` |
| The change set is canonical and excludes private keys and clients | `test_change_set_is_canonical_and_drops_private_ops` |

All tests live in `tests/unit/knowledge_graph/enrichment/test_writeback_governed.py`. The existing write-back, approval, preflight, bitemporal and ETL suites run unchanged against the new path.

## AU-BOUNDARY-R043 agent run loop

| Row | Proof | Test |
|---|---|---|
| `AU-BOUNDARY-R043.1` | The host plan drives the run. Steps share one history. Budgets and stop rules stop the run. | `tests/test_run_loop.py` |
| `AU-BOUNDARY-R043.2` | The fallback plan reads only the request and configuration. Missing providers fail closed. | `tests/test_run_loop.py` |
| `AU-BOUNDARY-R043.3` | The recursive loop caps slices and depth. It classifies each failure. | `tests/test_rlm.py` |

The tests use fakes for the host ports and pydantic-ai function models. They run under `pytest -n auto` in random order.

## AU-BOUNDARY-R043.4 scaffold migration acceptance

R043.4.1–R043.4.5 retain the historical R014.1–R014.5 acceptance. Use deterministic
model fixtures through the existing run loop: replace ordered items twice,
refuse malformed input atomically, gate the tool on the received decision,
inspect every outgoing request, and assert no scaffold content in stored history
or tool records. Check isolation across two runs and the absence of planner
imports or selection fields. The bound test files below provide the scaffold implementation evidence.

Snapshot follow-up must prove revision zero, repeated-content increments, read
stability, atomic item/revision capture, UTC expiry and lease capping, caller
isolation by tenant and principal, and identical unknown/foreign unavailability.
Test disabled, ended and restarted runs, absent ports, malformed or regressing
snapshots, expired leases, denied refresh and run switches. A served test must
reach the same state used by injection, without a model call or history rebuild.
The SDK, EG and Arbora boundaries remain separate acceptance obligations.

R043.4.1 is exercised by `tests/test_harness_plan.py`: full ordered replacement,
empty reset, stable old views, immutable items, two-run isolation, strict field
validation and invalid-later-item atomicity. The loop/tool/injection evidence for
R043.4.2–R043.4.5 is recorded below.

R043.4.2–R043.4.5 are exercised by `tests/test_plan_runtime.py` through
`graph.executor.run`: received on/off gating, two updates and a malformed update,
per-request injection across execution steps, separate-run isolation, retained
history inspection, budget-exit cleanup and the no-selection tool/import surface.
This evidence covers the scaffold loop, not a snapshot reader or served inspector.

## AU-BOUNDARY-R043.5.1 atomic revision acceptance

Read revision zero with empty items. Accept identical replacements and empty
resets with one increment each. Reject invalid items without an increment.
Preserve old immutable views. Exercise competing writers and interleaved reads
with bounded barriers, no sleeps or network, and prove every observed revision
corresponds to its complete item tuple. This proves no caller or lease authority.

## AU-BOUNDARY-R043.5.2 publication mechanism acceptance

`tests/test_plan_runtime.py` drives the real model/tool loop: revision zero
precedes the first request, two accepted updates publish in order, invalid input
publishes nothing, concurrent runs remain isolated, and host binding/publication/
close failures fail the run without exposing their payloads.
`tests/test_host_ports.py` covers absent/off bindings, cancellation during initial
publication, invalidation before close, retained publisher refusal, preservation
of original body errors during cleanup, and no rollback after uncertain writes.
These tests use a host double and establish no installed EG acceptance, verified
lease admission, deployed decision activation or served inspector availability.


AU-BOUNDARY-R043.5.2: Per-run declined publication leaves local state usable, closes exactly once on success or body failure, and never changes another run’s binding.

## AU-BOUNDARY-R043.6 and R043.7 dependency execution

Typed tool steps (AU-BOUNDARY-R043.6.1 to R043.6.7) use the same `tests/test_dependency.py` harness.
A fake `ToolCaller` replaces any network or fleet call.
The function model records requests. A test can therefore prove that no model ran a tool step.

- `AU-BOUNDARY-R043.6.1`: `tests/test_dependency.py::test_mixed_plan_validates` accepts a mixed DAG. `test_tool_step_without_caller_refuses_before_any_task` refuses with the typed reason and starts no task. A source scan finds no MCP or HTTP client import in the dependency module.
- `AU-BOUNDARY-R043.6.2`: `tests/test_dependency.py::test_bindings_resolve_from_prerequisite_outputs` resolves nested keys and array indexes from a tool step and a JSON prompt step. A repeat run gives byte-identical arguments. The function model records zero requests for the tool step.
- `AU-BOUNDARY-R043.6.3`: `tests/test_dependency.py::test_unresolved_binding_is_a_typed_failure` covers each cause. `test_failed_tool_step_does_not_start_dependents_and_makes_no_model_request` proves dependents stay unstarted. It also proves the function model sees no request.
- `AU-BOUNDARY-R043.6.4`: `tests/test_dependency.py::test_tool_step_records_resolved_arguments_and_result` checks the fields and order. `test_oversized_result_is_replaced_by_digest` checks the bound.
- `AU-BOUNDARY-R043.6.5`: `tests/test_dependency.py::test_mixed_dag_refills_slots_in_order` runs a prompt step, a bound tool step and a prompt step that reads the result at capacity two. It checks start order, the cap and the output task.
- `AU-BOUNDARY-R043.6.6`: `tests/test_dependency.py::test_execute_graph_routes_tool_steps_through_the_bound_caller` enters the public run API and observes the caller invoked. `test_cancel_cancels_inflight_tool_call` observes cancellation.
- `AU-BOUNDARY-R043.6.7`: `tests/test_dependency.py::test_tool_step_end_to_end_through_run_api`.


`tests/test_dependency.py` drives the public run API with deterministic
FunctionModel responses and explicit event gates, with no sleeps or network.
Cover capacity-two rolling refill, joins, deterministic output, isolated task
history and malformed DAGs. Cover budgets, stop markers, failures, timeout, parent
cancellation, iterator close and ordered unique lifecycle events. Existing
`tests/test_run_loop.py`, `tests/test_rlm.py` and `tests/test_plan_runtime.py`
remain compatibility gates. This proves in-run behavior, not served admission
or durable replay. Those proofs belong to Arbora and EG.

Test paused readers against both rolling dispatch and the driver deadline.
Test scaffold binding and initial publication against the same run timeout.
Assert completed asynchronous model cleanup under repeated cancellation, including nested runs.
Refuse oversized tool batches before fan-out, including the finite parallel fallback.
Keep root-only recursion and first-event iterator close compatible.

### Initial host accounting under AU-BOUNDARY-R043.7

Public run-loop and dependency tests provide `RunUsage` and measured elapsed
time, with original budgets intact. They prove exact one-time aggregate
accounting, pre-dispatch refusal of exhausted bounds, strict invalid-input
refusal and unchanged zero defaults. Exact tool limits permit model-only
answers, including at zero. Additional tool calls must have no side effects.
Prior tool usage above the limit must stop before binding or child dispatch.
Cover setup failure, child failure,
parent cancellation, serial multi-step limits, plan reuse and a provider
that reports no token counts. Model proposal text and request metadata must
not change the host ledger. No test uses a sleep or live provider.

Trusted provider failures after measured planning must preserve their original
error and include the reported counters exactly once. Terminal event usage
must equal result usage, even when the failed planning consumed a limit.
Neither a failed carrier nor its elapsed time may admit executable work or
replace a budget. Test synchronous and asynchronous providers, invalid
ledgers, reuse, cancellation and iterator close. A provider failure without
a ledger must not invent consumption. Existing success and zero-default
cases must keep their behavior.

Public executor tests must retain a failed aggregate receipt in `RunFailed`
with the same `RuntimeError` message and catch behavior. Prove equality with
the terminal event for planning, setup, serial and dependency failures.
Validate scalar fields and the entire public ledger. Source and caller
mutation must not alter the retained receipt; empty usage stays empty.
Exercise plain-dict serialization and exception copy and pickle round trips.
Forged provider and model receipts must not provide accounting. Successful
and stopped returns and original cancellation must remain unchanged.
The runtime facade exports the identical run contract objects.
Importing it must not load the graph runtime before a caller resolves those aliases.

### Run resume, history and live events under AU-BOUNDARY-R043.8 to R043.10

AU-BOUNDARY-R043.8 tests restart a function-model run from a completed
checkpoint and count model calls. An in-flight checkpoint, a changed plan and
a changed admission each refuse before any effect. AU-BOUNDARY-R043.9 tests
capture the earlier messages in the next run's first request and refuse
malformed history. AU-BOUNDARY-R043.10 tests record publication order, refuse
a tool effect after a failed call publication, and stop after a failed result
publication. Live qualification needs the EG checkpoint producer and the
Arbora admission-bound adapter.

## R043.11 client tool forwarding

`tests/test_client_capabilities.py` drives the real run loop with a scripted
model and a bound test transport. Assert exact method and parameters, one call,
closed permission refusals, byte and wait limits, absent transport refusal and
privacy-safe trace fields. This test does not grant permission or prove the
Arbora gateway and native approval producers.

## AU-BOUNDARY-R043.12: measured request emission

Bind all six tests in `tests/test_graph_metrics.py` to this producer row. Repeated FunctionModel requests prove attribution, response counts, client-clock timing and equal terminal-event metrics; negative cases prove no fabricated failed-request usage, invalid-count omission, zero requests without a model, no zero-duration TPS and bounded partial coverage. These deterministic tests do not measure a live provider or prove streaming coverage.

## Remaining checkpoint-backed history acceptance

R043.9.2 remains open. A real completed checkpoint must populate the next
function-model request without importing prior budgets, plan or outputs.
A current-run checkpoint must take precedence over earlier conversation
history. Malformed or incomplete checkpoints must make no harness model or
tool call. Installed EG and Arbora acceptance must restart between turns,
continue the same authorized context and refuse a foreign context. The
text-only R043.9.1 fixtures do not satisfy these checks.


## Single-read checkpoint selection (R043.9.2.1)

Use a real FunctionModel and capture host calls. A single immutable selection
contains the requested run identity and both presence fields. Change the host
registration after selection and prove the harness does not read again or mix
providers. This simulates the consumer race; it does not prove native atomicity.

Check current-run precedence with empty, malformed and in-flight content.
Refuse before the decision provider or model binding. Never parse or fall back
to previous history. An empty bound selection starts a fresh turn without a
legacy read. Refused, partial, wrong-run and malformed selections make no model
call. Provider exceptions expose no private details. Completed text uses the
existing strict codec and one-request fresh usage limit.

R043.9.2.2 retains genuine restoration and installed producer acceptance.
Selection tests do not satisfy durable write ordering or no-replayed-tool proof.

## Received text-only execution

`AU-BOUNDARY-R043.13.1`: `tests/test_text_only.py` exercises the real chat-completion model and run loop with simulated HTTP. It rejects tool-bearing defaults and requests, unsupported providers, SDK retries and nontext output. It checks host-hook isolation, cancellation and measured budget use.

`AU-BOUNDARY-R043.13.2`: Installed native ownership, durable outcomes, restart, real provider behavior and configured admission limits remain required. Source tests do not prove them.

## Child explorer acceptance (pending)

- `AU-BOUNDARY-R043.14`: run two steps with a join and a separately identified child/grandchild; assert distinct dependency/containment, stable replay IDs, no inferred identity and terminal drain ordering.
- `AU-BOUNDARY-R043.15`: use controlled model gates, not sleeps, for duplicate input and completion/cancellation races; assert one boundary application, exact target and unchanged shared budgets. Failed receipt publication must not imply safe unkeyed replay.
- `AU-BOUNDARY-R043.16`: capture the first request of a fresh child continuation; assert child-only permitted history, new input, fresh turn counters and enclosing ceilings. Mismatched/incomplete/unsupported history refuses before any effect; current checkpoint takes precedence.
- `AU-BOUNDARY-R043.17`: assert bounded public record kind/provenance/visibility, no private reasoning, durable publisher order and before-effect/after-effect refusal behavior. Fixtures do not prove a released host's transcript support.

### R043.11 host request and streaming seam

`tests/test_hosted_run.py` drives the real sequential run loop with Pydantic AI
FunctionModel responses and streams. It checks unchanged request identity at
planning/history selection, malformed-history refusal, client-only tool binding,
received limits and synchronous iterator teardown. Native text deltas and tool
transitions reach an awaited observer before completion. Observer failure,
cancellation and backpressure retain the existing failed/stopped outcomes without
retry. Thinking reaches only the trusted observer and never the public run sink.
Dependency streaming refuses before model calls. These are scripted-model harness
tests, not stdio, authenticated remote ACP or live-provider acceptance.

`tests/test_run_loop.py` additionally drives both terminal `run` and the typed
executor with native observers. FunctionModel streams prove unchanged host
request/session mapping, native tools, received limits, private-thinking
separation, sanitized observer failures and cancellation without retries.
These producer tests do not qualify Arbora admission or durable continuation.
