# AU-HARNESS-CYCLE — Design and implementation plan

Governing spec: [spec.md](spec.md).

## Existing system and reuse

Commit c45f277c0 re-homed the AHE pillar out of agent-utilities. The
program optimization, ablation, evaluation-loop and parallel-engine
substrate now belongs to epistemic-graph, and the served operations
belong to arbora. The harness package keeps the constraint engine,
contract validator, edit engine, plan, plan runtime and quality gates.
The run loop enters through `execute_graph` at
`agent_utilities/graph/runner.py:80`. Host-provided ports live in
`agent_utilities/host_ports.py`: `RunDecisionProvider` answers the run
plan, `RunEventSink` receives run events, and `ScaffoldPublisher`
publishes plan revisions. The cycle reuses these ports and adds one
cycle port beside them.

## Architecture

One module owns the loop: `agent_utilities/harness/harness_cycle.py`.
The loop is one tick: propose, probe, compose, persist, reflect. Each
step is one function with one refusal path. The module holds no store
and no ranking. It queries the library (EG-ORCH-VOCAB-R802.3). It
enters the probes through `execute_graph`. It delegates the
composition to the engine (EG-DECISION-ENGINE-R803). It writes through
the generated-client envelopes. arbora drives the tick through the
`HarnessCycleProvider` port. Without a registered provider the port
fails closed, as every other port does.

## Interfaces and data model

`HarnessCandidate` and `CycleOutcome` are frozen dataclasses that carry
references, not content. The probe runs under an identical task, seed,
model and budget, and the conditions digest binds them. The compose
call is one served `Decide` under the `HARNESS_CONTRAST` question kind.
The persist step writes three graph facts: the adopted component
version (EG-ORCH-VOCAB-R801.2), the contrast and the lesson
(EG-ORCH-VOCAB-R802.1). The in-run harness switch reuses `apply_edits`
at `agent_utilities/harness/edit_engine.py:457`.

## Live integration path

arbora registers `HarnessCycleProvider` at startup and drives one tick
per task. The tick reads the candidate budget k from the stage
allocation in the run decision (EG-DECISION-ENGINE-R136.3). The
reflect step reports the outcome through the existing `RunEventSink`,
and arbora writes the trace into epistemic-graph.

## Quality and release gates

Focused tests:
`uv run --frozen python -m pytest tests/harness/test_harness_cycle.py -q`.
Served proof:
`uv run --frozen python -m pytest tests/harness/test_harness_cycle_served.py -q`.
Language-native gates: ruff, mypy, CCCC, jscpd, dupehound and KISS on
the changed code through the repository gates in
`.config/pre-commit.yaml` and `release.yml`.

## Risks, alternatives, and decisions

The cycle can drift from the engine on the contrast semantics. The
decision: the engine owns the composition (EG-DECISION-ENGINE-R803),
and the cycle owns only probe scheduling and persistence. The rejected
alternative kept a local ranking inside the cycle. It fails the
no-duplicate-authority check, because the measured contrast is the
only authority.

## Reachability

`drive_cycle_tick` has no production caller. No host registers a `HarnessCycleProvider` yet.
Superseded evidence: the unit tests of R001.5 and R001.8 run against fake persist and decide ports.
They prove the AU step logic and the closed port. They do not prove host registration or engine envelopes.
`AU-HARNESS-CYCLE-R001.10` owns persistence through the released EG client.
`AU-HARNESS-CYCLE-R001.11` owns the host-registered tick. The host registration row belongs to arbora.
The arbora row waits for the release of EG-ORCH-VOCAB-R801.2, EG-ORCH-VOCAB-R802 and EG-DECISION-ENGINE-R803.

## Probe-condition evidence gap

The R001.3 test currently proves that probe records carry the same digest of
requested values, not that each host-admitted run actually uses those values.
In `test_probe_shares_conditions_and_records_trace`, the requested model is
`model-1`, while the host decision returns a `FunctionModel`; the test does
not compare the admitted budget or model settings with the digest inputs.
`execute_graph` passes the requested model and seed metadata to the host, but
`resolve_plan` accepts the host's returned `RunPlan` without binding it to the
probe's budget reference or verifying a seed. Thus the recorded `VERIFIED`
row is unit evidence for digest propagation only; it is not proof of identical
served probe conditions.

Before enabling a host provider, bind each probe to an owner-admitted plan and
verify the actual model identity, supported seed settings, complete enforced
budget and candidate version before model or tool effects. A digest over
requested inputs alone cannot establish those facts. Add an effect-counting
regression for a substituted model, seed or budget and an installed hosted
journey that reads the real admitted conditions and traces; otherwise abstain.
