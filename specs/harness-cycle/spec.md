# AU-HARNESS-CYCLE — Harness evolution cycle

Owner: agent-utilities. Cross-repo IDs: EG-ORCH-VOCAB-R801, EG-ORCH-VOCAB-R802, EG-DECISION-ENGINE-R803, EG-DECISION-ENGINE-R136.3, ARBORA-OPS-R068.

## Purpose and user stories

The harness improves itself while a task runs. For one task, the
harness proposes candidate component versions. It probes each under
identical conditions. It composes the adopted harness from the
measured contrast. It persists the result as graph facts. It reflects
the lesson for the next task. A developer runs a task through arbora.
The developer then reads the adopted harness version, the contrast and
the lesson from the graph. The operator sees which component changed,
why, and the evidence run behind it.

## Requirements and acceptance

- R001: The harness runs one Propose-Probe-Compose cycle per task tick under a candidate budget k. It adopts a candidate only on a measured improvement and abstains otherwise. Acceptance: the hot-path proof of R001.9.

## Why a new spec

Commit c45f277c0 re-homed the `harness-evolution` spec (AU-HARNESS,
snapshot 5eb8d85a3) out of agent-utilities. The substrate moved to
epistemic-graph: the versioned components and the improvement library
(EG-ORCH-VOCAB-R801/R802) and the contrastive composition decision
(EG-DECISION-ENGINE-R803). It moved to arbora: the served read and
adoption operations (ARBORA-OPS-R068). This repository keeps the
in-run loop itself. It covers the propose, probe, compose, persist and
reflect steps between two served operations. `AU-BOUNDARY` owns the
boundary decision. No remaining spec owns the cycle. The
`agent_utilities/harness/` package today holds the constraint,
contract, edit, plan and quality-gate engines only.

## Scope and interfaces

Owned: the cycle loop and its models
(`agent_utilities/harness/harness_cycle.py`). The probe entry runs
through `execute_graph` (`agent_utilities/graph/runner.py:80`). The
harness switch runs through `apply_edits`
(`agent_utilities/harness/edit_engine.py:457`). arbora drives the host
port. The candidate budget k comes from the stage allocation of
EG-DECISION-ENGINE-R136.3. Non-goals: versioned component storage and
lesson storage (epistemic-graph). The served operations (arbora). Any
local ranking of candidates. The measured contrast is the authority.

## Traceability

| Requirement | Design section | Test ID | Evidence |
|---|---|---|---|
| R001 | plan.md#harness-evolution-cycle | T-001 |  |

## Open questions

None.
