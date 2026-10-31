# MERISTEM-HARNESS-CYCLE — Harness evolution cycle

Owner: meristem. Cross-repo IDs: MEDULLA-ORCH-VOCAB-R801, MEDULLA-ORCH-VOCAB-R802, MEDULLA-DECISION-ENGINE-R803, MEDULLA-DECISION-ENGINE-R136.3, ARBORA-OPS-R068.

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

Commit c45f277c0 re-homed the `harness-evolution` spec (MERISTEM-HARNESS,
snapshot 5eb8d85a3) out of agent-utilities. The substrate moved to
medulla: the versioned components and the improvement library
(MEDULLA-ORCH-VOCAB-R801/R802) and the contrastive composition decision
(MEDULLA-DECISION-ENGINE-R803). It moved to arbora: the served read and
adoption operations (ARBORA-OPS-R068). This repository keeps the
in-run loop itself. It covers the propose, probe, compose, persist and
reflect steps between two served operations. `MERISTEM-BOUNDARY` owns the
boundary decision. No remaining spec owns the cycle. The
`meristem/harness/` package today holds the constraint,
contract, edit, plan and quality-gate engines only.

## Scope and interfaces

Owned: the cycle loop and its models
(`meristem/harness/harness_cycle.py`). Also owned: the consumer rows
R003 onward, where the in-run harness applies received medulla decisions. The probe entry runs
through `execute_graph` (`meristem/graph/runner.py:80`). The
harness switch runs through `apply_edits`
(`meristem/harness/edit_engine.py:457`). arbora drives the host
port. The candidate budget k comes from the stage allocation of
MEDULLA-DECISION-ENGINE-R136.3. Non-goals: versioned component storage and
lesson storage (medulla). The served operations (arbora). Any
local ranking of candidates. The measured contrast is the authority.

## Traceability

| Requirement | Design section | Test ID | Evidence |
|---|---|---|---|
| R001 | plan.md#harness-evolution-cycle | T-001 |  |

## Open questions

None.
