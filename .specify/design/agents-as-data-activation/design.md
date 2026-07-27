# Design Document: Agents-as-data activation layer (ADR-6 / W2.3)

> Every feature begins with a design document. This gates creation through
> the Knowledge Graph to enforce the **Extend-Before-Invent** principle.

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| AU-ORCH.dispatch.* | agent dispatch worker / queue dispatch | high | AU-ORCH |
| AU-OS.state.unified-durable-state-externalization | WorkItem CAS + durable state | high | AU-OS |
| AU-ORCH.scheduling.claim-pacing-backpressure | W2.4 QoS priority claim | med | AU-ORCH |

### Extension Analysis

- **Primary Extension Point**: the existing dispatch/WorkItem-CAS machinery + the `eg-statechart` `MachineInstance` store + per-tenant graph nodes.
- **Extension Strategy**: compose (three existing stores + one worker loop) — no new store or transport.
- **New Concept Required?**: Yes — a distinct dispatch pattern (durable dormancy, zero live resources while dormant).

## Problem

A conventionally "running" agent holds a thread/connection/heartbeat even while idle, so N dormant agents cost O(N) live resources. We want a **dormant** agent to be pure durable data (a statechart instance + a graph node) with **zero** live footprint, activated only on demand, at fleet scale.

## Design

**`CONCEPT:AU-ORCH.dispatch.agents-as-data-activation`** — a dormant agent is a durable `eg-statechart` `MachineInstance` (`dormant ⇄ active`) plus a per-tenant graph node — no thread/connection/heartbeat while dormant. `orchestration/agent_activation.py` composes the three existing stores (statecharts, WorkItem CAS, graphs) with one stateless worker loop:

- an activation event (`deliver_activation`) appends to the instance mailbox and submits a WorkItem in the QoS lane derived from the activation **source** (direct⇒Interactive, timer/orch/broker⇒Orch, cdc⇒Ingest);
- a stateless worker (`agent-activation-worker`, one process = one worker) claims via the CAS lease (the liveness signal), drives the instance `dormant → active`, runs the pluggable executor under the ADR-4 delegation chain + W2.4 `priority_scope`, writes `:RunTrace`/`:ToolCall` provenance, heartbeats **only while active** (bounded-time revocation), commits the WorkItem terminally, and releases the instance to `dormant`;
- a dead worker's lease expires and the activation re-queues (bounded retries → dead_letter, the ADR-5 machinery).

Canonical template `AGENT_LIFECYCLE_DEF` mirrors eg's `agent_lifecycle_statechart.rs`.

## Wire-First

Live worker entry-point `agent-activation-worker` (`[project.scripts]`). Scale proof: 100 000 dormant instances @ ~1.54 KiB/inst + 1 000 concurrent activations touching only the activated instances (O(activations), not O(dormant)), peak RSS 283 MiB. Covered by `tests/unit/orchestration/test_agent_activation.py` + `_scale.py`.
