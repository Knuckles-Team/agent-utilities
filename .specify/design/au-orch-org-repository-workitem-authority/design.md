# Design Document: Durable repository-development WorkItem authority

CONCEPT:AU-ORCH.org.repository-workitem-authority

> `agent_utilities/orchestration/repository_work_item.py`
> (`submit_repository_work_item`, `claim_repository_work_item`,
> `checkpoint_repository_work_item`, `heartbeat_repository_work_item`,
> `commit_repository_work_item`, `list_repository_work_items`,
> `get_repository_operation_payload[_for_claim]`).

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| the engine-native `WorkItem` state machine (state, dependencies, leases, retries, fences, checkpoints, cancellation, terminal effects) | the sole existing durable-task authority this module adapts, not replaces | high | ORCH |

### Extension Analysis

- **Primary Extension Point**: the engine-native `WorkItem` state machine.
- **Extension Strategy**: adapt — project Repository Manager's job vocabulary
  onto the existing `WorkItem` authority through a frozen JSON contract,
  never a second state machine.
- **New Concept Required?**: Yes — the adapter boundary itself (what crosses
  the process boundary and what stays on each side) is the decision.

## Problem

Repository Manager (a separate package) and agent-utilities need to share
durable work-item state for repository-development jobs, without either
package importing the other's internals, and without a second, competing
notion of "what state is a job in" alongside the engine's own `WorkItem`
authority.

## Decision

This module is the agent-utilities side of the repository-development v1
boundary. It deliberately does **not** import Repository Manager: the two
packages exchange a frozen JSON contract, and this adapter projects that
contract onto the one engine-native `WorkItem` state machine. The adapter
stores only opaque repository/job correlations and content digests —
repository paths, command bodies, credentials, and log contents remain in the
Repository Manager domain or artifact store, never duplicated into the graph.
The `WorkItem` stays the sole authority for state, dependencies, leases,
retries, fences, checkpoints, cancellation, and terminal effects; this module
adds no parallel bookkeeping of any of those.

## Wire-First

`submit_repository_work_item`/`claim_repository_work_item`/
`checkpoint_repository_work_item`/`heartbeat_repository_work_item`/
`commit_repository_work_item`/`list_repository_work_items` are the adapter's
public surface; `get_repository_operation_payload[_for_claim]` resolve the
opaque payload a caller acts on.
