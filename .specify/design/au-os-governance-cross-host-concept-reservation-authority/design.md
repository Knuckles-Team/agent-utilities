# Design Document: Cross-host concept reservation authority

CONCEPT:AU-OS.governance.cross-host-concept-reservation-authority

> Full prose lives at
> [`docs/architecture/concept-reservation-authority.md`](../../../docs/architecture/concept-reservation-authority.md)
> (an operator-facing architecture page, not under `.specify/design/`, so it
> does not itself satisfy `scripts/check_concept_governance.py`'s design-doc
> gate). This file is the gate-visible pointer to that decision — see this
> concept's own module docstring in
> `agent_utilities/governance/concept_reservation.py` for the authoritative
> source.

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `AU-OS.governance.concept-id-coordination` | the local, single-checkout concept-id ledger (`agent-utilities concept reserve`) | high | OS |

### Extension Analysis

- **Primary Extension Point**: the existing local `git-common-dir`-locked
  concept ledger, which only serializes writers that share one checkout.
- **Extension Strategy**: augment — a second, cross-host-safe backend for the
  same "reserve a concept id before writing its marker" problem, not a new
  problem.
- **New Concept Required?**: Yes — separate clones/hosts have separate locks
  and ledgers, so the local allocator alone cannot prevent two hosts from
  reserving the same id.

## Problem

`agent_utilities.governance.concept_reservation`'s local allocator (a shared
`git-common-dir` lock) only serializes writers that share one linked-worktree
checkout. Separate clones and separate hosts each have their own lock and
ledger, so two independent hosts can both "win" a reservation for the same
concept id with nothing to arbitrate between them.

## Decision

Route cross-host reservation through the epistemic-graph engine's existing
durable primitives instead of inventing a new coordination store:

- `CreateNodeIfAbsent(node_id, properties)` atomically chooses the first
  writer across every host that can reach the engine.
- `CompareAndSetNodeFields(node_id, conditions, updates)` performs a fenced
  lifecycle/reclaim update under the engine's own write guard.
- `GetNodeProperties` / bounded `GetNodesByLabel` provide point/query reads.

One authoritative graph, one node identity per concept id
(`concept-reservation:<concept_id>`). Cross-host clients route to that same
graph authority; a local Git lock, JSON file, or fixture is never treated as a
global fallback. The local allocator's ledger remains a useful auditable,
merge-friendly *projection* — it is not replaced, only no longer assumed
sufficient once more than one checkout/host is in play.

## Wire-First

`agent_utilities/governance/concept_reservation.py` implements the adapter;
see `docs/architecture/concept-reservation-authority.md` for the full
request/replay/conflict semantics and the reasoning behind the single
canonical node identity.
