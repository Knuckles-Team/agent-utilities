# Design Document: Cross-modal warm-fork fan-out

> The agent-utilities side of the epistemic-graph cross-modal seam (was engine spec **EG-397**).
> Concept: `CONCEPT:AU-ORCH.sandbox.crossmodal-fork-fanout` (flat `ORCH-1.106`).

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `AU-KG.coordination.warm-fork-fanout` (KG-2.323) | warm fork fanout — the `graph_fork` verb | ~80% | KG |
| `AU-ORCH.sandbox.warmforkfanoutcapability` | warm-fork fan-out capability (ORCH-1.86..93) | ~75% | ORCH |
| `AU-ORCH.sandbox.shared-host-helper-bridge` | ForkableSandbox + WarmParentRegistry primitive | ~70% | ORCH |
| `AU-ORCH.sandbox.tiered-rlm-sandbox` | Sandbox contract / capabilities | ~55% | ORCH |
| `AU-KG.retrieval.*` (hybrid retriever) | vector+graph+text fusion candidate set | ~50% | KG |

### Extension Analysis

- **Primary Extension Point**: `CONCEPT:AU-ORCH.sandbox.warmforkfanoutcapability` +
  `CONCEPT:AU-KG.coordination.warm-fork-fanout`.
- **Extension Strategy**: compose. The existing warm-fork primitive (`ForkableSandbox.execute`
  warms/reuses one parent through `WarmParentRegistry`; the `graph_fork` verb fans branches out
  over it) already exists. What did **not** exist is the *seam that consumes the engine's
  cross-modal context*: retrieve a vector+graph+text candidate set from the epistemic-graph
  engine ONCE, hold it in the warm parent, and fork N copy-on-write branches that reuse that one
  candidate set with no recompute. That is a distinct capability (a specific consumer of the
  engine cross-modal fusion), so it gets its own governed concept and composes the two existing
  ones rather than duplicating them.
- **New Concept Required?**: Yes — the reuse-the-engine-cross-modal-context-across-a-fork-cohort
  behaviour is not expressible as either the generic `graph_fork` verb (which only seeds a
  caller-supplied `vars`) or the bare `ForkableSandbox` (which knows nothing about the engine).

### New Concept Proposal

- **Proposed ID**: `CONCEPT:AU-ORCH.sandbox.crossmodal-fork-fanout` (flat `ORCH-1.106`).
- **Augments Pillar**: ORCH (sandbox / warm-fork family), consuming KG cross-modal retrieval.
- **Justification**: closes the last handoff-1 loose end — the engine kept `EG-397` `#[ignore]`d
  because the warm-fork primitive lives in agent-utilities, not the engine. This is that
  agent-utilities half: `agent_utilities/runtime/crossmodal_fork.py`
  (`CrossModalForkFanout`) + the `context_query` path on the `graph_fork` verb.

## Capability

1. **Retrieve once.** A `CrossModalRetriever` (defaulting to `HybridRetriever.retrieve_hybrid`,
   the engine vector+graph+text fusion arm) is invoked exactly once through a `_RecomputeGuard`
   that raises `RecomputeError` on any second retrieval.
2. **Warm one parent.** A single warm-fork parent is warmed via `WarmParentRegistry`
   (imports/deps resident copy-on-write, paid once).
3. **Fork N CoW branches.** Each branch reuses that one candidate set (bound to `candidates` in
   its namespace) and runs its own divergent computation; a forked child is a separate process
   with its own copy, so a branch's mutation cannot leak into a sibling.

**Reuse proof**: `CrossModalForkResult.retrieval_calls == 1` regardless of branch count.
**Isolation proof**: divergent per-branch mutations are observed only by the mutating branch.

## Honest scope / what a live rung adds

The local `forkserver` rung shares loaded *modules* copy-on-write and serialises the fork step
(the stdlib forkserver control socket is a process singleton — `max_concurrency` defaults to 1).
A live KV-cache-fork rung (LMCacheMPConnector snapshot → branch, the vLLM path) additionally
shares the candidate set's KV/embedding pages as copy-on-write memory, making the data-sharing
itself zero-copy and enabling true parallel fan-out — the same `ForkableSandbox` seam, a stronger
backend.
