# Spec: L1 Cache Fidelity & Truth-in-Docs (make the tiered/100M claims real)

> CONCEPT:OS-5.5 (Massive-Scale Architecture). Surfaced by an external code
> review of `agent-utilities` + `epistemic-graph`. This spec validates that
> review against source and remediates the *real* gaps it exposed (and the ones
> it missed). Companion engine-side spec lives at
> `epistemic-graph/.specify/specs/review-remediation-20260609/spec.md`.

## Finding (what we validated against source)

An external reviewer claimed the projects were "docs + Python scaffolding" with
PyO3 FFI and no working tiered/subgraph layer. Validated claim-by-claim:

| Reviewer claim | Verdict | Source of truth |
|---|---|---|
| epistemic-graph "explicitly uses PyO3" | **FALSE** | `epistemic-graph/Cargo.toml` (no `pyo3`, `crate-type=["rlib"]`); `pyproject.toml` `[tool.maturin] bindings="bin"`; transport is UDS+msgpack (`epistemic_graph/pool.py:50`) |
| "no L1/L3 tiering / no subgraph load from L3" | **FALSE** | default backend is `tiered` (`backends/__init__.py:169`); `core/engine.py:618 load_subgraph()`, `load_for_centrality`, `load_for_impact_analysis`, `hydrate_compute_engine` |
| VF2 "couldn't confirm" | **FALSE** | `epistemic-graph/src/graph.rs:583 vf2_subgraph_match()`, wired `src/server.rs:998` |
| Ebbinghaus/temporal "not implemented" | **PARTLY TRUE** | schema only (`src/types.rs:62-68` confidence + validity window); **no decay function** |
| "100M agents" unvalidated | **TRUE** | `README.md:75`, `docs/.../OS-5.5-Massive_Scale_Architecture.md:5`; no scale benchmark |
| No benchmarks | **PARTLY** | transport bench exists (`epistemic-graph/scripts/bench_transport.py`); no algo/scale bench |

**The reviewer chased a phantom (PyO3) — misled by our own stale docs** —
`epistemic-graph/README.md:22,41` ("PyO3 FFI") and `docs/journey.md:127`
("PyO3-bound") contradict the authoritative `AGENTS.md:18` / OS-5.5 ("no PyO3").

**The four real gaps** (pre-mortem against the 100M claim):

- **Gap A — L1 can't traverse edges.** `backends/tiered_backend.py:46-48` routes
  relationship-pattern reads to **L3 Postgres** because *"the L1 epistemic
  interpreter can't traverse edges (it returns every node)."* The Rust engine
  *can* (`graph.rs` DFS/BFS/neighbors). The one op L1 exists to accelerate falls
  back to the slow tier — this nullifies the performance story. The file's own
  docstring is also self-contradictory ("Reads never touch L3" line 16 vs line
  47).
- **Gap B — subgraph loop is load-only + detached.** `load_subgraph()`
  (`engine.py:618`) builds a *fresh detached* engine and never writes deltas
  back; `reconcile_to_durable()` (`tiered_backend.py:267`) does **full
  enumeration** (`list(graph._get_all_nodes())`), not dirty-delta. No
  checkout→mutate→write-back-delta cycle, no background non-blocking sync.
- **Gap C — doc/reality drift** (PyO3, Ebbinghaus, un-scoped 100M) destroys
  reviewer trust in the real substance.
- **Gap D — 100M is unvalidated** and rests on Gaps A/B. The design is sound
  (100M *concurrent agents each on a bounded subgraph*, not one 100M-node graph),
  but that defense is only real once bounded-checkout + write-back are the
  default path and a benchmark proves it.

## User Stories

### US-1: Docs match the implementation (P0 — Gap C)
**As** a new evaluator, **I want** every architecture claim to match the code,
**so that** the real substance isn't dismissed over a false PyO3/forgetting claim.
- [ ] T1.1 — Purge PyO3 from `epistemic-graph/README.md:22,41` → "out-of-process MessagePack over UDS/TCP; no in-process FFI". (engine spec T-E1)
- [ ] T1.2 — Fix `agent-utilities/docs/journey.md:127` "PyO3-bound" → MessagePack/UDS client.
- [ ] T1.3 — Reword Ebbinghaus claim (`epistemic-graph/README.md:52`) to "temporal-validity + belief-confidence schema; decay curve = roadmap" UNLESS US-? engine decay lands first (then keep claim, link impl).
- [ ] T1.4 — Reframe "100M" in `README.md:75` + `OS-5.5-Massive_Scale_Architecture.md` as "architectural blueprint; current validation: single-node + transport bench" with a maturity banner.
- [ ] T1.5 — Add a one-paragraph "where this is on the maturity curve" section to both READMEs.
- **Accept:** `grep -ri pyo3` across both repos returns only historical/changelog mentions; no claim states a capability without a maturity qualifier or a passing test.

### US-2: L1 serves relationship traversal natively (P1 — Gap A, the crux)
**As** the tiered backend, **I want** single/multi-hop reads resolved in the Rust
L1 engine, **so that** "L1 cache" is true and traversal no longer falls to L3.
- [ ] T2.1 — Expose the Rust engine's neighbor/DFS/BFS/path ops through the L1 epistemic query interpreter so a relationship pattern (`-[..]-`, `[*1..n]`) resolves in L1 instead of "returns every node". (`backends/epistemic_graph_backend.py`, `core/graph_compute.py`; engine support per engine spec.)
- [ ] T2.2 — Update `_is_traversal()` routing in `backends/tiered_backend.py:53` so traversal reads go to **L1** when L1 can satisfy them; L3 only for cold-miss / unsupported patterns.
- [ ] T2.3 — Fix the contradictory `tiered_backend.py` docstring (lines 16 vs 46-48) to describe the new routing truthfully.
- [ ] T2.4 — Parity test: same Cypher traversal returns identical results from L1 and L3 over a seeded graph (no "returns every node").
- **Accept:** a `MATCH (n)-[*1..3]-(t {id}) RETURN n` read hits L1 only (assert via backend counters `_l3_writes`/new `_l3_reads`); parity test green.

### US-3: Bounded checkout → mutate → delta write-back (P2 — Gap B)
**As** an agent working a subgraph, **I want** to check out a bounded working set,
mutate it in L1, and flush **only the deltas** to L3 in the background, **so that**
writes are fast, durable, and non-blocking.
- [ ] T3.1 — Add a dirty-set / edit-ledger to the checked-out `GraphComputeEngine` returned by `engine.py:618 load_subgraph()` (track add/update/delete of nodes+edges).
- [ ] T3.2 — `flush_deltas_to_durable()` that writes only changed nodes/edges, batched into one transaction (replaces full-enumeration on the hot path; keep `reconcile_to_durable` as periodic safety net).
- [ ] T3.3 — Run flush **non-blocking** (background task/queue via the consolidated scheduler, à la KG-2.8 `reconcile_durable`); agent loop never blocks on L3.
- [ ] T3.4 — Checkout epoch/version stamp + conflict policy on write-back (compare-on-write; last-writer-wins + log, or reject+reload).
- [ ] T3.5 — Unify: the default `TieredGraphBackend` read path uses the bounded loader, not a detached one-off, so checkout and tiered routing are one mechanism.
- **Accept:** mutate-then-flush writes only touched rows (assert delta count « full count); flush runs off the request hot path; conflict test exercises a concurrent-L3-change case.

### US-4: Scale & performance are measured, not asserted (P3 — Gap D)
**As** a maintainer, **I want** benchmarks that turn "100M" into a measured
projection, **so that** the claim is defensible.
- [ ] T4.1 — Scale harness: N agents × bounded subgraph; measure checkout / compute / delta-flush latency + RSS; publish to `epistemic-graph/docs/benchmarks.md`. (engine spec T-E3)
- [ ] T4.2 — Convert "100M" claims to a measured extrapolation (per-agent footprint × concurrency, with the bounded-subgraph assumption stated explicitly).
- [ ] T4.3 — Wire a `benchmark` smoke into CI (`tests/integration/server/test_benchmark_router.py` already exists) so regressions surface.
- **Accept:** `docs/benchmarks.md` shows per-agent RSS + p50/p99 checkout/flush; README "100M" links to the extrapolation table.

### US-5: Onboarding & signaling (P4 — reviewer's valid soft points)
**As** a first-time visitor, **I want** a 5-minute path and clear repo scoping,
**so that** the "utilities vs platform" and two-repo split don't confuse.
- [ ] T5.1 — Add `CONTRIBUTING.md` to both repos.
- [ ] T5.2 — README: lead with a 5-minute quickstart (agent factory) **before** the manifesto.
- [ ] T5.3 — Add a short "epistemic-graph is the separate Rust engine" note + link in `agent-utilities/README.md`, and the reverse link in the engine README.
- **Accept:** README first screen is runnable in <5 min; both repos cross-link.

## Non-Functional Requirements
- [ ] No behavior change to the agent factory (`agent/factory.py:246`) — it's the honest, working entrypoint; keep it green.
- [ ] All existing backend / tiered / assimilation tests stay green; ruff + mypy clean.
- [ ] No new top-level concept id — this is an OS-5.5 refinement, coordinated with KG-2.8 (durable-tier-autoheal).
- [ ] Changes are additive/back-compatible: `memory`/`file`/`postgresql` backends and `GRAPH_BACKEND` selection unchanged.

## Execution order
P0 (US-1) first — cheap, immediately defuses the review. Then P1 (US-2, the
performance crux) → P2 (US-3, closes the loop) → P3 (US-4, proves it) → P4 (US-5).
US-2/US-3 depend on engine-side support tracked in the companion spec.

## Status
**PLANNED** — 2026-06-09. Cross-repo program; engine slice in
`epistemic-graph/.specify/specs/review-remediation-20260609/`.
