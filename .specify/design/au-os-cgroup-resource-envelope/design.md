# Design Document: One canonical container-aware CPU/memory envelope reader for every autosizing call site

CONCEPT:AU-OS.host.cgroup-resource-envelope

> Realised by `agent_utilities/core/cgroup_resources.py`
> (`effective_cpu_cores`/`effective_memory_limit_bytes`), consumed by
> `agent_utilities/knowledge_graph/core/engine_tasks.py:333`
> (`compute_ingest_worker_count`) and
> `agent_utilities/runtime/warm_registry.py:44`
> (`compute_warm_parent_count`) — the two autosizing helpers AGENTS.md's
> *Configuration discipline* section already names as the canonical
> "auto-size a hardware tunable" pattern.

## KG Analysis

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| AU-OS.host.so-they-are-idle | Warm-parent pool idle reap | low — different concern (lifecycle/reaping, not sizing input) | OS |
| AU-KG.coordination.embedder-breaker | KG background-daemon role resolution | low — different concern (role, not resource envelope) | KG |

### Extension Analysis

- **Primary Extension Point**: none — `compute_ingest_worker_count` and
  `compute_warm_parent_count` are referenced in AGENTS.md prose as the
  canonical auto-sizing helpers, but neither that prose nor either call site
  carries an existing `CONCEPT:` id that this could extend; there is no live,
  documented parent to declare in `agent_utilities/governance/concept_lineage.yaml`.
- **Extension Strategy**: new
- **New Concept Required?**: Yes

### New Concept Proposal

- **Proposed ID**: CONCEPT:AU-OS.host.cgroup-resource-envelope
- **Augments Pillar**: OS
- **15-Phase Pipeline Integration**: none (host/runtime concern, not a KG
  ingestion/orchestration pipeline phase)
- **Justification**: see Decision below.

## Decision — read the tighter of the cgroup limit and the host view, in ONE place, and route every autosizing helper through it

**The bug.** The live r18/r19 GraphOS pod ran inside a `1.5`-CPU / `4`-GiB
cgroup (`cpu.max = 150000 100000`), but every autosizing call site read
`os.cpu_count()` and `psutil.virtual_memory()` — the **host's** 192-core,
multi-terabyte view, not the container's. `compute_ingest_worker_count`
sized 69 ingest workers into a cgroup that could physically schedule 1.5 of
them concurrently, producing 65.9% throttled CPU scheduling periods and
starving interactive query/engine-call latency (U-64/U-65/U-73,
BUG-110).

**The fix.** `agent_utilities/core/cgroup_resources.py` is the one module
that reads the effective (host-or-container, whichever is tighter) resource
envelope: `cgroup_cpu_limit_cores()`/`cgroup_memory_limit_bytes()` read
`/sys/fs/cgroup/cpu.max` + `/sys/fs/cgroup/memory.max` (cgroup v2, unified
hierarchy) or `cpu.cfs_quota_us`/`cpu.cfs_period_us` +
`memory.limit_in_bytes` (cgroup v1) directly (no external dependency),
and `effective_cpu_cores()`/`effective_memory_limit_bytes()` take the
`min()` of that cgroup limit and the host-visible value, never exceeding
either. `compute_ingest_worker_count` and `compute_warm_parent_count` both
now anchor to these two functions instead of calling `os.cpu_count()` /
`psutil.virtual_memory()` directly — the same auto-sizing helpers
AGENTS.md's *Configuration discipline* section already names as the
canonical "don't add a concurrency env flag, auto-size it" pattern; this
decision corrects what that auto-sizing reads, not the auto-sizing
principle itself.

**Why every read is best-effort, fail-open to the host view.** A missing
cgroup file, an unreadable path (non-Linux, no cgroup controller mounted,
a bare-metal host process), or the kernel's "unlimited" sentinel
(`"max"` for v2; `-1` quota or a value at/above `2**62` for v1) all resolve
to `None`, and every caller then falls back to the host-visible value —
so a genuine bare-metal deployment keeps today's host-sized behavior
unchanged. The alternative — treating a read failure as "assume tightly
constrained" — was rejected: it would silently under-size every
non-containerized deployment (the overwhelmingly common case for `tiny`/
`single-node-prod` profiles) to guard against a container-only failure
mode, trading a real, common regression for a hypothetical, container-only
one.

**Why one shared module instead of inlining the read at each call site.**
Two independent call sites (`compute_ingest_worker_count`,
`compute_warm_parent_count`) need the identical tighter-of-cgroup-or-host
computation, and a bug in that computation is exactly the class of defect
this decision exists to close — a second inlined copy is a second place
that same class of bug (or a future cgroup v3 shape) can reappear
independently. `agent_utilities/core/cgroup_resources.py` is a pure,
dependency-light (`os` + optional `psutil`) module in `agent_utilities/core/`,
consistent with *Sprawl boundaries*' "the ONE place" convention for a
capability with multiple consumers.

## Data Flow

1. **ORCH**: not directly invoked by the orchestrator; it is read at
   ingest-worker-pool and warm-parent-pool sizing time, both of which are
   ORCH-adjacent capacity decisions.
2. **KG**: no node/edge reads or writes — pure `/sys/fs/cgroup/*` file
   reads plus optional `psutil.virtual_memory()`.
3. **AHE**: not part of a self-improvement cycle.
4. **ECO**: not exposed as an MCP tool or A2A capability — an internal
   sizing primitive, not an operator-facing surface.
5. **OS**: this IS the OS-pillar host-resource-detection primitive that
   `compute_ingest_worker_count`/`compute_warm_parent_count` (both already
   governed by *Configuration discipline*'s auto-sizing rule) depend on.

## Risk Assessment

- **Blast Radius**: `agent_utilities/core/cgroup_resources.py` (new,
  additive module); `agent_utilities/knowledge_graph/core/engine_tasks.py`'s
  `compute_ingest_worker_count` and `agent_utilities/runtime/warm_registry.py`'s
  `compute_warm_parent_count` (both change their CPU/memory *input source*,
  not their sizing formula).
- **Backward Compatible**: Yes for any deployment without a cgroup limit
  (bare-metal, non-Linux, bare containers with no quota set) — the effective
  values are unchanged from the pre-existing host-only reads. A deployment
  running under a real cgroup CPU/memory limit now sizes smaller, which is
  the fix, not a regression.
- **Known weak point**: cgroup v2's `cpu.max` and v1's
  `cpu.cfs_quota_us`/`cpu.cfs_period_us` are read as two independent file
  opens (not one atomic snapshot); a limit changed by the container runtime
  between the two reads could theoretically be read inconsistently. Not
  observed in practice (the limit is set once at container creation and is
  effectively static for the pod's lifetime), and inconsistency here only
  ever biases the computed core count by at most one file's staleness
  window, never past the host ceiling.
