# Design Document: The hard numpy/scipy drop

> Backfilled under the concept-lineage rule (CONCEPT:AU-OS.governance.concept-lineage-parent-doc).
> One decision, authored in `agent_utilities/numeric/__init__.py`.

CONCEPT:AU-KG.compute.numpy-scipy-drop

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `AU-KG.compute.graph-compute-engine` | the other half of "the engine is the runtime"; different subsystem, same conviction | 0.40 | KG |
| `AU-OS.deployment.workspace-venv-reconciler` | shares the failure mode this creates — a hard runtime dependency on a compiled wheel | 0.30 | OS |

### Extension Analysis

- **Primary Extension Point**: `agent_utilities/numeric/` (the `xp` namespace).
- **Extension Strategy**: new surface, replacing a dependency.
- **New Concept Required?**: No.

## The decision

`CONCEPT:AU-KG.compute.numpy-scipy-drop` — `agent_utilities/numeric/__init__.py:10-31`.

numpy and scipy are **fully removed** from agent-utilities: the package
imports numpy nowhere and declares it in no dependency file
(`requirements.txt`/`pyproject.toml` carry neither). The compiled
`epistemic_graph.numeric` kernel (pure-Rust `faer` + `ndarray`,
BLAS/LAPACK-free) is the SOLE numeric backend, and it is a **hard**
requirement: importing `agent_utilities.numeric` raises `ImportError` when
the kernel is absent.

### The alternative that was rejected, explicitly

**A numpy fallback.** It is the obviously safer design and it was refused on
purpose — the module states directly "there is no alternate-module or numpy
fallback." A fallback would mean the compiled kernel's numerical behavior and
numpy's could silently diverge on the same input, discoverable only when a
result was subtly WRONG rather than cleanly missing. `tests/test_numeric_parity.py`
exists precisely because the two must agree; a fallback would make that
parity test optional in production instead of load-bearing.

The cost is a hard deployment coupling — an environment without
`epistemic-graph[full]` cannot import the module at all. That is deliberate:
loud absence is preferred over quiet divergence.

### The subtlety this decision turns on

numpy still RUNS; agent-utilities just does not DEPEND on it. The compiled
kernel is a rust-numpy container: numpy lives inside the numeric component of
the `epistemic-graph[full]` runtime as the kernel's own zero-copy interop
dependency, and the compiled module re-exports numpy's array primitives
(`ndarray`, dtypes, `newaxis`/`pi`/`inf`/`nan`). The kernel's compiled
fast-path is deliberately NARROW — contiguous 1-D/2-D `float64` for
element-wise + linalg ops; general (lists/N-D/`axis`/`keepdims`/int) for
reductions/stats. For inputs OUTSIDE that fast-path (N-D element-wise,
`axis` norms, the `random` Generator API, `cov`/`corrcoef`/`save`/`load`) the
shim delegates to the numpy module the KERNEL ITSELF already loaded —
obtained via `sys.modules[_KERNEL.ndarray.__module__]`, never through an
`import numpy` statement in this package.

This is the part that reads as contradictory from outside, and is exactly why
the decision needs a document rather than a bare marker: "we removed numpy"
and "numpy executes some of our N-D element-wise ops" are both true
simultaneously, and the reconciliation is the DEPENDENCY EDGE (agent-utilities
declares no numpy dependency), not the call graph (numpy code still executes,
just as the kernel's internal implementation detail).

**What breaks if violated**: adding `import numpy` directly anywhere in
agent-utilities (bypassing the `xp` shim) reintroduces exactly the dependency
this decision removed, and bypasses the parity guarantee `test_numeric_parity.py`
exists to enforce — a direct numpy call has no obligation to agree with the
compiled kernel's fast-path behavior on the same input.

### The three per-surface markers — instances of the one decision

`surface-analytics-program` (`numeric/__init__.py:3`) — Surface A of the
Analytics Program's "one kernel, two surfaces": the `xp` namespace mirroring
the subset of the numpy API a 598-site audit found agent-utilities actually
uses (reductions/stats, element-wise, the linalg-6 + `LinAlgError`, random,
four scipy ops). `numeric-kernel` (`numeric/__init__.py:4`) is the
engine-side surface of that same pair — the compiled kernel itself.
`ufunc-method-surface` (`numeric/__init__.py:40`) — `xp.maximum`/`xp.minimum`
are small `_Ufunc` wrappers so `.accumulate`/`.reduce`/`.outer`/`.at` route to
the kernel's cumulative op on a bare 1-D `float64` input; a shape detail of
the shim's API surface, not a separate design choice.

## Data Flow

1. **ORCH**: none.
2. **KG**: every numeric path in the KG (embeddings, spectral analysis,
   scoring) flows through this one module.
3. **AHE**: the harness's compiled-kernel note (`harness/__init__.py:68`)
   documents the same hard dependency from the harness side.
4. **ECO**: none.
5. **OS**: the runtime contract is a pinned `epistemic-graph[full]` version
   range — see `docs/guides/numeric-kernel.md`.

## Risk Assessment

- **Blast Radius**: every numpy call site in the package (598 audited at
  migration time).
- **Backward Compatible**: at the call-site level yes — the migration
  (`from agent_utilities.numeric import xp as np`) keeps the `np` alias, so
  expression bodies are unchanged; at the ENVIRONMENT level no.
- **Breaking Changes**: an environment without the compiled kernel cannot
  import `agent_utilities.numeric` at all. This is the intended behavior, not
  a regression — loud absence over quiet divergence.
- **Known hazard**: published wheels have shipped without the numeric kernel
  before (the 2.14.0-2.23.0 packaging regression — see
  `eg-wheel-kernel-packaging-regression`), and this decision converts that
  from a silently-degraded mode into a hard import failure. The pinned
  version range in `docs/guides/numeric-kernel.md` is the mitigation, not a
  guarantee that a bad wheel is caught before deploy.
