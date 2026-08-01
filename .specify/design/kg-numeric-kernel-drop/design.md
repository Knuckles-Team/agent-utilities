# Design Document: The hard numpy/scipy drop

> One decision, authored in `agent_utilities/numeric/__init__.py` and its module
> header. Backfilled under the concept-lineage rule
> (CONCEPT:AU-OS.governance.concept-lineage-parent-doc): four sibling
> `AU-KG.compute` markers are surfaces of this one decision and point here.

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
- **New Concept Required?**: No new ones. This names the decision the markers
  already carried.

## The decision

numpy and scipy are **fully removed** from agent-utilities. The package imports
numpy nowhere and declares it in no dependency file. The compiled
`epistemic_graph.numeric` kernel (pure-Rust `faer` + `ndarray`, BLAS/LAPACK-free)
is the sole numeric backend, and it is a **hard** requirement: importing
`agent_utilities.numeric` raises `ImportError` when the kernel is absent.

### The alternative that was rejected, explicitly

**A numpy fallback.** It is the safe design and it was refused on purpose:
"there is no alternate-module or numpy fallback". A fallback would mean the
compiled kernel's numerical behaviour and numpy's could silently diverge on the
same input, and nobody would find out until a result was wrong rather than
missing. `tests/test_numeric_parity.py` exists precisely because the two must
agree; a fallback would make the parity test optional in production.

The cost is a hard deployment coupling — an environment without
`epistemic-graph[full]` cannot import the module at all. That is deliberate:
loud absence over quiet divergence.

### The subtlety this decision turns on

numpy still *runs*; agent-utilities just does not *depend* on it. The kernel is
a rust-numpy container: numpy lives inside the numeric component of the
`epistemic-graph[full]` runtime as the kernel's own zero-copy interop
dependency, and the compiled module re-exports numpy's array primitives. For
inputs outside the kernel's deliberately narrow compiled fast-path (contiguous
1-D/2-D `float64` for element-wise and linalg; general for reductions/stats) the
shim delegates to the numpy module **the kernel itself already loaded**,
obtained via `sys.modules[_KERNEL.ndarray.__module__]` — never through an
`import numpy` statement here.

This is the part that reads as contradictory from outside and is why the
decision needs a document rather than a marker: "we removed numpy" and "numpy
executes our N-D element-wise ops" are both true, and the reconciliation is the
dependency edge, not the call graph.

## What the pointers to this decision are

- `surface-analytics-program` — Surface A of the Analytics Program's "one
  kernel, two surfaces"; the `xp` namespace mirroring the subset of the numpy
  API that the 598-site audit found agent-utilities actually uses.
- `numeric-kernel` — the engine-side surface of that same pair.
- `ufunc-method-surface` — `xp.maximum`/`xp.minimum` as small `_Ufunc` wrappers
  so `.accumulate`/`.reduce`/`.outer`/`.at` route to the kernel's cumulative op
  on a bare 1-D float64 input. A shape detail of the shim, not a separate choice.
- `executed-p2-p3-rollout` — the mechanical call-site migration
  (`from agent_utilities.numeric import xp as np`). A rollout record, not a
  decision; its id is a past-tense sentence fragment, and it is retired rather
  than pointed.
- `is-installed-kernel-discovery` — a CI job description in
  `docs/guides/numeric-kernel.md` that gates the kernel against numpy. It exists
  only in prose, its id is a sentence fragment, and the parity contract it
  describes is stated above; retired.

## Data Flow

1. **ORCH**: none.
2. **KG**: every numeric path in the KG (embeddings, spectral analysis, scoring)
   flows through this one module.
3. **AHE**: the harness's compiled-kernel note in `harness/__init__.py`.
4. **ECO**: none.
5. **OS**: the runtime contract is a pinned `epistemic-graph[full]` range — see
   `docs/guides/numeric-kernel.md`.

## Risk Assessment

- **Blast Radius**: every numpy call site in the package (598 audited).
- **Backward Compatible**: at the call-site level yes (the `np` alias is kept, so
  expression bodies are unchanged); at the *environment* level no.
- **Breaking Changes**: an environment without the compiled kernel cannot import
  `agent_utilities.numeric`. This is the intended behaviour, not a regression.
- **Known hazard**: published wheels have shipped without the numeric kernel
  before (the 2.14.0-2.23.0 packaging regression), and this decision converts
  that from a degraded mode into a hard import failure. The pinned certified
  protocol range in `docs/guides/numeric-kernel.md` is the mitigation.
