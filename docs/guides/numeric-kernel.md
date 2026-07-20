# Numeric kernel — the `xp` backend (Analytics Program P1–P5)

`agent_utilities.numeric` exposes a numpy-compatible namespace, `xp`, that routes the
numeric ops agent-utilities actually uses through the compiled **`eg-numeric`** kernel
(pure-Rust faer + ndarray, BLAS/LAPACK-free). It is the **sole numeric backend**. Call
sites use it as a drop-in:

```python
from agent_utilities.numeric import xp as np   # instead of `import numpy as np`
```

Every numeric call site imports `xp`, so the body stays unchanged — only the import line
differs from plain numpy.

## P5 final — the hard numpy/scipy drop (CONCEPT:AU-KG.compute.numpy-scipy-drop)

numpy and scipy are **fully removed from agent-utilities**. The package **imports numpy
nowhere** and **declares it in no dependency** — `requirements.txt` and `pyproject.toml`
carry neither numpy nor scipy. The compiled `epistemic_graph.numeric` kernel is the
**sole numeric backend** and a **hard requirement**:

- `from agent_utilities.numeric import xp` **raises `ImportError`** when the compiled
  kernel is absent. There is **NO alternate numeric module or numpy fallback** for a
  missing kernel.
- The four scipy ops are now **native kernel exports** (engine CONCEPT:EG-KG.compute.concept-5), reached as
  `xp.eigsh` (`scipy.sparse.linalg.eigsh`, k smallest-magnitude symmetric eigenpairs),
  `xp.spearmanr`, `xp.ks_2samp`, and `xp.norm_ppf` / `xp.norm_pdf`
  (`scipy.stats.norm.ppf` / `.pdf`). No scipy import remains anywhere in agent-utilities.

### How numpy still works without agent-utilities depending on it

The kernel is a **rust-numpy container**: numpy lives **inside the numeric component of
`epistemic-graph[full]`**
as the kernel's own zero-copy interop dependency, and the compiled module re-exports
numpy's array primitives (`ndarray`, the dtypes, `newaxis` / `pi` / `inf` / `nan`). The
`xp` shim obtains that module from the kernel itself
(`sys.modules[_KERNEL.ndarray.__module__]`) — **never via an `import numpy` statement**
and **never as an agent-utilities dependency**.

The kernel's compiled fast path is deliberately **narrow**:

| Op class | Kernel fast path | Otherwise |
|----------|------------------|-----------|
| Reductions / stats (`sum`/`mean`/`std`/`var`/`min`/`max`/…) | eligible `ndarray` / `list` / `tuple`, incl. N-D + `axis` + `keepdims` | kernel-internal numpy (pandas `Series`/`DataFrame` wrappers preserved) |
| Element-wise (`sqrt`/`log`/`exp`/`abs`/`clip`/…) | contiguous 1-D `float64` `ndarray` / `list` | kernel-internal numpy (preserves ufunc-dispatch wrappers) |
| Linalg (`norm`/`solve`/`svd`/`eigh`/…) | contiguous 1-D/2-D `float64` | kernel-internal numpy (`axis` norms, batched, complex) |
| scipy ops (`eigsh`/`spearmanr`/`ks_2samp`/`norm_ppf`/`norm_pdf`) | always native kernel | — |

So numpy is an **internal implementation detail of the kernel** — used for the long tail
the compiled kernel does not expose natively (the `random` Generator API, `cov`/`corrcoef`,
`save`/`load`, axis norms, N-D element-wise, pandas-wrapped inputs) — and agent-utilities'
whole numeric surface flows through this one module.

### Packaging

| Where | Contents | Role |
|-------|----------|------|
| base `dependencies` | `epistemic-graph[full]>=2.23.1,<3.0.0` (the certified protocol range) | Full CPU features and the kernel are a hard runtime contract. There is one release artifact: the platform-specific numeric library is folded into the `epistemic-graph` wheel (`epistemic_graph.numeric`, engine CONCEPT:AU-KG.compute.is-installed-kernel-discovery); `[full]` pulls the NumPy ABI dependency used internally through its numeric member. Release and Agent Utilities consumer CI must import the module and execute a real operation, so an incomplete wheel cannot pass merely because its metadata declares the extra. |
| `[test]` extra | `numpy>=2.4.6` | **Dev/test-only** ground-truth reference for `tests/test_numeric_parity.py`. NEVER a runtime dependency. |

- numpy/scipy are **NOT** in base `dependencies` and **NOT** in any leaf extra
  (`finance`/`embeddings`/`ann` no longer declare them; `finance` gets numpy/scipy
  transitively via statsmodels / hmmlearn / pandas for their own use, not agent-utilities').
- The `numeric-fallback` extra is **deleted** — there is no fallback path any more.
- All floors are **loose** (`>=`), never exact pins.
- Parity is enforced: `tests/test_numeric_parity.py` asserts `np.allclose(kernel, numpy)`
  on randomized inputs (incl. nan/inf/singular matrices), and the engine's `numeric-parity`
  CI job (engine CONCEPT:AU-KG.compute.is-installed-kernel-discovery) gates the kernel against numpy so the two can never diverge.

## Dev vs prod: two different install paths

**Prod / published installs** pull the kernel through the mandatory base dependency:

```bash
pip install agent-utilities                       # base requires epistemic-graph[full]>=2.23.1,<3.0.0
python -c "from agent_utilities.numeric import xp; print(xp.sum([1.0, 2.0]))"
```

There is **ONE published package**: the `eg-numeric` Surface-A `.so` (cp39-abi3) is
**folded into the `epistemic-graph` wheel** as `epistemic_graph.numeric`. The sole
supported `[full]` engine profile includes numpy for the kernel's zero-copy interop.
There is **no separate `eg-numeric` package on PyPI**.

**Dev is editable and non-publishing.** Build the full epistemic-graph wheel from the
sibling checkout and install that wheel into the development environment. The runtime
module remains `epistemic_graph.numeric`; a top-level development module is not a
supported package shape.

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin build --release
python -m pip install --force-reinstall target/wheels/epistemic_graph-*.whl
python -c "from agent_utilities.numeric import xp; print(xp.sum([1.0, 2.0]))"
```

With the kernel installed, `test_numeric_parity.py` exercises the KERNEL path against numpy as
ground truth. Remove the engine wheel and `from agent_utilities.numeric import xp`
raises the clean `ImportError` — proving there is no silent numpy fallback.

## Honest status — the drop is complete (CONCEPT:AU-KG.compute.numpy-scipy-drop)

- numpy/scipy are **gone from agent-utilities' declared dependencies and source imports**
  (grep for `import numpy` / `import scipy` across `agent_utilities/` + `scripts/` is zero;
  the only direct numpy import is the dev-only parity test, which `pytest.importorskip`s it).
- The kernel is the **sole numeric backend** and a **hard base dependency**; the shim is
  **kernel-or-raise** (no fallback).
- numpy persists ONLY as an **internal detail of the kernel** (rust-numpy) — reached through
  the kernel, not through an agent-utilities import — serving the long-tail array ops the
  compiled kernel does not yet expose natively.

### Candidates for future native kernel ops

To shrink the kernel-internal-numpy tail to zero, the kernel would need to natively expose:
a seeded **`random` Generator** surface (`default_rng` → `normal`/`uniform`/`integers`/
`choice`/`shuffle`/`standard_normal`/`random`; the module-level seeded `normal`/`uniform`/
`integers` already exist), **`cov`/`corrcoef`**, **`save`/`load`** (`.npy`), axis-aware
**`linalg.norm`**, and N-D element-wise (`sqrt`/`log`/`exp`/`clip`/…). Until then the shim
routes those through the kernel's bundled numpy — correct and parity-clean, just not compiled.
