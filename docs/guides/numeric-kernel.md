# Numeric kernel — the `xp` backend (Analytics Program P1–P5)

`agent_utilities.numeric` exposes a numpy-compatible namespace, `xp`, that routes the
numeric ops agent-utilities actually uses through the compiled **`eg-numeric`** kernel
(pure-Rust faer + ndarray, BLAS/LAPACK-free) when it is importable, and **falls back to
numpy** when it is not. Call sites use it as a drop-in:

```python
from agent_utilities.numeric import xp as np   # instead of `import numpy as np`
```

All 38 numeric call sites (28 light-op files, the linalg-6, and the 3 ufunc-method finance
files) already import `xp`, so switching backends is mechanical — no body changes.

## P5 — kernel is the PRIMARY backend, numpy is fallback-only (CONCEPT:KG-2.317)

The shim's kernel-discovery loop is **kernel-first**: it prefers `epistemic_graph.numeric` /
`numeric`; only if neither imports does it bind numpy. `HAVE_KERNEL` reflects which path is
live. Packaging now states the same intent explicitly:

| Extra | Contents | Role |
|-------|----------|------|
| `numeric-kernel` | `epistemic-graph>=2.0.0` (carries the `eg-numeric` kernel behind its maturin `numeric` feature) | **Primary** numeric backend. Installing it makes the shim kernel-LIVE (`HAVE_KERNEL == True`). |
| `numeric-fallback` | `numpy>=2.4.6`, `scipy>=1.17.1` | **Fallback** path. Kept so the abstraction works wherever the kernel wheel is absent. |

- numpy/scipy are **NOT** in base `dependencies`. They live only in the leaf-numeric
  consumer extras (`finance`, `embeddings`, `ann`) that do their own array work, plus the
  explicit `numeric-fallback` extra above.
- The numpy fallback code in the shim is **deliberately kept** — dropping it would break
  every environment that does not ship the kernel wheel.
- Parity is enforced: `tests/test_numeric_parity.py` asserts `np.allclose(kernel, numpy)`
  on randomized inputs (incl. nan/inf/singular matrices), and the engine's `numeric-parity`
  CI job (engine CONCEPT:EG-346) gates the kernel against numpy so the two can never diverge.

### Making the shim kernel-live locally

The wheel is built from the epistemic-graph repo:

```bash
# in the epistemic-graph checkout
maturin build --release -m crates/eg-numeric/Cargo.toml --features python
# on Python > 3.13 also export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
pip install target/wheels/eg_numeric-*.whl   # or the numeric-featured epistemic-graph wheel
python -c "from agent_utilities.numeric import HAVE_KERNEL; print(HAVE_KERNEL)"  # -> True
```

With the kernel installed, `test_numeric_parity.py` exercises the KERNEL path (not just the
fallback). Without it, the same test validates the numpy fallback path.

## Honest status and the ONE remaining step to fully drop numpy

What P5 delivered:
- Kernel is the declared **primary** backend (`numeric-kernel` extra, kernel-first discovery).
- numpy/scipy are **demoted** to an explicit `numeric-fallback` extra (out of base deps).
- Both paths are **parity-gated** (`test_numeric_parity.py` + engine `numeric-parity` CI).

What P5 did **not** do (and why): numpy is **not** removed. The `eg-numeric` Surface-A wheel
is currently folded into the `epistemic-graph` engine wheel behind a maturin `numeric`
feature and is **not yet published to a package index**. Until it is, agent-utilities cannot
hard-depend on `pip install eg-numeric` in every environment, so the numpy fallback must
stay.

**Remaining step:** publish the `eg-numeric` wheel (or the numeric-featured `epistemic-graph`
wheel) to a resolvable package index. Once it is a hard, index-resolvable dependency, the
`numeric-fallback` extra and the shim's numpy fallback branch can be retired and numpy
dropped entirely — a mechanical change, since every call site already uses `xp`.
