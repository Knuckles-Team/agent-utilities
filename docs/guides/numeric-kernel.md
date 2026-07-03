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
| `numeric-kernel` | `eg-numeric>=0.1.0`, `epistemic-graph>=2.6.0` (both **loose floors**) | **Primary** numeric backend. Installing it pulls the published `eg-numeric` kernel wheel as a HARD dependency, so the shim is kernel-LIVE (`HAVE_KERNEL == True`). |
| `numeric-fallback` | `numpy>=2.4.6`, `scipy>=1.17.1` | **Fallback** path. Kept so the abstraction works wherever the kernel wheel is absent (e.g. a minimal env with no engine). |

- numpy/scipy are **NOT** in base `dependencies` and are **not** a primary dep of any kind.
  They live only in the leaf-numeric consumer extras (`finance`, `embeddings`, `ann`) that do
  genuinely numpy/scipy-specific array work, plus the explicit `numeric-fallback` extra above.
- The numpy fallback code in the shim is **deliberately kept** — dropping it would break
  every environment that does not ship the kernel wheel.
- All floors are **loose** (`>=`), never exact pins: published deps express minimums; dev
  overlays live source (see the editable dev path below).
- Parity is enforced: `tests/test_numeric_parity.py` asserts `np.allclose(kernel, numpy)`
  on randomized inputs (incl. nan/inf/singular matrices), and the engine's `numeric-parity`
  CI job (engine CONCEPT:EG-346) gates the kernel against numpy so the two can never diverge.

## Dev vs prod: two different install paths

**Prod / published installs** pull the kernel as a resolvable wheel:

```bash
pip install 'agent-utilities[numeric-kernel]'   # -> eg-numeric>=0.1.0 + epistemic-graph>=2.6.0
python -c "from agent_utilities.numeric import xp; print(xp.HAVE_KERNEL)"  # -> True
```

The `eg-numeric` Surface-A wheel is now **published** (cp39-abi3), shipped alongside the
`epistemic-graph` 2.6.0 engine release, so `numeric-kernel` resolves it directly.

**Dev is editable and non-publishing.** The fleet dev deploy already source-mounts the repos
(`PYTHONPATH=/au:/eg`, `services/graph-os/compose.dev.yml`), so dev builds the kernel **from
source**, NOT from the published wheel. Install an editable `eg-numeric` into your venv with:

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 \
  maturin develop \
  -m /home/apps/workspace/agent-packages/epistemic-graph/crates/eg-numeric/Cargo.toml \
  --features python
python -c "from agent_utilities.numeric import xp; print(xp.HAVE_KERNEL)"  # -> True
```

`maturin develop` compiles the pyo3 extension and installs it editable into the active venv
as `epistemic_graph.numeric` — so the shim goes kernel-LIVE against your **local source**.
Dev never depends on any published artifact. (Prefer `maturin develop` for dev;
`maturin build --release … && pip install target/wheels/eg_numeric-*.whl` produces the same
wheel prod ships if you want to test the packaged artifact.)

With the kernel installed either way, `test_numeric_parity.py` exercises the KERNEL path (not
just the fallback). Uninstall it (`pip uninstall eg-numeric`) and the same test validates the
numpy fallback path.

## Honest status — the full numpy drop (CONCEPT:KG-2.319)

What the drop delivered:
- Kernel is the declared **primary** backend and now a **hard dependency**: `numeric-kernel`
  declares `eg-numeric>=0.1.0` directly (loose floor) — the wheel is published, so this is
  index-resolvable, not just an engine carrier.
- numpy is **no longer a base/primary dependency** of any kind. It survives only as the
  `numeric-fallback` extra and inside the leaf `finance`/`embeddings`/`ann` extras.
- Both paths remain **parity-gated** (`test_numeric_parity.py` + engine `numeric-parity` CI).

Honest caveat — numpy is **not** 100% gone, and that is correct:
- The `xp` shim keeps its numpy fallback branch on purpose, so kernel-absent (minimal)
  environments still work.
- Some leaf extras legitimately need numpy/scipy for ops with **no kernel equivalent** —
  `scipy.stats` (`norm`/`spearmanr`/`ks_2samp`) in `finance`, and `scipy.sparse.linalg.eigsh`
  in the spectral navigator (which degrades to the `xp`/numpy dense path when scipy is
  absent). Those stay declared only in the specific extras that use them.

The "drop" is complete in the sense that matters: numpy is no longer a base or primary
dependency — only a fallback / leaf-extra concern.
