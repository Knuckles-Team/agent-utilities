# Native numeric kernel contract

`agent_utilities.numeric` exposes `xp` as a deliberately thin adapter over the
certified `epistemic_graph.numeric` extension:

```python
from agent_utilities.numeric import xp
value = xp.sum([1.0, 2.0])
```

The extension owns validation, shape semantics, arithmetic, and computation.
The adapter only converts scalar/list boundary values, invokes an allowlisted
native function, and preserves scalar/list/tuple results. It has no array
class, broadcasting implementation, indexing engine, or alternate numeric
backend. If the extension is absent, importing `xp` raises `ImportError`; if
an operation is not in the native contract, `UnsupportedNumericOperationError`
is raised.

## Native surface

The current AU contract is the following allowlist, matching the corresponding
exports in `epistemic_graph.numeric`:

- reductions/statistics: `sum`, `prod`, `mean`, `var`, `std`, `amin`, `amax`,
  `argmin`, `argmax`, `argsort`, `cumsum`, `cumprod`, `percentile`, `quantile`;
- element-wise: `sqrt`, `log`, `exp`, `absolute`/`abs`, `tanh`, `clip`,
  `nan_to_num`, `isnan`, `maximum`, `minimum`, `where`;
- vector/matrix: `norm`, `norm_ord`, `dot`, `matmul`, `solve`, `svdvals`,
  `svd`, `eigh`, `eigsh`, `pinv`, `lstsq`, `qr`, `cholesky`, `det`, `inv`,
  `matrix_power`;
- statistics/distributions: `spearmanr`, `ks_2samp`, `norm_ppf`, `norm_pdf`;
- native seeded operations: `normal`, `uniform`, `integers`, `choice_indices`,
  `permutation_indices`, and `kmeans`.

`xp.linalg` is an allowlisted view of the vector/matrix operations. It does not
expose arbitrary extension attributes. `xp.random.default_rng(seed)` is a thin,
deterministic adapter over the engine's seeded one-shot operations; it owns only
a seed and draw counter and returns builtin lists. It is not a general array or
mutable RNG implementation. In particular, `xp.array`, constructors, ufunc
methods, and dtype/constant objects are not silently emulated.

`xp.random.Generator.choice` and `shuffle` delegate selection and permutation to
the native batch index operations. The AU boundary performs only bounded
probability validation and maps returned indices back to caller values; it does
not sample in Python, use modulo selection, or invoke a per-element native call.

All boundary trees are bounded before recursive conversion: rank is at most 8,
leaf elements at most 1,000,000, and container nodes at most 2,000,000. Random
shape requests use the same rank and element-product limits, including an
explicit zero-size result.

## AU consumer seams and guardrails

The NE-153 consumer audit is complete for the bounded AU call sites. Callers
now use explicit builtin-list loops plus the native scalar/vector operations:

- dataframe producers cross the boundary with `list(series)` or Arrow
  `to_pylist()`; no pandas/polars object is passed to `xp`;
- object-dtype and capability-index vectors cross a versioned JSON artifact
  seam through `save_numeric_artifact`/`load_numeric_artifact`, with bounded
  size, rank/elements, schema, digest, and symlink checks;
- stateful random call sites use the deterministic adapter and fixed seeds;
- unsupported constructors, arbitrary attributes, and unbounded array behavior
  remain typed failures rather than Python-list emulation.

The adapter still does not implement these general-array surfaces:

- bounded containers and constructors (`array`/`asarray`, `zeros`, `ones`,
  `empty`, `full`, `arange`, `eye`, `diag`, `fill_diagonal`, `diff`,
  `concatenate`, `stack`, `vstack`, `reshape`, `sort`);
- a native container protocol for shape/size/indexing, slicing, arithmetic,
  broadcasting, or transpose semantics;
- `cov`, `corrcoef`, `save`, `load`, `roll`, `triu_indices`, `log2`, `any`,
  and `allclose` as implicit compatibility surfaces;
- dataframe/object-array coercion, dtype constants, or arbitrary extension
  attributes.

This is a guardrail, not permission to grow a Python numeric runtime in AU.
Future callers must land an engine contract or an explicit bounded serialization
seam before entering this facade.

### Numeric artifact envelope

Artifacts are regular UTF-8 JSON files no larger than 64 MiB. The exact
top-level schema is:

```json
{
  "schema": "eg-numeric-list-v1",
  "values": [[1.0, 2.0]],
  "digest": "<64 lowercase hexadecimal SHA-256 characters>"
}
```

`digest` is SHA-256 over the canonical JSON encoding (UTF-8,
`ensure_ascii=false`, sorted keys, compact separators, and `allow_nan=false`)
of the `schema` and `values` members only. Loading requires the exact three
keys, revalidates the rank/elements/node bounds, and verifies the digest before
returning values. Save/load use descriptor-relative no-follow operations,
reject symlinks and non-regular files, and save via a private fsynced temporary
file followed by an atomic replacement.

## Dependency contract

The AU base metadata, numeric acceptance environment, and default test/guardrail
dependency groups do not declare an external array runtime. The reconciled EG
package metadata also declares no NumPy dependency. Numerical parity is owned
and exercised in the producer repository at
`crates/eg-numeric/tests/test_kernel_parity.py`; AU does not install a second
array runtime merely to retest the producer's implementation. Finance/dataframe
integrations remain behind the explicit `finance` extra; any transitive array
dependency there is isolated to that optional domain profile and is not a
numeric-kernel dependency.

## Verification

Focused contract tests exercise native calls, scalar/list conversion,
deterministic random draws, artifact round-trips, and fail-closed unsupported
operations in `tests/unit/test_numeric_facade.py`.
The production module contains neither an `import numpy` path nor a
`sys.modules` lookup. The focused test command is:

```bash
python -m pytest -q tests/unit/test_numeric_facade.py
```
