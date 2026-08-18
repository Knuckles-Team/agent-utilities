# Design: native numeric-kernel boundary

CONCEPT:AU-KG.compute.numpy-scipy-drop

## Decision

`agent_utilities.numeric.xp` is a thin, explicit adapter over the certified
`epistemic_graph.numeric` extension. The engine is the only numeric authority.
AU performs only scalar/list conversion and direct calls to an allowlisted
native method. It does not define an array class or reproduce broadcasting,
arithmetic, reductions, indexing, random, linalg, or persistence in Python.

Absence is loud: a missing kernel raises `ImportError`, and an operation that is
not in the allowlist raises `UnsupportedNumericOperationError`. The adapter
never discovers private modules or dispatches to a kernel-owned implementation
table.

## Boundary and ownership

1. A caller supplies scalars or bounded builtin list/tuple trees.
2. The adapter recursively validates/converts those values.
3. The native extension validates shape/size and performs computation.
4. Native scalar/list/tuple results cross back unchanged in kind.

Container semantics, numerical algorithms, error behavior, and resource bounds
are engine responsibilities. Arrow/list conversion may be added only when an
existing supported producer boundary requires it; it is not a license to add a
Python array runtime.

## Missing native primitives

The AU production audit currently identifies these engine-level gaps:

- constructors and container operations: `array`, `asarray`, `zeros`, `ones`,
  `empty`, `full`, `arange`, `eye`, `diag`, `fill_diagonal`, `diff`,
  `concatenate`, `stack`, `vstack`, `reshape`, `sort`;
- shape/size/indexing/slicing, arithmetic/broadcasting, transpose, and
  axis-aware reductions needed by existing callers;
- `cov`, `corrcoef`, `roll`, `triu_indices`, `log2`, `any`, and `all`;
- legacy linalg result/error compatibility, notably the `lstsq` tuple contract.

These are reported for engine or call-site design. They are intentionally not
implemented in AU. A bounded stateful seeded generator (`xp.random.default_rng`/
`RandomState`, delegating every draw to native `normal`/`uniform`/`integers`/
`choice_indices`/`permutation_indices` calls and keeping only a seed/draw
counter in Python) and a bounded artifact seam (`save_numeric_artifact` /
`load_numeric_artifact`, deliberately not full `numpy.save`/`.load`
compatibility) are already implemented in `agent_utilities/numeric/__init__.py`.

## Dependency and parity policy

The AU base, numeric acceptance, and default test/guardrail profiles contain no
direct external array dependency. The current lock still carries the legacy
`epistemic-graph` package edge; the EG NumPy-retirement change must land before
the final AU lock regeneration removes that transitive native-profile edge. The
parity oracle is an explicit developer test selected by
`AU_ENABLE_NUMERIC_PARITY=1`; it is not project metadata and cannot become a
runtime fallback. Optional finance/dataframe dependencies keep their own
isolated profile and do not define the native numeric contract.

## Verification

`tests/unit/test_numeric_facade.py` proves direct native dispatch, boundary
conversion, and fail-closed unsupported attributes. `tests/unit/test_numeric_parity.py`
is an opt-in external comparison corpus and contains no fallback assertions.
