"""``xp`` — a numpy-compatible numeric namespace backed SOLELY by the epistemic-graph kernel.

CONCEPT:AU-KG.compute.surface-analytics-program — Surface A of the Analytics Program's "one kernel, two surfaces"
(engine side: CONCEPT:AU-KG.compute.numeric-kernel). ``xp`` mirrors the subset of the numpy API that
agent-utilities actually uses (the 598-site audit: reductions/stats, element-wise,
the linalg-6 + ``LinAlgError``, random, the four scipy ops) and routes those ops
through the compiled ``epistemic_graph.numeric`` kernel (pure-Rust faer + ndarray,
BLAS/LAPACK-free).

CONCEPT:AU-KG.compute.numpy-scipy-drop — **the hard numpy/scipy drop (Analytics Program P5 final).** numpy
and scipy are **fully removed** from agent-utilities: this package **imports numpy
nowhere and declares it in no dependency** (``requirements.txt`` / ``pyproject.toml``
carry neither). The compiled ``epistemic_graph.numeric`` kernel is the **sole numeric
backend** and is a **hard requirement** — importing this module **raises ``ImportError``
when the kernel is absent**; there is NO numpy fallback for a missing kernel (the old
``HAVE_KERNEL``-false path and the ``import numpy as _np`` binding are gone).

How numpy still functions **without agent-utilities depending on it**: the kernel is a
**rust-numpy container** — numpy lives INSIDE ``epistemic-graph[numeric]`` as the
kernel's own zero-copy interop dependency, and the compiled module re-exports numpy's
array primitives (``ndarray``, the dtypes, ``newaxis`` / ``pi`` / ``inf`` / ``nan``).
The kernel's compiled fast-path is deliberately **narrow** — contiguous 1-D/2-D
``float64`` for the element-wise + linalg ops; general (lists / N-D / ``axis`` /
``keepdims`` / int) for the reductions/stats. For inputs OUTSIDE that compiled
fast-path (N-D element-wise, ``axis`` norms, the ``random`` Generator API,
``cov`` / ``corrcoef`` / ``save`` / ``load`` / …) the shim delegates to the numpy
module **the kernel itself already loaded** — obtained from the kernel
(``sys.modules[_KERNEL.ndarray.__module__]``), never via an ``import numpy`` statement
in this package and never as an agent-utilities dependency. So numpy is an
**internal implementation detail of the kernel**, exactly as the model states, and
agent-utilities' whole numeric surface flows through this one module.

Migration (executed) is mechanical — the old ``numpy as np`` import line becomes::

    from agent_utilities.numeric import xp as np    # kernel-backed drop-in for `np`

CONCEPT:AU-KG.compute.executed-p2-p3-rollout — the executed P2/P3 rollout: the agent-utilities numpy call sites were
swapped to this ``xp`` surface, keeping the ``np`` alias so bodies are unchanged.

CONCEPT:AU-KG.compute.ufunc-method-surface — the ufunc-method surface. ``xp.maximum`` / ``xp.minimum`` are small
``_Ufunc`` wrapper objects: calling them keeps the kernel-routed element-wise behaviour,
while ``.accumulate`` / ``.reduce`` / ``.outer`` / ``.at`` forward to the kernel's cumulative
op (``cummax`` / ``cummin``) on a bare 1-D float64 input, else to the (kernel-internal)
numpy ufunc method.

The four scipy ops used by ``domains/finance`` + ``spectral_navigator`` are now native
kernel exports (CONCEPT:EG-KG.compute.concept-5): ``xp.eigsh`` (``scipy.sparse.linalg.eigsh``, k
smallest-magnitude symmetric eigenpairs), ``xp.spearmanr`` (``scipy.stats.spearmanr``),
``xp.ks_2samp`` (``scipy.stats.ks_2samp``), ``xp.norm_ppf`` / ``xp.norm_pdf``
(``scipy.stats.norm.ppf`` / ``.pdf``). No scipy import remains anywhere in agent-utilities.

Routing is conservative: a kernel fast-path op is used ONLY when the input matches the
kernel's supported domain (contiguous 1-D/2-D float64, no ``axis``/``keepdims`` where the
kernel op lacks it); otherwise the kernel's internal numpy handles it. The kernel is
parity-proven ``np.allclose`` vs numpy (``tests/test_numeric_parity.py``), so both paths
agree. See ``docs/guides/numeric-kernel.md``.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, TypeAlias

#: Type-annotation aliases for the ``xp`` surface. Call sites migrated from
#: ``import numpy as np`` to ``from agent_utilities.numeric import xp as np`` keep the
#: ``np.ndarray`` / ``np.random.Generator`` *values* working (``xp.__getattr__`` resolves
#: them to the kernel-internal numpy classes at runtime), but ``np.ndarray`` is no longer
#: valid in a TYPE annotation: ``np`` is now an ``_XP`` instance, not the numpy module, so
#: mypy cannot resolve an attribute chain on it as a type. These aliases give annotations a
#: real, statically-resolvable name without importing numpy (kept as ``Any`` — the array's
#: concrete shape/dtype isn't tracked pre-migration either) — import alongside ``xp``:
#: ``from agent_utilities.numeric import xp as np, NDArray``.
NDArray: TypeAlias = Any
RandomGenerator: TypeAlias = Any

# ---------------------------------------------------------------------------
# Kernel discovery — REQUIRED. Prefer the engine-shipped ``epistemic_graph.numeric``;
# also accept a standalone ``numeric`` build (the editable dev / parity wheel). If
# neither is importable, raise — there is NO numpy fallback for a missing kernel.
# ---------------------------------------------------------------------------
_KERNEL: Any = None
KERNEL_SOURCE: str | None = None
for _modpath in ("epistemic_graph.numeric", "numeric"):
    try:
        _mod = importlib.import_module(_modpath)
    except Exception:
        continue
    if getattr(_mod, "__kernel__", None) == "eg-numeric":
        _KERNEL = _mod
        KERNEL_SOURCE = _modpath
        break

if _KERNEL is None:
    raise ImportError(
        "epistemic-graph kernel required: pip install epistemic-graph[numeric]>=2.7.0"
    )

#: The kernel is always live once this module imports (import raises otherwise). Kept as a
#: public boolean so callers can still introspect ``xp.HAVE_KERNEL`` / ``xp.KERNEL_SOURCE``.
HAVE_KERNEL: bool = True

# numpy is an INTERNAL dependency of the kernel (rust-numpy) — the kernel already imported
# it, and re-exports ``numpy.ndarray``. Grab that module here so the shim can serve the
# long-tail array ops the compiled kernel does not expose natively (random Generator API,
# ``cov`` / ``corrcoef`` / ``save`` / ``load`` / ``atleast_2d`` / ``roll`` / ``triu_indices`` /
# ``allclose`` / axis norms / N-D element-wise). This is NOT an ``import numpy`` and NOT an
# agent-utilities dependency — it is the numpy that shipped inside ``epistemic-graph[numeric]``.
_knp: Any = sys.modules[_KERNEL.ndarray.__module__]

#: ``numpy.linalg.LinAlgError``-compatible exception. The kernel raises its own
#: ``LinAlgError`` (a distinct type); we expose the (kernel-internal) numpy one so existing
#: ``except np.linalg.LinAlgError`` handlers keep working, and normalize kernel errors to it.
LinAlgError = _knp.linalg.LinAlgError
_KERNEL_LINALG_ERROR = getattr(_KERNEL, "LinAlgError", ())


def _kernel_eligible(x: Any) -> bool:
    """Whether *x* may take the compiled kernel fast path.

    ONLY raw numpy ``ndarray`` / ``list`` / ``tuple`` inputs are eligible. Duck-typed
    array wrappers (pandas ``Series`` / ``DataFrame``, xarray, …) are NOT: numpy's ufunc
    protocol preserves such wrappers on element-wise/cumulative ops, but the compiled
    kernel returns a bare ``ndarray`` and would silently strip the wrapper. Deferring
    those to the kernel-internal numpy keeps ``np.log(series)`` a ``Series`` (etc.)."""
    return isinstance(x, _knp.ndarray | list | tuple)


def _f64_1d(x: Any) -> Any:
    """Return a contiguous 1-D float64 view of *x*, or ``None`` if it doesn't fit
    the kernel's compiled domain (so the caller delegates to the kernel-internal numpy)."""
    if not _kernel_eligible(x):
        return None
    a = _knp.asarray(x)
    if a.ndim == 1 and a.dtype == _knp.float64:
        return _knp.ascontiguousarray(a)
    return None


def _f64_2d(x: Any) -> Any:
    if not _kernel_eligible(x):
        return None
    a = _knp.asarray(x)
    if a.ndim == 2 and a.dtype == _knp.float64:
        return _knp.ascontiguousarray(a)
    return None


class _Linalg:
    """The ``xp.linalg`` sub-namespace (kernel fast-path, kernel-internal-numpy tail)."""

    LinAlgError = LinAlgError

    def __getattr__(
        self, name: str
    ) -> Any:  # kernel-internal numpy for everything else
        return getattr(_knp.linalg, name)

    # -- kernel-routed (contiguous float64) --
    def norm(
        self, x: Any, ord: Any = None, axis: Any = None, keepdims: bool = False
    ) -> Any:
        if axis is None and not keepdims:
            a = _f64_1d(x)
            if a is not None:
                if ord is None or ord == 2:
                    return _KERNEL.norm(a)
                if ord in (1, _knp.inf, -_knp.inf):
                    return _KERNEL.norm_ord(a, float(ord))
        return _knp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

    def solve(self, a: Any, b: Any) -> Any:
        am, bv = _f64_2d(a), _f64_1d(b)
        if am is not None and bv is not None:
            try:
                return _KERNEL.solve(am, bv)
            except _KERNEL_LINALG_ERROR as e:  # type: ignore[misc]
                raise LinAlgError(str(e)) from None
        return _knp.linalg.solve(a, b)

    def svd(self, a: Any, full_matrices: bool = True, compute_uv: bool = True) -> Any:
        if full_matrices:
            am = _f64_2d(a)
            if am is not None:
                if not compute_uv:
                    return _KERNEL.svdvals(am)
                return _KERNEL.svd(am)
        return _knp.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)

    def eigh(self, a: Any, UPLO: str = "L") -> Any:
        am = _f64_2d(a)
        if am is not None:
            return _KERNEL.eigh(am)
        return _knp.linalg.eigh(a, UPLO=UPLO)

    def pinv(self, a: Any, *args: Any, **kwargs: Any) -> Any:
        if not args and not kwargs:
            am = _f64_2d(a)
            if am is not None:
                return _KERNEL.pinv(am)
        return _knp.linalg.pinv(a, *args, **kwargs)

    def lstsq(self, a: Any, b: Any, rcond: Any = None) -> Any:
        # numpy returns (x, residuals, rank, s); the kernel returns x only. Only route
        # when the caller can accept the numpy-shaped tuple built here.
        am, bv = _f64_2d(a), _f64_1d(b)
        if am is not None and bv is not None:
            x = _KERNEL.lstsq(am, bv)
            resid = _knp.asarray([], dtype=_knp.float64)
            rank = int(_knp.linalg.matrix_rank(am))
            s = _KERNEL.svdvals(am)
            return x, resid, rank, s
        return _knp.linalg.lstsq(a, b, rcond=rcond)

    def qr(self, a: Any, mode: str = "reduced") -> Any:
        if mode == "reduced":
            am = _f64_2d(a)
            if am is not None:
                return _KERNEL.qr(am)
        return _knp.linalg.qr(a, mode=mode)

    def cholesky(self, a: Any, *args: Any, **kwargs: Any) -> Any:
        if not args and not kwargs:
            am = _f64_2d(a)
            if am is not None:
                try:
                    return _KERNEL.cholesky(am)
                except _KERNEL_LINALG_ERROR as e:  # type: ignore[misc]
                    raise LinAlgError(str(e)) from None
        return _knp.linalg.cholesky(a, *args, **kwargs)

    def det(self, a: Any) -> Any:
        am = _f64_2d(a)
        if am is not None:
            return _KERNEL.det(am)
        return _knp.linalg.det(a)

    def inv(self, a: Any) -> Any:
        am = _f64_2d(a)
        if am is not None:
            try:
                return _KERNEL.inv(am)
            except _KERNEL_LINALG_ERROR as e:  # type: ignore[misc]
                raise LinAlgError(str(e)) from None
        return _knp.linalg.inv(a)

    def matrix_power(self, a: Any, n: int) -> Any:
        am = _f64_2d(a)
        if am is not None:
            return _KERNEL.matrix_power(am, int(n))
        return _knp.linalg.matrix_power(a, n)


def _maximum_call(a: Any, b: Any, **kw: Any) -> Any:
    if not kw:
        va, vb = _f64_1d(a), _f64_1d(b)
        if va is not None and vb is not None and va.shape == vb.shape:
            return _KERNEL.maximum(va, vb)
    return _knp.maximum(a, b, **kw)


def _minimum_call(a: Any, b: Any, **kw: Any) -> Any:
    if not kw:
        va, vb = _f64_1d(a), _f64_1d(b)
        if va is not None and vb is not None and va.shape == vb.shape:
            return _KERNEL.minimum(va, vb)
    return _knp.minimum(a, b, **kw)


class _Ufunc:
    """A callable mirroring a numpy ufunc's method surface (CONCEPT:AU-KG.compute.ufunc-method-surface).

    Calling routes through *call* (kernel-accelerated on the fast path). ``.accumulate``
    is kernel-routed (``cummax`` / ``cummin``) on a bare 1-D float64 input; the other
    ufunc methods forward to the kernel-internal numpy ufunc.
    """

    __slots__ = ("_name", "_npufunc", "_call", "_kernel_accum")

    def __init__(self, name: str, call: Any, kernel_accum: str | None = None) -> None:
        self._name = name
        self._npufunc = getattr(_knp, name)
        self._call = call
        self._kernel_accum = kernel_accum

    def __call__(self, *args: Any, **kw: Any) -> Any:
        return self._call(*args, **kw)

    def accumulate(
        self, array: Any, axis: int = 0, dtype: Any = None, out: Any = None
    ) -> Any:
        if (
            self._kernel_accum is not None
            and axis in (0, -1)
            and dtype is None
            and out is None
        ):
            kfn = getattr(_KERNEL, self._kernel_accum, None)
            if kfn is not None:
                v = _f64_1d(array)
                if v is not None:
                    return kfn(v)
        return self._npufunc.accumulate(array, axis=axis, dtype=dtype, out=out)

    def reduce(self, *args: Any, **kw: Any) -> Any:
        return self._npufunc.reduce(*args, **kw)

    def outer(self, a: Any, b: Any, **kw: Any) -> Any:
        return self._npufunc.outer(a, b, **kw)

    def at(self, a: Any, indices: Any, b: Any = None) -> Any:
        if b is None:
            return self._npufunc.at(a, indices)
        return self._npufunc.at(a, indices, b)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<xp ufunc {self._name!r}>"


class _XP:
    """The ``xp`` namespace: kernel-native fast path, kernel-internal-numpy long tail."""

    linalg = _Linalg()
    LinAlgError = LinAlgError
    HAVE_KERNEL = HAVE_KERNEL
    KERNEL_SOURCE = KERNEL_SOURCE

    def __getattr__(self, name: str) -> Any:
        # Prefer a native kernel export (constructors, dtypes, constants, ``ndarray``,
        # seeded ``normal`` / ``uniform`` / ``integers``, the four scipy ops, …); fall to
        # the kernel-internal numpy for the long tail (``random``, ``cov``, ``corrcoef``,
        # ``save`` / ``load``, ``log2`` / ``any`` / ``atleast_2d`` / ``roll`` /
        # ``triu_indices`` / ``allclose`` / ``where`` broadcast forms, …).
        try:
            return getattr(_KERNEL, name)
        except AttributeError:
            return getattr(_knp, name)

    # ---- reductions / stats ----
    # The kernel handles eligible (ndarray / list / tuple) inputs incl. N-D + axis +
    # keepdims; non-eligible array wrappers (pandas Series / DataFrame) defer to the
    # kernel-internal numpy so axis semantics + any wrapper are preserved exactly.
    def _reduce(self, name: str, a: Any, axis: Any, kw: Any) -> Any:
        if axis is None and not kw:
            v = _f64_1d(a)
            if v is not None:
                return getattr(_KERNEL, name)(v)
        if _kernel_eligible(a):
            return getattr(_KERNEL, name)(a, axis=axis, **kw)
        return getattr(_knp, name)(a, axis=axis, **kw)

    def sum(self, a: Any, axis: Any = None, **kw: Any) -> Any:
        return self._reduce("sum", a, axis, kw)

    def prod(self, a: Any, axis: Any = None, **kw: Any) -> Any:
        return self._reduce("prod", a, axis, kw)

    def mean(self, a: Any, axis: Any = None, **kw: Any) -> Any:
        return self._reduce("mean", a, axis, kw)

    def std(self, a: Any, axis: Any = None, ddof: int = 0, **kw: Any) -> Any:
        # CRITICAL: the kernel signature is ``std(a, axis=None, ddof=0, keepdims=False)`` —
        # pass ``ddof`` by KEYWORD (positional would bind to ``axis``).
        if axis is None and not kw:
            v = _f64_1d(a)
            if v is not None:
                return _KERNEL.std(v, ddof=ddof)
        if _kernel_eligible(a):
            return _KERNEL.std(a, axis=axis, ddof=ddof, **kw)
        return _knp.std(a, axis=axis, ddof=ddof, **kw)

    def var(self, a: Any, axis: Any = None, ddof: int = 0, **kw: Any) -> Any:
        # CRITICAL: same as ``std`` — ``ddof`` is keyword, not positional.
        if axis is None and not kw:
            v = _f64_1d(a)
            if v is not None:
                return _KERNEL.var(v, ddof=ddof)
        if _kernel_eligible(a):
            return _KERNEL.var(a, axis=axis, ddof=ddof, **kw)
        return _knp.var(a, axis=axis, ddof=ddof, **kw)

    def min(self, a: Any, axis: Any = None, **kw: Any) -> Any:
        if axis is None and not kw:
            v = _f64_1d(a)
            if v is not None and v.size:
                return _KERNEL.amin(v)
        if _kernel_eligible(a):
            return _KERNEL.amin(a, axis=axis, **kw)
        return _knp.amin(a, axis=axis, **kw)

    amin = min

    def max(self, a: Any, axis: Any = None, **kw: Any) -> Any:
        if axis is None and not kw:
            v = _f64_1d(a)
            if v is not None and v.size:
                return _KERNEL.amax(v)
        if _kernel_eligible(a):
            return _KERNEL.amax(a, axis=axis, **kw)
        return _knp.amax(a, axis=axis, **kw)

    amax = max

    def argmin(self, a: Any, axis: Any = None, **kw: Any) -> Any:
        if axis is None and not kw:
            v = _f64_1d(a)
            if v is not None and v.size:
                return _KERNEL.argmin(v)
        if _kernel_eligible(a):
            return _KERNEL.argmin(a, axis=axis, **kw)
        return _knp.argmin(a, axis=axis, **kw)

    def argmax(self, a: Any, axis: Any = None, **kw: Any) -> Any:
        if axis is None and not kw:
            v = _f64_1d(a)
            if v is not None and v.size:
                return _KERNEL.argmax(v)
        if _kernel_eligible(a):
            return _KERNEL.argmax(a, axis=axis, **kw)
        return _knp.argmax(a, axis=axis, **kw)

    def argsort(self, a: Any, axis: int = -1, kind: Any = None, **kw: Any) -> Any:
        if axis in (-1, 0) and not kw:
            v = _f64_1d(a)
            if v is not None:
                return _KERNEL.argsort(v)
        return _knp.argsort(a, axis=axis, kind=kind, **kw)

    def cumsum(self, a: Any, axis: Any = None, **kw: Any) -> Any:
        if axis is None and not kw:
            v = _f64_1d(a)
            if v is not None:
                return _KERNEL.cumsum(v)
        return _knp.cumsum(a, axis=axis, **kw)

    def cumprod(self, a: Any, axis: Any = None, **kw: Any) -> Any:
        if axis is None and not kw:
            v = _f64_1d(a)
            if v is not None:
                return _KERNEL.cumprod(v)
        return _knp.cumprod(a, axis=axis, **kw)

    def percentile(self, a: Any, q: Any, axis: Any = None, **kw: Any) -> Any:
        if axis is None and not kw and _knp.isscalar(q):
            v = _f64_1d(a)
            if v is not None and v.size:
                return _KERNEL.percentile(v, float(q))
        return _knp.percentile(a, q, axis=axis, **kw)

    def quantile(self, a: Any, q: Any, axis: Any = None, **kw: Any) -> Any:
        if axis is None and not kw and _knp.isscalar(q):
            v = _f64_1d(a)
            if v is not None and v.size:
                return _KERNEL.quantile(v, float(q))
        return _knp.quantile(a, q, axis=axis, **kw)

    # ---- element-wise (kernel fast path is contiguous 1-D float64) ----
    def _ew1(self, name: str, a: Any, **kw: Any) -> Any:
        if not kw:
            v = _f64_1d(a)
            if v is not None:
                return getattr(_KERNEL, name)(v)
        return getattr(_knp, name)(a, **kw)

    def sqrt(self, a: Any, **kw: Any) -> Any:
        return self._ew1("sqrt", a, **kw)

    def log(self, a: Any, **kw: Any) -> Any:
        return self._ew1("log", a, **kw)

    def exp(self, a: Any, **kw: Any) -> Any:
        return self._ew1("exp", a, **kw)

    def abs(self, a: Any, **kw: Any) -> Any:
        if not kw:
            v = _f64_1d(a)
            if v is not None:
                return _KERNEL.absolute(v)
        return _knp.absolute(a, **kw)

    absolute = abs

    def tanh(self, a: Any, **kw: Any) -> Any:
        return self._ew1("tanh", a, **kw)

    def clip(self, a: Any, a_min: Any = None, a_max: Any = None, **kw: Any) -> Any:
        if (
            not kw
            and a_min is not None
            and a_max is not None
            and _knp.isscalar(a_min)
            and _knp.isscalar(a_max)
        ):
            v = _f64_1d(a)
            if v is not None:
                return _KERNEL.clip(v, float(a_min), float(a_max))
        return _knp.clip(a, a_min, a_max, **kw)

    def nan_to_num(
        self,
        a: Any,
        copy: bool = True,
        nan: float = 0.0,
        posinf: Any = None,
        neginf: Any = None,
    ) -> Any:
        if _knp.isscalar(nan):
            v = _f64_1d(a)
            if v is not None:
                pi = (
                    float(posinf)
                    if posinf is not None
                    else _knp.finfo(_knp.float64).max
                )
                ni = (
                    float(neginf)
                    if neginf is not None
                    else _knp.finfo(_knp.float64).min
                )
                return _KERNEL.nan_to_num(v, float(nan), pi, ni)
        return _knp.nan_to_num(a, copy=copy, nan=nan, posinf=posinf, neginf=neginf)

    def isnan(self, a: Any, **kw: Any) -> Any:
        if not kw:
            v = _f64_1d(a)
            if v is not None:
                return _knp.asarray(_KERNEL.isnan(v), dtype=bool)
        return _knp.isnan(a, **kw)

    #: ufunc-method surface (CONCEPT:AU-KG.compute.ufunc-method-surface).
    maximum = _Ufunc("maximum", _maximum_call, kernel_accum="cummax")
    minimum = _Ufunc("minimum", _minimum_call, kernel_accum="cummin")

    def where(self, condition: Any, *args: Any) -> Any:
        if len(args) == 2:
            cond = _knp.asarray(condition)
            va, vb = _f64_1d(args[0]), _f64_1d(args[1])
            if (
                cond.ndim == 1
                and va is not None
                and vb is not None
                and cond.shape == va.shape == vb.shape
            ):
                return _KERNEL.where_(cond.astype(bool).tolist(), va, vb)
        return _knp.where(condition, *args)

    def dot(self, a: Any, b: Any, **kw: Any) -> Any:
        if not kw:
            va, vb = _f64_1d(a), _f64_1d(b)
            if va is not None and vb is not None and va.shape == vb.shape:
                return _KERNEL.dot(va, vb)
            ma, mb = _f64_2d(a), _f64_2d(b)
            if ma is not None and mb is not None and ma.shape[1] == mb.shape[0]:
                return _KERNEL.matmul(ma, mb)
        return _knp.dot(a, b, **kw)

    def matmul(self, a: Any, b: Any, **kw: Any) -> Any:
        if not kw:
            ma, mb = _f64_2d(a), _f64_2d(b)
            if ma is not None and mb is not None and ma.shape[1] == mb.shape[0]:
                return _KERNEL.matmul(ma, mb)
        return _knp.matmul(a, b, **kw)

    # ---- scipy ops, now native kernel exports (CONCEPT:EG-KG.compute.concept-5 / KG-2.324) ----
    def eigsh(self, a: Any, k: int, which: str = "SM") -> Any:
        """``scipy.sparse.linalg.eigsh(A, k, which="SM")`` — the ``k`` smallest-magnitude
        symmetric eigenpairs. The kernel op is always smallest-magnitude; ``which`` is
        accepted for call-site compatibility and must be ``"SM"``."""
        if which != "SM":
            raise ValueError(f"xp.eigsh only supports which='SM' (got {which!r})")
        return _KERNEL.eigsh(
            _knp.ascontiguousarray(_knp.asarray(a, dtype=_knp.float64)), int(k)
        )

    def spearmanr(self, a: Any, b: Any) -> Any:
        """``scipy.stats.spearmanr(a, b)`` → ``(correlation, pvalue)``."""
        return _KERNEL.spearmanr(
            _knp.asarray(a, dtype=_knp.float64), _knp.asarray(b, dtype=_knp.float64)
        )

    def ks_2samp(self, a: Any, b: Any) -> Any:
        """``scipy.stats.ks_2samp(a, b)`` → ``(statistic, pvalue)``."""
        return _KERNEL.ks_2samp(
            _knp.asarray(a, dtype=_knp.float64), _knp.asarray(b, dtype=_knp.float64)
        )

    def norm_ppf(self, q: Any) -> Any:
        """``scipy.stats.norm.ppf(q)`` — standard-normal inverse CDF."""
        return _KERNEL.norm_ppf(float(q))

    def norm_pdf(self, x: Any) -> Any:
        """``scipy.stats.norm.pdf(x)`` — standard-normal PDF."""
        return _KERNEL.norm_pdf(float(x))


#: The public namespace. Import as ``from agent_utilities.numeric import xp as np``.
xp = _XP()

__all__ = [
    "xp",
    "HAVE_KERNEL",
    "KERNEL_SOURCE",
    "LinAlgError",
    "NDArray",
    "RandomGenerator",
]
