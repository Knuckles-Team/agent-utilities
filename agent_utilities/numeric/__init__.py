"""``xp`` — a numpy-compatible numeric namespace backed by the epistemic-graph kernel.

CONCEPT:KG-2.312 — Surface A of the Analytics Program's "one kernel, two surfaces"
(engine side: CONCEPT:EG-321). ``xp`` mirrors the subset of the numpy API that
agent-utilities actually uses (the 598-site audit: reductions/stats, element-wise,
the linalg-6 + ``LinAlgError``, random) and routes those ops through the compiled
``epistemic_graph.numeric`` kernel (pure-Rust faer + ndarray, BLAS/LAPACK-free)
**when it is importable**, transparently **falling back to numpy** when the compiled
module is absent — so nothing breaks before the engine wheel is rebuilt.

Migration (phases P2-P3) is then mechanical::

    import numpy as np              # before
    from agent_utilities.numeric import xp as np   # after

CONCEPT:KG-2.313 — the executed P2/P3 rollout: 34 agent-utilities numpy call
sites (28 light-op files + the 6 linalg files ``optimization_engine``,
``world_model``, ``formal_reasoning_core``, ``spectral_navigator``, finance
``cross_market_arb`` and ``signal_fusion``) were swapped to this ``xp`` surface,
keeping the ``np`` alias so bodies are unchanged. Behaviour is identical today
(kernel absent → numpy fallback) and becomes kernel-accelerated once the
``epistemic_graph.numeric`` wheel is deployed.

CONCEPT:KG-2.314 — the ufunc-method surface. ``xp.maximum`` / ``xp.minimum`` are
not plain callables but small ``_Ufunc`` wrapper objects: calling them keeps the
kernel-routed element-wise behaviour, while ``.accumulate`` / ``.reduce`` /
``.outer`` / ``.at`` forward to numpy's real ufunc methods (and to the kernel
when ``HAVE_KERNEL`` and a matching kernel op exists). This lets the three
finance files (``composite_backtest``, ``profit_attribution``,
``research_autopilot``) that call ``np.maximum.accumulate(...)`` migrate onto the
``xp`` surface — previously ``xp.maximum`` was a bare callable with no
``.accumulate`` attribute, blocking their migration.

CONCEPT:KG-2.315 — the shim goes **kernel-LIVE**. Once the ``eg-numeric`` Surface-A
pyo3 wheel (engine CONCEPT:EG-346) is installed, the kernel-discovery loop below finds
``epistemic_graph.numeric`` / ``numeric`` (``__kernel__ == "eg-numeric"``), so
``HAVE_KERNEL`` flips to ``True`` and every routed op executes the compiled faer/ndarray
kernel instead of numpy — with ZERO code change at the call sites (the ``np`` alias is
unchanged). The wheel is built with
``maturin build --release -m crates/eg-numeric/Cargo.toml --features python`` (on Python
> 3.13 add ``PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1``); ``tests/test_numeric_parity.py``
then exercises the KERNEL path ``np.allclose`` vs numpy — not just the fallback — and the
engine's ``numeric-parity`` CI job gates the kernel against numpy so it can never diverge.

Any attribute not explicitly overridden below is delegated straight to numpy, so
``xp`` is a drop-in for ``import numpy as np`` (``xp.array``, ``xp.zeros``,
``xp.newaxis``, ``xp.float64``, ``xp.linalg.eig`` … all resolve to numpy).

Routing is conservative: a kernel op is used ONLY when the input matches the
kernel's supported domain (contiguous 1-D/2-D float64, no ``axis``/``keepdims``);
otherwise numpy handles it. The kernel is parity-proven ``np.allclose`` vs numpy
(``tests/test_numeric_parity.py``), so both paths agree.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as _np

# ---------------------------------------------------------------------------
# Kernel discovery — prefer the engine-shipped ``epistemic_graph.numeric``; also
# accept a standalone ``numeric`` build (the pre-fold parity/dev wheel). Absent →
# numpy fallback (HAVE_KERNEL = False).
# ---------------------------------------------------------------------------
_kernel: Any = None
KERNEL_SOURCE: str | None = None
for _modpath in ("epistemic_graph.numeric", "numeric"):
    try:
        _mod = importlib.import_module(_modpath)
    except Exception:
        continue
    if getattr(_mod, "__kernel__", None) == "eg-numeric":
        _kernel = _mod
        KERNEL_SOURCE = _modpath
        break

HAVE_KERNEL: bool = _kernel is not None

#: ``numpy.linalg.LinAlgError``-compatible exception. The kernel raises its own
#: ``LinAlgError`` (a distinct type); we expose numpy's so existing
#: ``except np.linalg.LinAlgError`` handlers keep working, and normalize kernel
#: errors to it at the boundary.
LinAlgError = _np.linalg.LinAlgError
_KERNEL_LINALG_ERROR = getattr(_kernel, "LinAlgError", ()) if HAVE_KERNEL else ()


def _f64_1d(x: Any) -> _np.ndarray | None:
    """Return a contiguous 1-D float64 view of *x*, or ``None`` if it doesn't fit
    the kernel's domain (so the caller falls back to numpy)."""
    a = _np.asarray(x)
    if a.ndim == 1 and a.dtype == _np.float64:
        return _np.ascontiguousarray(a)
    return None


def _f64_2d(x: Any) -> _np.ndarray | None:
    a = _np.asarray(x)
    if a.ndim == 2 and a.dtype == _np.float64:
        return _np.ascontiguousarray(a)
    return None


class _Linalg:
    """The ``xp.linalg`` sub-namespace (numpy-delegating, kernel-accelerated)."""

    LinAlgError = LinAlgError

    def __getattr__(self, name: str) -> Any:  # numpy fallback for everything else
        return getattr(_np.linalg, name)

    # -- kernel-routed (2-D float64) --
    def norm(
        self, x: Any, ord: Any = None, axis: Any = None, keepdims: bool = False
    ) -> Any:
        if HAVE_KERNEL and axis is None and not keepdims:
            a = _f64_1d(x)
            if a is not None:
                if ord is None or ord == 2:
                    return _kernel.norm(a)
                if ord in (1, _np.inf, -_np.inf):
                    return _kernel.norm_ord(a, float(ord))
        return _np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

    def solve(self, a: Any, b: Any) -> Any:
        if HAVE_KERNEL:
            am, bv = _f64_2d(a), _f64_1d(b)
            if am is not None and bv is not None:
                try:
                    return _kernel.solve(am, bv)
                except _KERNEL_LINALG_ERROR as e:  # type: ignore[misc]
                    raise LinAlgError(str(e)) from None
        return _np.linalg.solve(a, b)

    def svd(self, a: Any, full_matrices: bool = True, compute_uv: bool = True) -> Any:
        if HAVE_KERNEL and full_matrices:
            am = _f64_2d(a)
            if am is not None:
                if not compute_uv:
                    return _kernel.svdvals(am)
                return _kernel.svd(am)
        return _np.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)

    def eigh(self, a: Any, UPLO: str = "L") -> Any:
        if HAVE_KERNEL:
            am = _f64_2d(a)
            if am is not None:
                return _kernel.eigh(am)
        return _np.linalg.eigh(a, UPLO=UPLO)

    def pinv(self, a: Any, *args: Any, **kwargs: Any) -> Any:
        if HAVE_KERNEL and not args and not kwargs:
            am = _f64_2d(a)
            if am is not None:
                return _kernel.pinv(am)
        return _np.linalg.pinv(a, *args, **kwargs)

    def lstsq(self, a: Any, b: Any, rcond: Any = None) -> Any:
        # numpy returns (x, residuals, rank, s); the kernel returns x only. Only
        # route when the caller can accept the numpy-shaped tuple built here.
        if HAVE_KERNEL:
            am, bv = _f64_2d(a), _f64_1d(b)
            if am is not None and bv is not None:
                x = _kernel.lstsq(am, bv)
                resid = _np.asarray([], dtype=_np.float64)
                rank = int(_np.linalg.matrix_rank(am))
                s = _kernel.svdvals(am)
                return x, resid, rank, s
        return _np.linalg.lstsq(a, b, rcond=rcond)

    def qr(self, a: Any, mode: str = "reduced") -> Any:
        if HAVE_KERNEL and mode == "reduced":
            am = _f64_2d(a)
            if am is not None:
                return _kernel.qr(am)
        return _np.linalg.qr(a, mode=mode)

    def cholesky(self, a: Any, *args: Any, **kwargs: Any) -> Any:
        if HAVE_KERNEL and not args and not kwargs:
            am = _f64_2d(a)
            if am is not None:
                try:
                    return _kernel.cholesky(am)
                except _KERNEL_LINALG_ERROR as e:  # type: ignore[misc]
                    raise LinAlgError(str(e)) from None
        return _np.linalg.cholesky(a, *args, **kwargs)

    def det(self, a: Any) -> Any:
        if HAVE_KERNEL:
            am = _f64_2d(a)
            if am is not None:
                return _kernel.det(am)
        return _np.linalg.det(a)

    def inv(self, a: Any) -> Any:
        if HAVE_KERNEL:
            am = _f64_2d(a)
            if am is not None:
                try:
                    return _kernel.inv(am)
                except _KERNEL_LINALG_ERROR as e:  # type: ignore[misc]
                    raise LinAlgError(str(e)) from None
        return _np.linalg.inv(a)

    def matrix_power(self, a: Any, n: int) -> Any:
        if HAVE_KERNEL:
            am = _f64_2d(a)
            if am is not None:
                return _kernel.matrix_power(am, int(n))
        return _np.linalg.matrix_power(a, n)


def _maximum_call(a: Any, b: Any, **kw: Any) -> Any:
    if HAVE_KERNEL and not kw:
        va, vb = _f64_1d(a), _f64_1d(b)
        if va is not None and vb is not None and va.shape == vb.shape:
            return _kernel.maximum(va, vb)
    return _np.maximum(a, b, **kw)


def _minimum_call(a: Any, b: Any, **kw: Any) -> Any:
    if HAVE_KERNEL and not kw:
        va, vb = _f64_1d(a), _f64_1d(b)
        if va is not None and vb is not None and va.shape == vb.shape:
            return _kernel.minimum(va, vb)
    return _np.minimum(a, b, **kw)


class _Ufunc:
    """A callable that mirrors a numpy ufunc's method surface (CONCEPT:KG-2.314).

    Calling the object routes through *call* (which may be kernel-accelerated),
    keeping ``xp.maximum(a, b)`` identical to before. The ufunc methods
    ``.accumulate`` / ``.reduce`` / ``.outer`` / ``.at`` forward to numpy's real
    ufunc (``numpy.maximum`` / ``numpy.minimum`` …), so ``np.maximum.accumulate``
    resolves under the ``xp`` shim exactly as under plain numpy. When
    ``HAVE_KERNEL`` and the kernel exposes a matching cumulative op
    (``cummax`` / ``cummin``), ``.accumulate`` on a bare 1-D float64 input is
    kernel-routed; otherwise numpy handles it (parity-proven).
    """

    __slots__ = ("_name", "_npufunc", "_call", "_kernel_accum")

    def __init__(self, name: str, call: Any, kernel_accum: str | None = None) -> None:
        self._name = name
        self._npufunc = getattr(_np, name)
        self._call = call
        self._kernel_accum = kernel_accum

    def __call__(self, *args: Any, **kw: Any) -> Any:
        return self._call(*args, **kw)

    def accumulate(
        self, array: Any, axis: int = 0, dtype: Any = None, out: Any = None
    ) -> Any:
        if (
            HAVE_KERNEL
            and self._kernel_accum is not None
            and axis in (0, -1)
            and dtype is None
            and out is None
        ):
            kfn = getattr(_kernel, self._kernel_accum, None)
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
    """The ``xp`` namespace: numpy-delegating, kernel-accelerated where safe."""

    linalg = _Linalg()
    LinAlgError = LinAlgError

    def __getattr__(self, name: str) -> Any:  # numpy fallback for everything else
        return getattr(_np, name)

    # ---- reductions / stats (route only for bare 1-D float64, no axis) ----
    def sum(self, a: Any, axis: Any = None, **kw: Any) -> Any:
        if HAVE_KERNEL and axis is None and not kw:
            v = _f64_1d(a)
            if v is not None:
                return _kernel.sum(v)
        return _np.sum(a, axis=axis, **kw)

    def prod(self, a: Any, axis: Any = None, **kw: Any) -> Any:
        if HAVE_KERNEL and axis is None and not kw:
            v = _f64_1d(a)
            if v is not None:
                return _kernel.prod(v)
        return _np.prod(a, axis=axis, **kw)

    def mean(self, a: Any, axis: Any = None, **kw: Any) -> Any:
        if HAVE_KERNEL and axis is None and not kw:
            v = _f64_1d(a)
            if v is not None:
                return _kernel.mean(v)
        return _np.mean(a, axis=axis, **kw)

    def std(self, a: Any, axis: Any = None, ddof: int = 0, **kw: Any) -> Any:
        if HAVE_KERNEL and axis is None and not kw:
            v = _f64_1d(a)
            if v is not None:
                return _kernel.std(v, ddof)
        return _np.std(a, axis=axis, ddof=ddof, **kw)

    def var(self, a: Any, axis: Any = None, ddof: int = 0, **kw: Any) -> Any:
        if HAVE_KERNEL and axis is None and not kw:
            v = _f64_1d(a)
            if v is not None:
                return _kernel.var(v, ddof)
        return _np.var(a, axis=axis, ddof=ddof, **kw)

    def min(self, a: Any, axis: Any = None, **kw: Any) -> Any:
        if HAVE_KERNEL and axis is None and not kw:
            v = _f64_1d(a)
            if v is not None and v.size:
                return _kernel.amin(v)
        return _np.min(a, axis=axis, **kw)

    amin = min

    def max(self, a: Any, axis: Any = None, **kw: Any) -> Any:
        if HAVE_KERNEL and axis is None and not kw:
            v = _f64_1d(a)
            if v is not None and v.size:
                return _kernel.amax(v)
        return _np.max(a, axis=axis, **kw)

    amax = max

    def argmin(self, a: Any, axis: Any = None, **kw: Any) -> Any:
        if HAVE_KERNEL and axis is None and not kw:
            v = _f64_1d(a)
            if v is not None and v.size:
                return _kernel.argmin(v)
        return _np.argmin(a, axis=axis, **kw)

    def argmax(self, a: Any, axis: Any = None, **kw: Any) -> Any:
        if HAVE_KERNEL and axis is None and not kw:
            v = _f64_1d(a)
            if v is not None and v.size:
                return _kernel.argmax(v)
        return _np.argmax(a, axis=axis, **kw)

    def argsort(self, a: Any, axis: int = -1, kind: Any = None, **kw: Any) -> Any:
        if HAVE_KERNEL and axis in (-1, 0) and not kw:
            v = _f64_1d(a)
            if v is not None:
                return _kernel.argsort(v)
        return _np.argsort(a, axis=axis, kind=kind, **kw)

    def cumsum(self, a: Any, axis: Any = None, **kw: Any) -> Any:
        if HAVE_KERNEL and axis is None and not kw:
            v = _f64_1d(a)
            if v is not None:
                return _kernel.cumsum(v)
        return _np.cumsum(a, axis=axis, **kw)

    def cumprod(self, a: Any, axis: Any = None, **kw: Any) -> Any:
        if HAVE_KERNEL and axis is None and not kw:
            v = _f64_1d(a)
            if v is not None:
                return _kernel.cumprod(v)
        return _np.cumprod(a, axis=axis, **kw)

    def percentile(self, a: Any, q: Any, axis: Any = None, **kw: Any) -> Any:
        if HAVE_KERNEL and axis is None and not kw and _np.isscalar(q):
            v = _f64_1d(a)
            if v is not None and v.size:
                return _kernel.percentile(v, float(q))
        return _np.percentile(a, q, axis=axis, **kw)

    def quantile(self, a: Any, q: Any, axis: Any = None, **kw: Any) -> Any:
        if HAVE_KERNEL and axis is None and not kw and _np.isscalar(q):
            v = _f64_1d(a)
            if v is not None and v.size:
                return _kernel.quantile(v, float(q))
        return _np.quantile(a, q, axis=axis, **kw)

    # ---- element-wise (route bare 1-D float64) ----
    def _ew1(self, npfn: Any, kfn: Any, a: Any, **kw: Any) -> Any:
        if HAVE_KERNEL and not kw:
            v = _f64_1d(a)
            if v is not None:
                return kfn(v)
        return npfn(a, **kw)

    def sqrt(self, a: Any, **kw: Any) -> Any:
        return self._ew1(_np.sqrt, _kernel.sqrt if HAVE_KERNEL else None, a, **kw)

    def log(self, a: Any, **kw: Any) -> Any:
        return self._ew1(_np.log, _kernel.log if HAVE_KERNEL else None, a, **kw)

    def exp(self, a: Any, **kw: Any) -> Any:
        return self._ew1(_np.exp, _kernel.exp if HAVE_KERNEL else None, a, **kw)

    def abs(self, a: Any, **kw: Any) -> Any:
        return self._ew1(_np.abs, _kernel.absolute if HAVE_KERNEL else None, a, **kw)

    absolute = abs

    def tanh(self, a: Any, **kw: Any) -> Any:
        return self._ew1(_np.tanh, _kernel.tanh if HAVE_KERNEL else None, a, **kw)

    def clip(self, a: Any, a_min: Any = None, a_max: Any = None, **kw: Any) -> Any:
        if (
            HAVE_KERNEL
            and not kw
            and a_min is not None
            and a_max is not None
            and _np.isscalar(a_min)
            and _np.isscalar(a_max)
        ):
            v = _f64_1d(a)
            if v is not None:
                return _kernel.clip(v, float(a_min), float(a_max))
        return _np.clip(a, a_min, a_max, **kw)

    def nan_to_num(
        self,
        a: Any,
        copy: bool = True,
        nan: float = 0.0,
        posinf: Any = None,
        neginf: Any = None,
    ) -> Any:
        if HAVE_KERNEL and _np.isscalar(nan):
            v = _f64_1d(a)
            if v is not None:
                pi = float(posinf) if posinf is not None else _np.finfo(_np.float64).max
                ni = float(neginf) if neginf is not None else _np.finfo(_np.float64).min
                return _kernel.nan_to_num(v, float(nan), pi, ni)
        return _np.nan_to_num(a, copy=copy, nan=nan, posinf=posinf, neginf=neginf)

    def isnan(self, a: Any, **kw: Any) -> Any:
        if HAVE_KERNEL and not kw:
            v = _f64_1d(a)
            if v is not None:
                return _np.asarray(_kernel.isnan(v), dtype=bool)
        return _np.isnan(a, **kw)

    #: ufunc-method surface (CONCEPT:KG-2.314): callable + ``.accumulate`` /
    #: ``.reduce`` / ``.outer`` / ``.at``. Class attributes (not methods) so
    #: ``xp.maximum`` yields the wrapper object, not a bound method.
    maximum = _Ufunc("maximum", _maximum_call, kernel_accum="cummax")
    minimum = _Ufunc("minimum", _minimum_call, kernel_accum="cummin")

    def where(self, condition: Any, *args: Any) -> Any:
        if HAVE_KERNEL and len(args) == 2:
            cond = _np.asarray(condition)
            va, vb = _f64_1d(args[0]), _f64_1d(args[1])
            if (
                cond.ndim == 1
                and va is not None
                and vb is not None
                and cond.shape == va.shape == vb.shape
            ):
                return _kernel.where_(cond.astype(bool).tolist(), va, vb)
        return _np.where(condition, *args)

    def dot(self, a: Any, b: Any, **kw: Any) -> Any:
        if HAVE_KERNEL and not kw:
            va, vb = _f64_1d(a), _f64_1d(b)
            if va is not None and vb is not None and va.shape == vb.shape:
                return _kernel.dot(va, vb)
            ma, mb = _f64_2d(a), _f64_2d(b)
            if ma is not None and mb is not None and ma.shape[1] == mb.shape[0]:
                return _kernel.matmul(ma, mb)
        return _np.dot(a, b, **kw)

    def matmul(self, a: Any, b: Any, **kw: Any) -> Any:
        if HAVE_KERNEL and not kw:
            ma, mb = _f64_2d(a), _f64_2d(b)
            if ma is not None and mb is not None and ma.shape[1] == mb.shape[0]:
                return _kernel.matmul(ma, mb)
        return _np.matmul(a, b, **kw)


#: The public namespace. Import as ``from agent_utilities.numeric import xp as np``.
xp = _XP()

__all__ = ["xp", "HAVE_KERNEL", "KERNEL_SOURCE", "LinAlgError"]
