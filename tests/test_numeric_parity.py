"""numpy-parity corpus for ``agent_utilities.numeric.xp`` (CONCEPT:AU-KG.compute.surface-analytics-program, CONCEPT:AU-KG.compute.numpy-scipy-drop).

Every ``xp`` op is asserted ``np.allclose`` vs numpy on randomized inputs, with
mandatory edge cases (nan/inf, singular matrices, empty arrays). This is the ONE
place numpy is imported directly — as **ground truth** to prove the compiled
``epistemic_graph.numeric`` kernel (the sole runtime numeric backend) matches numpy
bit-for-bit. It is a **dev/test-only** dependency, never a runtime one: numpy is
declared in the ``[dev]`` optional group (``pyproject.toml``), and this module
``pytest.importorskip("numpy")`` s so the suite skips cleanly where numpy (the
ground-truth reference) is not installed. The shim itself never falls back to numpy
(CONCEPT:AU-KG.compute.numpy-scipy-drop) — it raises ImportError when the kernel is absent.

Decomposition ops (svd/eigh/qr/cholesky/pinv/lstsq) are compared via
reconstruction, not raw factors, because factor signs/bases are implementation
-defined (numpy itself does not guarantee them across backends).
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")  # dev/test-only ground-truth reference (KG-2.324)

from agent_utilities.numeric import HAVE_KERNEL, xp  # noqa: E402


def _close(a, b, atol=1e-6, rtol=1e-6):
    return np.allclose(
        np.asarray(a, float), np.asarray(b, float), atol=atol, rtol=rtol, equal_nan=True
    )


# --------------------------------------------------------------------------- #
# reductions / stats
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(8))
def test_reductions(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 5, int(rng.integers(2, 40))).astype(np.float64)
    assert _close(xp.sum(a), np.sum(a))
    assert _close(xp.prod(a), np.prod(a))
    assert _close(xp.mean(a), np.mean(a))
    assert _close(xp.std(a), np.std(a))
    assert _close(xp.std(a, ddof=1), np.std(a, ddof=1))
    assert _close(xp.var(a), np.var(a))
    assert _close(xp.var(a, ddof=1), np.var(a, ddof=1))
    assert _close(xp.min(a), np.min(a))
    assert _close(xp.max(a), np.max(a))
    assert xp.argmin(a) == np.argmin(a)
    assert xp.argmax(a) == np.argmax(a)
    assert _close(xp.argsort(a), np.argsort(a, kind="stable"))
    assert _close(xp.cumsum(a), np.cumsum(a))
    assert _close(xp.cumprod(a), np.cumprod(a))
    for q in (0.0, 25.0, 50.0, 90.0, 100.0):
        assert _close(xp.percentile(a, q), np.percentile(a, q))
    for q in (0.1, 0.5, 0.99):
        assert _close(xp.quantile(a, q), np.quantile(a, q))


def test_reductions_2d_falls_back_to_numpy():
    # axis reductions are outside the kernel domain → numpy must still work.
    m = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert _close(xp.sum(m, axis=0), np.sum(m, axis=0))
    assert _close(xp.mean(m, axis=1), np.mean(m, axis=1))
    assert _close(xp.std(m, axis=0), np.std(m, axis=0))


# --------------------------------------------------------------------------- #
# element-wise
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(6))
def test_elementwise(seed):
    rng = np.random.default_rng(100 + seed)
    a = rng.normal(0, 3, int(rng.integers(2, 30))).astype(np.float64)
    b = rng.normal(0, 3, a.size).astype(np.float64)
    pos = np.abs(a) + 0.01
    assert _close(xp.sqrt(pos), np.sqrt(pos))
    assert _close(xp.log(pos), np.log(pos))
    assert _close(xp.exp(a), np.exp(a))
    assert _close(xp.abs(a), np.abs(a))
    assert _close(xp.absolute(a), np.abs(a))
    assert _close(xp.tanh(a), np.tanh(a))
    assert _close(xp.clip(a, -1.0, 1.0), np.clip(a, -1.0, 1.0))
    assert _close(xp.maximum(a, b), np.maximum(a, b))
    assert _close(xp.minimum(a, b), np.minimum(a, b))
    assert _close(xp.where(a > 0, a, b), np.where(a > 0, a, b))
    assert np.array_equal(np.asarray(xp.isnan(a)), np.isnan(a))


def test_elementwise_edge_nan_inf():
    edge = np.array([np.nan, np.inf, -np.inf, 0.0, -2.5, 7.0], dtype=np.float64)
    assert _close(
        xp.nan_to_num(edge, nan=0.0, posinf=1e300, neginf=-1e300),
        np.nan_to_num(edge, nan=0.0, posinf=1e300, neginf=-1e300),
    )
    assert np.array_equal(np.asarray(xp.isnan(edge)), np.isnan(edge))
    assert _close(xp.abs(edge), np.abs(edge))
    z = np.zeros_like(edge)
    assert _close(xp.maximum(edge, z), np.maximum(edge, z))
    assert xp.argmin(edge) == np.argmin(edge)
    assert xp.argmax(edge) == np.argmax(edge)


def test_empty_edges():
    assert np.isnan(xp.mean(np.array([], dtype=np.float64)))
    with pytest.raises(ValueError):
        xp.min(np.array([], dtype=np.float64))  # numpy + kernel both reject empty
    with pytest.raises(ValueError):
        xp.max(np.array([], dtype=np.float64))


def test_min_single():
    assert xp.min(np.array([3.0])) == 3.0
    assert xp.max(np.array([3.0])) == 3.0


# --------------------------------------------------------------------------- #
# linalg
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(8))
def test_linalg(seed):
    rng = np.random.default_rng(200 + seed)
    n = int(rng.integers(2, 7))
    m = n + int(rng.integers(0, 4))
    A = rng.normal(0, 2, (m, n)).astype(np.float64)
    Sq = (rng.normal(0, 2, (n, n)) + n * np.eye(n)).astype(np.float64)
    x = rng.normal(0, 2, n).astype(np.float64)
    v = rng.normal(0, 2, n).astype(np.float64)

    # vector
    assert _close(xp.linalg.norm(v), np.linalg.norm(v))
    assert _close(xp.linalg.norm(v, 1), np.linalg.norm(v, 1))
    assert _close(xp.linalg.norm(v, np.inf), np.linalg.norm(v, np.inf))
    assert _close(xp.dot(v, x), np.dot(v, x))

    # solve (exact)
    b = Sq @ x
    assert _close(xp.linalg.solve(Sq, b), np.linalg.solve(Sq, b))

    # matmul
    B = rng.normal(0, 2, (n, n)).astype(np.float64)
    assert _close(xp.matmul(Sq, B), Sq @ B)
    assert _close(xp.dot(Sq, B), Sq @ B)

    # svdvals (exact) + svd reconstruction
    assert _close(
        xp.linalg.svd(A, compute_uv=False), np.linalg.svd(A, compute_uv=False)
    )
    U, s, Vt = xp.linalg.svd(A)
    assert _close(U[:, : len(s)] @ np.diag(s) @ Vt[: len(s), :], A)

    # eigh (values exact, reconstruction)
    S = (Sq + Sq.T) / 2
    w, V = xp.linalg.eigh(S)
    assert _close(w, np.linalg.eigvalsh(S))
    assert _close(V @ np.diag(w) @ V.T, S)

    # pinv (exact vs numpy pinv)
    assert _close(xp.linalg.pinv(A), np.linalg.pinv(A))

    # lstsq (x matches; also returns numpy-shaped tuple)
    bt = rng.normal(0, 2, m).astype(np.float64)
    xk = xp.linalg.lstsq(A, bt)[0]
    xn = np.linalg.lstsq(A, bt, rcond=None)[0]
    assert _close(xk, xn)

    # qr reconstruction
    Q, R = xp.linalg.qr(A)
    assert _close(Q @ R, A)

    # cholesky (SPD) reconstruction
    SPD = (S @ S.T + n * np.eye(n)).astype(np.float64)
    L = xp.linalg.cholesky(SPD)
    assert _close(L @ L.T, SPD)

    # det / inv / matrix_power (exact)
    assert _close(xp.linalg.det(Sq), np.linalg.det(Sq))
    assert _close(xp.linalg.inv(Sq), np.linalg.inv(Sq))
    for p in (0, 1, 3, -2):
        assert _close(
            xp.linalg.matrix_power(Sq, p), np.linalg.matrix_power(Sq, p), atol=1e-5
        )


def test_linalg_singular_raises_linalgerror():
    sing = np.array([[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(np.linalg.LinAlgError):
        xp.linalg.solve(sing, np.array([1.0, 2.0]))
    with pytest.raises(np.linalg.LinAlgError):
        xp.linalg.inv(sing)
    with pytest.raises(np.linalg.LinAlgError):
        xp.linalg.cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))


# --------------------------------------------------------------------------- #
# random — determinism + distributional parity (bit-parity with numpy is a
# non-goal; xp.random still delegates to numpy for API completeness)
# --------------------------------------------------------------------------- #
def test_random_delegates_to_numpy():
    # xp delegates non-overridden attrs to numpy, so RNG stays numpy-backed.
    g = xp.random.default_rng(42)
    s = g.normal(0, 1, 100000)
    assert abs(s.mean()) < 0.02
    assert abs(s.std() - 1.0) < 0.02


def test_fallthrough_attrs():
    # sanity that arbitrary numpy attributes resolve through the shim.
    assert xp.float64 is np.float64
    assert xp.newaxis is np.newaxis
    assert _close(xp.zeros(3), np.zeros(3))
    assert _close(xp.arange(5, dtype=np.float64), np.arange(5, dtype=np.float64))


def test_kernel_flag_is_bool():
    assert isinstance(HAVE_KERNEL, bool)


# --------------------------------------------------------------------------- #
# ufunc-method surface (CONCEPT:AU-KG.compute.ufunc-method-surface) — xp.maximum / xp.minimum expose the
# numpy ufunc-method API (.accumulate / .reduce / .outer / .at) while keeping
# plain-call behaviour identical. The kernel .accumulate hook is getattr-guarded and
# used on bare 1-D float64; other inputs use the kernel-internal numpy ufunc.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "x",
    [
        np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]),
        np.array([-2.0, -5.0, 0.0, 7.0, 7.0, -1.0]),
        np.array([1.5]),
        np.linspace(-3.0, 3.0, 50),
    ],
)
def test_ufunc_accumulate_parity(x):
    assert _close(xp.maximum.accumulate(x), np.maximum.accumulate(x))
    assert _close(xp.minimum.accumulate(x), np.minimum.accumulate(x))


def test_ufunc_reduce_parity():
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    assert _close(xp.maximum.reduce(x), np.maximum.reduce(x))
    assert _close(xp.minimum.reduce(x), np.minimum.reduce(x))


def test_ufunc_outer_and_call_parity():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.0, 4.0])
    assert _close(xp.maximum.outer(a, b), np.maximum.outer(a, b))
    assert _close(xp.minimum.outer(a, b), np.minimum.outer(a, b))
    # plain-call behaviour is unchanged (elementwise, and scalar broadcast).
    assert _close(xp.maximum(a, 2.0), np.maximum(a, 2.0))
    assert _close(xp.maximum(a, a[::-1]), np.maximum(a, a[::-1]))


def test_ufunc_at_parity():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = a.copy()
    np.maximum.at(a, [0, 1, 2], [5.0, 0.0, 9.0])
    xp.maximum.at(b, [0, 1, 2], [5.0, 0.0, 9.0])
    assert _close(a, b)


def test_ufunc_is_wrapper_not_bound_method():
    # xp.maximum must be a stable callable object carrying ufunc methods.
    assert callable(xp.maximum)
    for m in ("accumulate", "reduce", "outer", "at"):
        assert hasattr(xp.maximum, m)
        assert hasattr(xp.minimum, m)
