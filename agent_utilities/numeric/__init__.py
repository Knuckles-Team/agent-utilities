"""Thin adapter for the certified ``epistemic_graph.numeric`` kernel.

CONCEPT:AU-KG.compute.numeric-kernel
CONCEPT:AU-KG.compute.surface-analytics-program
CONCEPT:AU-KG.compute.numpy-scipy-drop

The Rust extension is the numeric authority.  This module deliberately does not
provide an array object, an operation dispatcher, or a compatibility fallback.
Inputs and outputs cross the boundary as scalars and bounded builtin lists (the
extension owns validation and computation).  A missing operation is an explicit
contract failure instead of silently selecting another numeric runtime.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import math
import os
import stat
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any, TypeAlias


class UnsupportedNumericOperationError(NotImplementedError):
    """Raised when AU asks for a surface the native kernel does not expose."""


class NumericKernelError(RuntimeError):
    """Compatibility exception when a kernel omits a typed error export."""


NDArray: TypeAlias = Any
RandomGenerator: TypeAlias = Any

_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_ARTIFACT_SCHEMA = "eg-numeric-list-v1"
_MAX_NUMERIC_RANK = 8
_MAX_NUMERIC_ELEMENTS = 1_000_000
_MAX_NUMERIC_NODES = 2_000_000


try:
    _KERNEL: Any = importlib.import_module("epistemic_graph.numeric")
except ImportError as exc:
    raise ImportError(
        "the certified epistemic_graph.numeric kernel is required by "
        "agent_utilities.numeric and is not importable"
    ) from exc

if getattr(_KERNEL, "__kernel__", None) != "eg-numeric":
    raise ImportError(
        "epistemic_graph.numeric is not the certified eg-numeric kernel "
        f"(found __kernel__={getattr(_KERNEL, '__kernel__', None)!r})"
    )


def _to_builtin(value: Any, *, _depth: int = 0, _state: list[int] | None = None) -> Any:
    """Convert boundary values without importing or implementing an array type.

    Native callers use scalars, tuples, and lists.  ``to_pylist`` (Arrow) is
    accepted only as an already-supported producer boundary; the returned
    value is recursively validated as a builtin tree.  No arithmetic,
    broadcasting, indexing, or shape logic belongs here.
    """

    if _state is None:
        _state = [0, 0]
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        _state[0] += 1
        if _state[0] > _MAX_NUMERIC_ELEMENTS:
            raise ValueError(
                f"numeric input exceeds the {_MAX_NUMERIC_ELEMENTS}-element limit"
            )
        return value
    if _depth >= _MAX_NUMERIC_RANK:
        raise ValueError(f"numeric input exceeds the rank-{_MAX_NUMERIC_RANK} limit")
    if isinstance(value, (list, tuple)):
        _state[1] += 1
        if _state[1] > _MAX_NUMERIC_NODES:
            raise ValueError(
                f"numeric input exceeds the {_MAX_NUMERIC_NODES}-node limit"
            )
        converted = [
            _to_builtin(item, _depth=_depth + 1, _state=_state) for item in value
        ]
        return converted if isinstance(value, list) else tuple(converted)
    to_pylist = getattr(value, "to_pylist", None)
    if callable(to_pylist):
        converted = to_pylist()
        if converted is not value:
            return _to_builtin(converted, _depth=_depth, _state=_state)
    raise TypeError(
        "numeric kernel inputs must be scalars, builtin list/tuple trees, or "
        "Arrow values exposing to_pylist(); "
        f"got {type(value).__name__}"
    )


def _from_builtin(value: Any) -> Any:
    """Keep native scalar/list/tuple results as plain Python values."""

    return _to_builtin(value)


def to_builtin(value: Any) -> Any:
    """Convert one supported producer value to a detached builtin tree.

    The boundary deliberately recognizes Arrow's ``to_pylist`` protocol and
    builtin sequences only.  NumPy/pandas/polars ``tolist`` methods are not a
    numeric runtime contract; callers must make an explicit ``list(...)`` or
    Arrow conversion at their own serialization seam.
    """

    return _to_builtin(value)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _artifact_payload(value: Any) -> bytes:
    converted = _to_builtin(value)
    try:
        body = {"schema": _ARTIFACT_SCHEMA, "values": converted}
        digest = hashlib.sha256(_canonical_json(body)).hexdigest()
        encoded = _canonical_json({**body, "digest": digest})
    except (TypeError, ValueError) as exc:
        raise ValueError("numeric artifact values must be JSON-compatible") from exc
    if len(encoded) > _MAX_ARTIFACT_BYTES:
        raise ValueError("numeric artifact exceeds its safe size bound")
    return encoded


def save_numeric_artifact(path: str | Path, value: Any) -> None:
    """Write a bounded, versioned JSON artifact for numeric builtin values.

    This is intentionally a list/artifact seam rather than ``save``/``load``
    compatibility.  It carries no dtype, object-array, or executable payload.
    """

    destination = Path(path)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.exists():
        existing = destination.lstat()
        if stat.S_ISLNK(existing.st_mode):
            raise ValueError("numeric artifact path must not be a symlink")
        if not stat.S_ISREG(existing.st_mode):
            raise ValueError("numeric artifact path must be a regular file")
    payload = _artifact_payload(value)
    try:
        parent_fd = os.open(
            destination.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError as exc:
        raise ValueError("numeric artifact parent must be a real directory") from exc
    temporary_name: str | None = None
    descriptor: int | None = None
    try:
        for attempt in range(16):
            candidate = f".{destination.name}.{os.getpid()}.{attempt}"
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o600,
                    dir_fd=parent_fd,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if descriptor is None or temporary_name is None:
            raise ValueError("could not allocate a private numeric artifact temporary")
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary_name,
            destination.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary_name = None
        os.fsync(parent_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:  # noqa: BLE001 — atomic replace may already have consumed the private temporary
                pass
        os.close(parent_fd)


def load_numeric_artifact(path: str | Path) -> Any:
    """Read and validate a bounded JSON numeric artifact."""

    source = Path(path)
    try:
        descriptor = os.open(
            source,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        metadata = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError("numeric artifact is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise ValueError("numeric artifact must be a regular file")
    if metadata.st_size <= 0 or metadata.st_size > _MAX_ARTIFACT_BYTES:
        os.close(descriptor)
        raise ValueError("numeric artifact exceeds its safe size bound")
    try:
        raw = os.read(descriptor, metadata.st_size)
        if len(raw) != metadata.st_size:
            raise ValueError("numeric artifact changed while being read")
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("numeric artifact is invalid") from exc
    finally:
        os.close(descriptor)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != _ARTIFACT_SCHEMA
        or "values" not in payload
        or set(payload) != {"schema", "values", "digest"}
    ):
        raise ValueError("numeric artifact schema is invalid")
    try:
        values = _to_builtin(payload.get("values"))
    except (TypeError, ValueError) as exc:
        raise ValueError("numeric artifact values are not builtin data") from exc
    expected = payload.get("digest")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("numeric artifact digest is invalid")
    if any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("numeric artifact digest is invalid")
    try:
        actual = hashlib.sha256(
            _canonical_json({"schema": _ARTIFACT_SCHEMA, "values": values})
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ValueError("numeric artifact digest is invalid") from exc
    if not hmac.compare_digest(actual, expected):
        raise ValueError("numeric artifact digest is invalid")
    return values


def _call_native(kernel: Any, native_name: str, *args: Any, **kwargs: Any) -> Any:
    """Call one allowlisted native function with only boundary conversion."""

    function = getattr(kernel, native_name, None)
    if not callable(function):
        raise UnsupportedNumericOperationError(
            f"native numeric kernel does not expose {native_name}()"
        )
    converted_args = tuple(_to_builtin(arg) for arg in args)
    converted_kwargs = {key: _to_builtin(value) for key, value in kwargs.items()}
    try:
        result = function(*converted_args, **converted_kwargs)
    except TypeError as exc:
        raise UnsupportedNumericOperationError(
            f"native numeric kernel rejected the {native_name} boundary: {exc}"
        ) from exc
    return _from_builtin(result)


# This is intentionally an explicit contract.  Adding a name requires the EG
# extension to export the operation and a focused contract test; unknown kernel
# attributes must never leak into AU as an accidental second API.
_ROOT_OPERATIONS = frozenset(
    {
        "sum",
        "prod",
        "mean",
        "var",
        "std",
        "amin",
        "amax",
        "argmin",
        "argmax",
        "argsort",
        "cumsum",
        "cumprod",
        "percentile",
        "quantile",
        "sqrt",
        "log",
        "exp",
        "absolute",
        "tanh",
        "clip",
        "nan_to_num",
        "isnan",
        "maximum",
        "minimum",
        "where_",
        "norm",
        "norm_ord",
        "dot",
        "matmul",
        "solve",
        "svdvals",
        "svd",
        "eigh",
        "eigsh",
        "pinv",
        "lstsq",
        "qr",
        "cholesky",
        "det",
        "inv",
        "matrix_power",
        "spearmanr",
        "ks_2samp",
        "norm_ppf",
        "norm_pdf",
        "kmeans",
        "normal",
        "uniform",
        "integers",
        "choice_indices",
        "permutation_indices",
    }
)

_ROOT_ALIASES = {
    "abs": "absolute",
    "min": "amin",
    "max": "amax",
    "where": "where_",
}

_LINALG_OPERATIONS = frozenset(
    {
        "norm",
        "solve",
        "svd",
        "eigh",
        "pinv",
        "lstsq",
        "qr",
        "cholesky",
        "det",
        "inv",
        "matrix_power",
    }
)


def _shape_from_size(
    size: int | tuple[int, ...] | list[int] | None,
) -> tuple[tuple[int, ...], int]:
    """Normalize a NumPy-shaped request without creating an array object."""

    shape: tuple[int, ...]
    if size is None:
        return (), 1
    if isinstance(size, int) and not isinstance(size, bool):
        shape = (size,)
    elif (
        isinstance(size, (tuple, list))
        and len(size) <= _MAX_NUMERIC_RANK
        and all(isinstance(item, int) and not isinstance(item, bool) for item in size)
    ):
        shape = tuple(size)
    else:
        raise TypeError(
            f"random size must be an integer or a tuple of at most {_MAX_NUMERIC_RANK} integers"
        )
    if any(item < 0 or item > _MAX_NUMERIC_ELEMENTS for item in shape):
        raise ValueError("random size entries must be non-negative")
    count = 1
    for item in shape:
        if item == 0:
            count = 0
            continue
        if count == 0:
            continue
        if item and count > _MAX_NUMERIC_ELEMENTS // item:
            raise ValueError(
                f"random size exceeds the {_MAX_NUMERIC_ELEMENTS}-element limit"
            )
        count *= item
    return shape, count


def _reshape(values: list[Any], shape: tuple[int, ...]) -> Any:
    if not shape:
        return values[0]
    width = 1
    for item in shape[1:]:
        width *= item
    return [
        _reshape(values[index * width : (index + 1) * width], shape[1:])
        for index in range(shape[0])
    ]


def _splitmix64(value: int) -> int:
    """Derive a reproducible independent seed for one generator draw."""

    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (value ^ (value >> 31)) & 0xFFFFFFFFFFFFFFFF


class _DeterministicRandom:
    """Small stateful facade over the engine's seeded one-shot draws.

    The object owns no values beyond a seed/draw counter.  Every operation is
    one bounded call to ``epistemic_graph.numeric`` and returns a scalar or
    builtin list, never an array-like object.
    """

    __slots__ = ("_kernel", "_seed", "_draw")

    def __init__(self, kernel: Any, seed: int) -> None:
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("random seed must be a non-negative integer")
        self._kernel = kernel
        self._seed = seed & 0xFFFFFFFFFFFFFFFF
        self._draw = 0

    def _next_seed(self) -> int:
        seed = _splitmix64(self._seed + self._draw)
        self._draw += 1
        return seed

    def _draw_values(
        self,
        native_name: str,
        *args: Any,
        size: int | tuple[int, ...] | list[int] | None,
    ) -> Any:
        shape, count = _shape_from_size(size)
        values = _call_native(
            self._kernel, native_name, *args, count, self._next_seed()
        )
        if not isinstance(values, list) or len(values) != count:
            raise NumericKernelError(
                f"native random operation {native_name} did not return a list"
            )
        return _reshape(values, shape)

    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: int | tuple[int, ...] | list[int] | None = None,
    ) -> Any:
        return self._draw_values("normal", loc, scale, size=size)

    def standard_normal(
        self, size: int | tuple[int, ...] | list[int] | None = None
    ) -> Any:
        return self.normal(0.0, 1.0, size)

    def randn(self, *size: int) -> Any:
        return self.standard_normal(size if size else None)

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: int | tuple[int, ...] | list[int] | None = None,
    ) -> Any:
        return self._draw_values("uniform", low, high, size=size)

    def random(self, size: int | tuple[int, ...] | list[int] | None = None) -> Any:
        return self.uniform(0.0, 1.0, size)

    def integers(
        self,
        low: int,
        high: int | None = None,
        size: int | tuple[int, ...] | list[int] | None = None,
        endpoint: bool = False,
    ) -> Any:
        if high is None:
            low, high = 0, low
        assert high is not None
        if endpoint:
            high += 1
        return self._draw_values("integers", low, high, size=size)

    def randint(
        self,
        low: int,
        high: int | None = None,
        size: int | tuple[int, ...] | list[int] | None = None,
    ) -> Any:
        return self.integers(low, high, size)

    def choice(
        self,
        values: int | list[Any] | tuple[Any, ...],
        size: int | tuple[int, ...] | list[int] | None = None,
        replace: bool = True,
        p: list[float] | tuple[float, ...] | None = None,
    ) -> Any:
        population = list(range(values)) if isinstance(values, int) else list(values)
        if not population:
            raise ValueError("choice cannot sample an empty population")
        shape, count = _shape_from_size(size)
        if not replace and count > len(population):
            raise ValueError(
                "cannot take a larger sample than population without replacement"
            )
        if p is not None:
            probabilities = [float(item) for item in p]
            if len(probabilities) != len(population) or any(
                not math.isfinite(item) or item < 0 for item in probabilities
            ):
                raise ValueError(
                    "choice probabilities must match the population and be non-negative"
                )
            total = sum(probabilities)
            if not math.isfinite(total) or total <= 0:
                raise ValueError("choice probabilities must have a positive sum")
        else:
            probabilities = None
        indices = _call_native(
            self._kernel,
            "choice_indices",
            len(population),
            count,
            replace,
            probabilities,
            self._next_seed(),
        )
        if not isinstance(indices, list):
            raise NumericKernelError("native choice operation did not return a list")
        if len(indices) != count or any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(population)
            for index in indices
        ):
            raise NumericKernelError("native choice operation returned invalid indices")
        if not replace and len(set(indices)) != len(indices):
            raise NumericKernelError(
                "native choice operation returned duplicate indices"
            )
        selected = [population[int(index)] for index in indices]
        return _reshape(selected, shape)

    def shuffle(self, values: list[Any]) -> None:
        permutation = _call_native(
            self._kernel,
            "permutation_indices",
            len(values),
            self._next_seed(),
        )
        if not isinstance(permutation, list) or len(permutation) != len(values):
            raise NumericKernelError(
                "native permutation operation did not return a list"
            )
        if any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(values)
            for index in permutation
        ) or len(set(permutation)) != len(values):
            raise NumericKernelError(
                "native permutation operation returned invalid indices"
            )
        values[:] = [values[int(index)] for index in permutation]

    def __getattr__(self, name: str) -> Any:
        raise UnsupportedNumericOperationError(
            f"xp.random.Generator.{name} is not part of the native random contract"
        )


class _RandomNamespace:
    """Allowlisted deterministic random constructors."""

    __slots__ = ("_kernel",)
    _OPERATIONS = frozenset({"default_rng", "RandomState"})

    def __init__(self, kernel: Any) -> None:
        self._kernel = kernel

    def default_rng(self, seed: int) -> _DeterministicRandom:
        return _DeterministicRandom(self._kernel, seed)

    RandomState = default_rng

    def __getattr__(self, name: str) -> Any:
        raise UnsupportedNumericOperationError(
            f"xp.random.{name} is not part of the native random contract"
        )

    def __dir__(self) -> list[str]:
        return sorted(self._OPERATIONS)


class _NativeNamespace:
    """An allowlisted view over one native kernel namespace."""

    __slots__ = ("_kernel", "_operations", "_aliases", "_namespace", "_error_type")

    def __init__(
        self,
        kernel: Any,
        operations: frozenset[str],
        aliases: Mapping[str, str] | None = None,
        namespace: str = "xp",
        error_type: type[BaseException] | None = None,
    ) -> None:
        self._kernel = kernel
        self._operations = operations
        self._aliases = dict(aliases or {})
        self._namespace = namespace
        self._error_type = error_type

    def __getattr__(self, name: str) -> Any:
        if name == "LinAlgError" and self._error_type is not None:
            return self._error_type
        native_name = self._aliases.get(name, name)
        if native_name not in self._operations:
            raise UnsupportedNumericOperationError(
                f"{self._namespace}.{name} is not part of the native numeric contract"
            )
        return partial(_call_native, self._kernel, native_name)

    def __dir__(self) -> list[str]:
        return sorted((*self._operations, *self._aliases))


class _XP(_NativeNamespace):
    """The explicit native root namespace; no fallback attribute resolution."""

    __slots__ = ("linalg", "random", "LinAlgError")

    def __init__(self, kernel: Any) -> None:
        super().__init__(kernel, _ROOT_OPERATIONS, _ROOT_ALIASES)
        error_type = getattr(kernel, "LinAlgError", NumericKernelError)
        self.linalg = _NativeNamespace(
            kernel,
            _LINALG_OPERATIONS,
            namespace="xp.linalg",
            error_type=error_type,
        )
        self.random = _RandomNamespace(kernel)
        self.LinAlgError = error_type


xp = _XP(_KERNEL)

__all__ = [
    "xp",
    "LinAlgError",
    "NDArray",
    "RandomGenerator",
    "to_builtin",
    "save_numeric_artifact",
    "load_numeric_artifact",
    "UnsupportedNumericOperationError",
    "NumericKernelError",
]

LinAlgError = xp.LinAlgError
