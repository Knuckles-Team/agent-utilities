#!/usr/bin/python
from __future__ import annotations

"""Functions runtime — typed function registry (CONCEPT:AU-KG.ontology.default-runtime-bound-import).

Palantir Foundry ``functions/overview`` parity: a *Function* is a typed,
versioned, releasable unit of user logic with declared inputs and a declared
output. Foundry distinguishes plain *query functions*, *Functions-on-Objects*
(which read/traverse the object graph), and authored functions consumed by
Actions. This module models that contract for the ontology layer:

  - :class:`FunctionKind` — PLAIN | ON_OBJECTS | QUERY (the Foundry function
    categories).
  - :class:`FunctionParameter` / :class:`FunctionSpec` — the typed signature:
    a name, a semver ``version`` string, typed inputs (param name → a Python
    type *or* a schema type name such as ``"string"`` / ``"FLOAT"``), a typed
    output, the ``kind``, a ``released`` publish flag, and the handler callable.
  - :class:`FunctionRegistry` — real registration with duplicate detection,
    multi-version storage, release/publish semantics, and lookup by name (latest
    released, or pinned version).

A module-level :data:`DEFAULT_FUNCTION_REGISTRY` is populated at import with a
real built-in (``object.summarize`` — a Functions-on-Object summarizer — and a
numeric aggregate ``numeric.aggregate``) so it is a live path, not an empty
shell. The :class:`~agent_utilities.knowledge_graph.ontology.functions.runtime.FunctionRuntime`
is the single governed invocation entry that validates I/O against these specs.
"""

import logging
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

# A function handler receives the validated keyword params and returns a value
# that the runtime coerces against the declared output type.
FunctionHandler = Callable[..., Any]


# Schema-type-name → Python type. Mirrors the SQL-ish column type vocabulary in
# models/schema_definition.py (STRING/INT/FLOAT/BOOLEAN) and Foundry's typed I/O
# so a FunctionSpec can declare inputs either as a Python type or a schema name.
_SCHEMA_TYPE_ALIASES: dict[str, type] = {
    "string": str,
    "str": str,
    "text": str,
    "int": int,
    "int64": int,
    "integer": int,
    "long": int,
    "float": float,
    "double": float,
    "decimal": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
    "object": dict,
    "map": dict,
    "dict": dict,
    "array": list,
    "list": list,
    "any": object,
}


def resolve_type(declared: Any) -> type:
    """Resolve a declared input/output type to a concrete Python type.

    Accepts an actual ``type`` (returned as-is) or a schema type name string
    (``"string"``, ``"FLOAT"``, ``"FLOAT[768]"``, ``"INT64"``…). Unknown names
    and ``None`` fall back to :class:`object` (accept-anything) rather than
    raising, so a spec is never un-registerable over a typo — the runtime still
    validates what it *can* resolve.
    """
    if isinstance(declared, type):
        return declared
    if declared is None:
        return object
    name = str(declared).strip().lower()
    # Strip vector/array decoration, e.g. "FLOAT[768]" → list of floats.
    base = name.split("[", 1)[0].strip()
    if "[" in name:
        return list
    return _SCHEMA_TYPE_ALIASES.get(base, object)


class FunctionKind(StrEnum):
    """Foundry function category (``functions/overview``).

    ``PLAIN`` is a pure typed function; ``ON_OBJECTS`` is a Functions-on-Objects
    function that reads/traverses the object graph through the KG facade;
    ``QUERY`` is a query function (typically read-only aggregation/search).
    """

    PLAIN = "plain"
    ON_OBJECTS = "on_objects"
    QUERY = "query"


class FunctionParameter(BaseModel):
    """A typed input parameter to a :class:`FunctionSpec`.

    ``type`` is the *declared* type — a Python type or a schema type name. The
    resolved Python type is exposed via :meth:`py_type`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    type: Any = "any"
    required: bool = True
    description: str = ""

    def py_type(self) -> type:
        """Return the concrete Python type this parameter resolves to."""
        return resolve_type(self.type)


class FunctionSpec(BaseModel):
    """A typed, versioned user function (Foundry ``functions/overview``). CONCEPT:AU-KG.ontology.default-runtime-bound-import.

    Declares the function's identity (``name`` + semver ``version``), its typed
    ``inputs`` and ``output``, its ``kind``, the publish ``released`` flag, and
    the bound ``handler`` callable. The
    :class:`~agent_utilities.knowledge_graph.ontology.functions.runtime.FunctionRuntime`
    validates call params against ``inputs`` and coerces the result to
    ``output`` before returning.

    Attributes:
        name: Function name (registry key; shared across versions).
        version: Semver string (``"MAJOR.MINOR.PATCH"``).
        inputs: Ordered typed parameter list.
        output: Declared output type (Python type or schema name).
        kind: PLAIN | ON_OBJECTS | QUERY.
        handler: The bound callable invoked with validated keyword params.
        released: Whether this version is published (selectable by latest-lookup).
        description: Human/LLM-facing description.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    version: str = "1.0.0"
    inputs: list[FunctionParameter] = Field(default_factory=list)
    output: Any = "any"
    kind: FunctionKind = FunctionKind.PLAIN
    handler: FunctionHandler
    released: bool = False
    description: str = ""

    @field_validator("version")
    @classmethod
    def _check_semver(cls, v: str) -> str:
        parts = v.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError(f"version must be semver 'MAJOR.MINOR.PATCH', got {v!r}")
        return v

    @field_validator("handler")
    @classmethod
    def _check_handler(cls, v: FunctionHandler) -> FunctionHandler:
        if not callable(v):
            raise ValueError("handler must be callable")
        return v

    def output_type(self) -> type:
        """Return the concrete Python type the output resolves to."""
        return resolve_type(self.output)

    def version_tuple(self) -> tuple[int, int, int]:
        """Return the (major, minor, patch) tuple for ordering."""
        a, b, c = (int(x) for x in self.version.split("."))
        return (a, b, c)

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Validate ``params`` against this spec's typed inputs.

        Returns a list of human-readable error strings — empty when valid. A
        missing required input, an unknown key, or a value whose runtime type is
        not assignable to the declared (resolved) type each yields one error.
        Resolved-to-``object`` types accept anything (no type check).
        """
        errors: list[str] = []
        known = {p.name for p in self.inputs}
        for p in self.inputs:
            if p.name not in params:
                if p.required:
                    errors.append(f"missing required input '{p.name}'")
                continue
            expected = p.py_type()
            if expected is object:
                continue
            value = params[p.name]
            if not _type_matches(value, expected):
                errors.append(
                    f"input '{p.name}' expected {expected.__name__}, "
                    f"got {type(value).__name__}"
                )
        for key in params:
            if key not in known:
                errors.append(f"unknown input '{key}'")
        return errors


def _type_matches(value: Any, expected: type) -> bool:
    """Whether ``value`` satisfies the ``expected`` declared type.

    Pragmatic, Foundry-style coercion rules: ``bool`` is *not* an ``int`` here
    (a boolean passed where a number is expected is a type error), ``int`` is
    accepted where ``float`` is declared (numeric widening), and ``list``/``dict``
    match by container kind.
    """
    if expected is float:
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is bool:
        return isinstance(value, bool)
    return isinstance(value, expected)


class FunctionRegistry:
    """Registry of typed, versioned user functions. CONCEPT:AU-KG.ontology.default-runtime-bound-import.

    Stores every (name, version) and tracks a per-name release set. Lookup
    returns the highest *released* version by default, or a pinned version when
    requested. Mirrors Foundry's function versioning/release model.
    """

    def __init__(self) -> None:
        # name -> { version_string -> FunctionSpec }
        self._specs: dict[str, dict[str, FunctionSpec]] = {}

    def register(self, spec: FunctionSpec, *, replace: bool = False) -> FunctionSpec:
        """Register a function version.

        Args:
            spec: The :class:`FunctionSpec` to register.
            replace: When ``False`` (default), re-registering the same
                (name, version) raises; when ``True`` it overwrites in place
                (used for re-publishing a draft).

        Raises:
            ValueError: On a duplicate (name, version) unless ``replace`` is set.
        """
        versions = self._specs.setdefault(spec.name, {})
        if spec.version in versions and not replace:
            raise ValueError(f"function already registered: {spec.name}@{spec.version}")
        versions[spec.version] = spec
        logger.debug(
            "Registered function %s@%s (kind=%s, released=%s)",
            spec.name,
            spec.version,
            spec.kind,
            spec.released,
        )
        return spec

    def release(self, name: str, version: str) -> FunctionSpec:
        """Mark a registered (name, version) as released/published.

        Raises:
            KeyError: If the (name, version) is not registered.
        """
        spec = self._get_exact(name, version)
        if spec is None:
            raise KeyError(f"cannot release unknown function {name}@{version}")
        spec.released = True
        logger.debug("Released function %s@%s", name, version)
        return spec

    def _get_exact(self, name: str, version: str) -> FunctionSpec | None:
        return self._specs.get(name, {}).get(version)

    def get(
        self, name: str, version: str | None = None, *, released_only: bool = False
    ) -> FunctionSpec | None:
        """Resolve a function by name (and optional pinned version).

        With ``version`` given, returns that exact version (or ``None``). Without
        it, returns the highest semver version — preferring *released* versions,
        falling back to the highest draft when none are released (unless
        ``released_only`` forces a released match).
        """
        versions = self._specs.get(name)
        if not versions:
            return None
        if version is not None:
            return versions.get(version)
        released = [s for s in versions.values() if s.released]
        if released:
            return max(released, key=lambda s: s.version_tuple())
        if released_only:
            return None
        return max(versions.values(), key=lambda s: s.version_tuple())

    def versions(self, name: str) -> list[str]:
        """Return all registered version strings for ``name``, semver-ascending."""
        versions = self._specs.get(name, {})
        return [
            s.version
            for s in sorted(versions.values(), key=lambda s: s.version_tuple())
        ]

    def list_functions(self) -> list[FunctionSpec]:
        """Return one spec per name (the resolved latest) for discovery."""
        out: list[FunctionSpec] = []
        for name in self._specs:
            spec = self.get(name)
            if spec is not None:
                out.append(spec)
        return out

    def functions_of_kind(self, kind: FunctionKind) -> list[FunctionSpec]:
        """Return all resolved-latest functions of a given :class:`FunctionKind`."""
        return [s for s in self.list_functions() if s.kind == kind]

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return sum(len(v) for v in self._specs.values())


# ── Built-in functions ──────────────────────────────────────────────────────


def _summarize_object(
    object_id: str = "", properties: dict[str, Any] | None = None
) -> str:
    """Functions-on-Object: a deterministic one-line summary of an object.

    Reads the supplied ``properties`` mapping (already materialized by a caller
    such as :class:`ObjectFunctionContext`) and renders ``name``/``type``/``id``
    into a stable summary. Pure and offline-safe — no graph call here; graph
    reads live in :mod:`objects`.
    """
    properties = properties or {}
    name = properties.get("name") or properties.get("title") or object_id or "object"
    otype = properties.get("type") or properties.get("kind") or ""
    desc = properties.get("description") or properties.get("summary") or ""
    head = f"{name} ({otype})" if otype else str(name)
    if desc:
        head = f"{head}: {str(desc)[:160]}"
    return head


def _numeric_aggregate(values: list[Any], op: str = "sum") -> float:
    """Query function: aggregate a numeric list (sum/mean/min/max/count).

    Non-numeric entries are ignored. An empty (or all-non-numeric) input yields
    ``0.0`` for every op except ``count`` (which counts numeric entries).
    """
    nums = [
        float(v)
        for v in values
        if isinstance(v, int | float) and not isinstance(v, bool)
    ]
    op = (op or "sum").lower()
    if op == "count":
        return float(len(nums))
    if not nums:
        return 0.0
    if op == "sum":
        return float(sum(nums))
    if op in ("mean", "avg", "average"):
        return float(sum(nums) / len(nums))
    if op == "min":
        return float(min(nums))
    if op == "max":
        return float(max(nums))
    raise ValueError(f"unknown aggregate op: {op!r}")


def register_builtins(registry: FunctionRegistry) -> None:
    """Register and release the built-in functions into ``registry``.

    Real, live-path functions so the default registry is not an empty shell:
      - ``object.summarize`` — a Functions-on-Object summarizer (ON_OBJECTS).
      - ``numeric.aggregate`` — a numeric aggregate query function (QUERY).
    """
    registry.register(
        FunctionSpec(
            name="object.summarize",
            version="1.0.0",
            kind=FunctionKind.ON_OBJECTS,
            inputs=[
                FunctionParameter(name="object_id", type="string", required=False),
                FunctionParameter(name="properties", type="object", required=False),
            ],
            output="string",
            handler=_summarize_object,
            description="Render a stable one-line summary of an ontology object.",
        )
    )
    registry.release("object.summarize", "1.0.0")
    registry.register(
        FunctionSpec(
            name="numeric.aggregate",
            version="1.0.0",
            kind=FunctionKind.QUERY,
            inputs=[
                FunctionParameter(name="values", type="array", required=True),
                FunctionParameter(name="op", type="string", required=False),
            ],
            output="float",
            handler=_numeric_aggregate,
            description="Aggregate a numeric list (sum/mean/min/max/count).",
        )
    )
    registry.release("numeric.aggregate", "1.0.0")


# CONCEPT:AU-KG.ontology.default-runtime-bound-import — populated at import (a live path, not an empty shell). The
# FunctionRuntime in runtime.py binds this registry as its default lookup source.
DEFAULT_FUNCTION_REGISTRY = FunctionRegistry()
register_builtins(DEFAULT_FUNCTION_REGISTRY)


__all__ = [
    "DEFAULT_FUNCTION_REGISTRY",
    "FunctionHandler",
    "FunctionKind",
    "FunctionParameter",
    "FunctionRegistry",
    "FunctionSpec",
    "register_builtins",
    "resolve_type",
]
