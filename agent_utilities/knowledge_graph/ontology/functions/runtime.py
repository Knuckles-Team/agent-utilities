#!/usr/bin/python
from __future__ import annotations

"""Functions runtime — typed invocation entry point (CONCEPT:KG-2.41).

Palantir Foundry ``functions/overview``: an authored Function is invoked with
typed inputs, runs, and returns a typed output; Actions and derived properties
consume those functions through one runtime. This module is that single
invocation entry for the ontology layer.

:class:`FunctionRuntime.invoke` performs the full governed cycle for one call:

  (a) resolve the :class:`FunctionSpec` from the registry (by name, optional
      pinned version; latest *released* otherwise),
  (b) validate the supplied params against the spec's typed input schema
      (missing/required, unknown keys, type mismatches),
  (c) inject the :class:`ObjectFunctionContext` for ``ON_OBJECTS`` functions so
      handlers can read/traverse the live graph,
  (d) invoke the handler,
  (e) validate & coerce the result to the declared output type,
  (f) record an :class:`~agent_utilities.observability.audit_logger.AuditLogger`
      entry (actor / function / status),
  (g) return a :class:`FunctionResult` carrying the typed value or the error.

This is the contract that Actions (CONCEPT:KG-2.25) and derived-properties will
consume — ``runtime.invoke(name, params, version=None)`` — so the I/O typing and
audit happen in exactly one place.
"""

import inspect
import logging
import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_utilities.observability.audit_logger import AuditLogger

from .objects import ObjectFunctionContext
from .registry import (
    DEFAULT_FUNCTION_REGISTRY,
    FunctionKind,
    FunctionRegistry,
    FunctionSpec,
)

logger = logging.getLogger(__name__)

FUNCTION_AUDIT = "ontology_function.invoke"
RESOURCE_FUNCTION = "function"


class FunctionResult(BaseModel):
    """The typed outcome of one :class:`FunctionRuntime` invocation. CONCEPT:KG-2.41."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = ""
    function_name: str
    version: str = ""
    actor_id: str = "system"
    ok: bool = False
    value: Any = None
    error: str = ""
    audit_ref: str = ""
    timestamp: float = Field(default_factory=time.time)

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            self.id = f"fninvoke:{self.function_name}:{uuid.uuid4().hex[:12]}"


def coerce_output(value: Any, expected: type) -> Any:
    """Coerce a handler result to the declared output type where safe.

    ``object`` (accept-anything) returns the value unchanged. Numeric widening
    (``int``→``float``) and a numeric/bool→``str`` render are applied; otherwise
    a non-matching value raises :class:`TypeError`, surfaced by the runtime as a
    typed-output error (never a silent wrong-type return).
    """
    if expected is object or value is None:
        return value
    if isinstance(value, expected) and not (
        expected in (int, float) and isinstance(value, bool)
    ):
        # Already the right type (but a bool is not an int/float).
        if expected is float and isinstance(value, int):
            return float(value)
        return value
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if expected is str and isinstance(value, int | float | bool):
        return str(value)
    raise TypeError(f"output expected {expected.__name__}, got {type(value).__name__}")


class FunctionRuntime:
    """The single governed invocation entry for typed functions. CONCEPT:KG-2.41.

    Args:
        registry: The :class:`FunctionRegistry` to resolve functions from
            (defaults to the import-populated :data:`DEFAULT_FUNCTION_REGISTRY`).
        audit: An :class:`AuditLogger`; a fresh in-memory logger is created when
            omitted.
        graph: An optional live :class:`KnowledgeGraph` facade used to build the
            :class:`ObjectFunctionContext` injected into ``ON_OBJECTS`` handlers.
        released_only: When ``True``, name-only lookups resolve *only* released
            versions (Foundry "use the published version" semantics). Pinned
            ``version=`` lookups are unaffected.
    """

    def __init__(
        self,
        registry: FunctionRegistry | None = None,
        audit: AuditLogger | None = None,
        graph: Any = None,
        released_only: bool = False,
    ) -> None:
        self.registry = registry or DEFAULT_FUNCTION_REGISTRY
        self.audit = audit or AuditLogger()
        self._graph = graph
        self.released_only = released_only
        self._object_context: ObjectFunctionContext | None = None

    @property
    def object_context(self) -> ObjectFunctionContext:
        """The lazily-built Functions-on-Objects graph context."""
        if self._object_context is None:
            self._object_context = ObjectFunctionContext(self._graph)
        return self._object_context

    def invoke(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        version: str | None = None,
        *,
        actor_id: str = "system",
    ) -> FunctionResult:
        """Validate, run, coerce, and audit one typed function call.

        Args:
            name: The function name to resolve.
            params: Keyword inputs validated against the spec's typed schema.
            version: Optional pinned semver; otherwise the latest (released)
                version is used.
            actor_id: The invoking actor (recorded in the audit entry).

        Returns:
            A :class:`FunctionResult` — ``ok=True`` with a typed ``value`` on
            success, or ``ok=False`` with a populated ``error`` on any
            resolution/validation/handler/output failure (never raises).
        """
        params = dict(params or {})
        spec = self.registry.get(name, version, released_only=self.released_only)
        if spec is None:
            return self._fail(
                name,
                version or "",
                actor_id,
                f"unknown function: {name!r}" + (f"@{version}" if version else ""),
            )

        # (b) Validate typed inputs.
        errors = spec.validate_params(params)
        if errors:
            return self._fail(spec.name, spec.version, actor_id, "; ".join(errors))

        # (c) Inject the object context for Functions-on-Objects.
        call_kwargs = dict(params)
        if spec.kind == FunctionKind.ON_OBJECTS:
            call_kwargs = self._inject_object_context(spec, call_kwargs)

        # (d) Run the handler.
        try:
            raw = spec.handler(**call_kwargs)
        except Exception as exc:  # noqa: BLE001 — surface as a typed error
            logger.warning(
                "Function handler %s@%s failed: %s", spec.name, spec.version, exc
            )
            return self._fail(
                spec.name, spec.version, actor_id, f"handler error: {exc}"
            )

        # (e) Validate & coerce the typed output.
        try:
            value = coerce_output(raw, spec.output_type())
        except TypeError as exc:
            return self._fail(spec.name, spec.version, actor_id, str(exc))

        result = FunctionResult(
            function_name=spec.name,
            version=spec.version,
            actor_id=actor_id,
            ok=True,
            value=value,
        )
        self._audit(result, status="success")
        return result

    # ── internals ──────────────────────────────────────────────────────

    def _inject_object_context(
        self, spec: FunctionSpec, call_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Pass the :class:`ObjectFunctionContext` to handlers that accept it.

        A Functions-on-Objects handler that declares a ``context`` (or ``ctx`` /
        ``objects``) parameter receives the live graph context; handlers that
        don't (pure summarizers over already-materialized props) are unaffected.
        """
        try:
            sig = inspect.signature(spec.handler)
        except (TypeError, ValueError):
            return call_kwargs
        for pname in ("context", "ctx", "objects"):
            if pname in sig.parameters and pname not in call_kwargs:
                call_kwargs[pname] = self.object_context
                break
        return call_kwargs

    def _fail(
        self, name: str, version: str, actor_id: str, error: str
    ) -> FunctionResult:
        result = FunctionResult(
            function_name=name,
            version=version,
            actor_id=actor_id,
            ok=False,
            error=error,
        )
        self._audit(result, status="error")
        return result

    def _audit(self, result: FunctionResult, status: str) -> None:
        """Emit an AuditLog entry for an invocation (never raises)."""
        record = self.audit.log(
            actor=result.actor_id,
            action=FUNCTION_AUDIT,
            resource_type=RESOURCE_FUNCTION,
            resource_id=f"{result.function_name}@{result.version}",
            details={"status": status, "error": result.error},
        )
        if record is not None:
            result.audit_ref = record.id


# CONCEPT:KG-2.41 — a default runtime bound to the import-populated registry, so
# Actions/derived-properties have a ready single invocation entry (live path).
DEFAULT_FUNCTION_RUNTIME = FunctionRuntime()


__all__ = [
    "DEFAULT_FUNCTION_RUNTIME",
    "FUNCTION_AUDIT",
    "FunctionResult",
    "FunctionRuntime",
    "coerce_output",
]
