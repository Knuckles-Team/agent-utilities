"""Structured-output contracts for RLM subagent calls.

CONCEPT:AU-ORCH.execution.predict-rlm-runtime — Structured Predict-RLM Runtime (subagent extension)

A Recursive Language Model degrades when subagents return free-form prose: the
parent has to re-read and re-classify dozens of unstructured blurbs, losing the
plot. The fix is to force each subagent to return a *schema-constrained* value
(e.g. a boolean relevance flag) so the parent routes on a clean typed value —
an external attention mask over the original context.

This module normalizes every schema form a caller might supply — a Pydantic
``BaseModel`` subclass, a primitive type (``int``/``bool``/``str``/``float``),
a typing generic (``list[Model]``, ``dict[str, int]``), or a raw JSON-Schema
``dict`` — into a single :class:`SchemaContract` that can both *render* the
contract for the LLM prompt and *validate* (and coerce) a returned value.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError

logger = logging.getLogger(__name__)

try:  # optional — used for raw JSON-Schema dict validation only
    import jsonschema as _jsonschema

    _HAS_JSONSCHEMA = True
except ImportError:  # pragma: no cover - exercised in the no-dep environment
    _jsonschema = None
    _HAS_JSONSCHEMA = False


@dataclass
class SchemaContract:
    """A normalized, validatable output contract for an RLM (sub)agent.

    Attributes:
        json_schema: The contract as a plain JSON Schema dict (shown to the LLM).
        json_schema_str: Pretty-printed ``json_schema`` for prompt injection.
    """

    json_schema: dict[str, Any]
    json_schema_str: str
    # One of these performs the actual validation, depending on the source spec.
    _adapter: TypeAdapter | None = None
    _model: type[BaseModel] | None = None
    _raw_schema: dict[str, Any] | None = None

    @classmethod
    def from_spec(cls, spec: Any) -> SchemaContract:
        """Normalize any supported schema spec into a :class:`SchemaContract`.

        Supported forms:
            * a Pydantic ``BaseModel`` subclass
            * a primitive type (``int``, ``bool``, ``str``, ``float``)
            * a typing generic (``list[Model]``, ``dict[str, int]``, ...)
            * a raw JSON-Schema ``dict`` (e.g. ``{"type": "boolean"}``)
        """
        # Pydantic model class — richest case, native JSON Schema + validation.
        if isinstance(spec, type) and issubclass(spec, BaseModel):
            schema = spec.model_json_schema()
            return cls(
                json_schema=schema,
                json_schema_str=json.dumps(schema, indent=2),
                _model=spec,
            )

        # Raw JSON-Schema dict — keep verbatim, validate via jsonschema if present.
        if isinstance(spec, dict):
            return cls(
                json_schema=spec,
                json_schema_str=json.dumps(spec, indent=2),
                _raw_schema=spec,
            )

        # Primitive type or typing generic — delegate to a pydantic TypeAdapter.
        try:
            adapter: TypeAdapter = TypeAdapter(spec)
            schema = adapter.json_schema()
        except Exception as exc:  # noqa: BLE001 - bad spec must surface clearly
            raise TypeError(
                f"Unsupported RLM output schema spec {spec!r}: {exc}"
            ) from exc
        return cls(
            json_schema=schema,
            json_schema_str=json.dumps(schema, indent=2),
            _adapter=adapter,
        )

    @property
    def model_type(self) -> type[BaseModel] | None:
        """The contract's Pydantic model class, if the spec was one.

        ``None`` for a primitive/generic (``TypeAdapter``) or raw JSON-Schema spec —
        there is no single Python type to hand a consumer (e.g. pydantic-ai's
        ``output_type=``) in those cases; fall back to prompting ``json_schema_str``
        and validating the result with :meth:`validate` instead.
        """
        return self._model

    def validate(self, value: Any) -> tuple[bool, Any, str | None]:
        """Validate (and coerce) ``value`` against the contract.

        Returns ``(ok, coerced_value, error)``. On success ``error`` is ``None``
        and ``coerced_value`` is the type-coerced value (e.g. ``"true"`` → a
        Python ``True`` for a boolean contract). On failure ``ok`` is ``False``
        and ``error`` is a human/LLM-readable message naming the violations.
        """
        try:
            if self._model is not None:
                return True, self._coerce_model(value), None
            if self._adapter is not None:
                return True, self._adapter.validate_python(value), None
            return self._validate_raw(value)
        except ValidationError as exc:
            return False, value, _format_pydantic_error(exc)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the REPL loop
            return False, value, str(exc)

    def _coerce_model(self, value: Any) -> Any:
        assert self._model is not None
        if isinstance(value, self._model):
            return value
        if isinstance(value, dict):
            return self._model.model_validate(value)
        if isinstance(value, str):
            return self._model.model_validate_json(value)
        return self._model.model_validate(value)

    def _validate_raw(self, value: Any) -> tuple[bool, Any, str | None]:
        """Validate against a raw JSON-Schema dict.

        Uses the ``jsonschema`` package when available. When it is absent we fall
        back to a shallow ``type``/``required`` check and log a degraded-validation
        warning — never a silent pass, which would re-introduce the failure mode
        this module exists to prevent.
        """
        assert self._raw_schema is not None
        if _HAS_JSONSCHEMA:
            _jsonschema.validate(instance=value, schema=self._raw_schema)
            return True, value, None

        logger.warning(
            "jsonschema not installed — RLM output contract %s validated shallowly "
            "(install the 'jsonschema' extra for full validation).",
            self._raw_schema.get("type", "<no-type>"),
        )
        err = _shallow_jsonschema_check(value, self._raw_schema)
        return (err is None), value, err


_JSON_TYPE_TO_PY: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": (list, tuple),
    "null": type(None),
}


def _shallow_jsonschema_check(value: Any, schema: dict[str, Any]) -> str | None:
    """Best-effort ``type``/``required`` validation when ``jsonschema`` is absent."""
    expected = schema.get("type")
    if expected:
        py = _JSON_TYPE_TO_PY.get(expected)
        # bool is a subclass of int — guard so a boolean isn't accepted as integer.
        if expected == "integer" and isinstance(value, bool):
            return "(root): must be integer, got boolean"
        if py is not None and not isinstance(value, py):
            return f"(root): must be {expected}, got {type(value).__name__}"
    if expected == "object" and isinstance(value, dict):
        missing = [k for k in schema.get("required", []) if k not in value]
        if missing:
            return f"(root): missing required keys {missing}"
    return None


def _format_pydantic_error(exc: ValidationError) -> str:
    """Render a ``ValidationError`` as ``path: message`` lines (article §3 style)."""
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
        lines.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "\n".join(lines)
