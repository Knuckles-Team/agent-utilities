#!/usr/bin/python
from __future__ import annotations

"""Ontology Value Types — semantic wrappers over base property types.

CONCEPT:AU-KG.ontology.value-type-shacl-load — Ontology Value Types.

Palantir Foundry doc matched: ontology *"Value types"* — a value type is a
**semantic wrapper around a base (field) data type** that attaches *metadata and
constraints* (regex/pattern, numeric min/max, length bounds, an allowed-value
enumeration, and unit/format display metadata). In Foundry a value type such as
``EmailAddress`` or ``ISOCurrencyCode`` is reusable across many object
properties: every property declared with that value type inherits the same
validation, so the constraint is authored once and enforced everywhere.

This module ports that abstraction onto the existing agent-utilities fabric.
A :class:`ValueType` is a named wrapper over a concrete
:class:`~agent_utilities.knowledge_graph.ontology.property_types.PropertyType`
(the Stage-A base-type registry) plus a :class:`ValueConstraints` block. From a
single declaration it compiles to **three coupled artifacts** so the constraint
is enforced on every layer the platform already runs:

1.  a **runtime validator** — :meth:`ValueType.validate` / :meth:`ValueType.coerce`
    first coerce through the base ``PropertyType`` (so an ``ISOCurrencyCode`` is
    a real string, a ``Percentage`` a real float) and then apply the constraints;
2.  a **SHACL shape** — :meth:`ValueType.to_shacl` asks EG to compile a reusable
    ``sh:NodeShape`` turtle fragment (``sh:pattern``, ``sh:minInclusive`` /
    ``sh:maxInclusive``, ``sh:minLength`` / ``sh:maxLength``, ``sh:in``) so the
    committed epistemic-graph SHACL gate enforces the same rules at graph write
    time. The shape document is submitted through EG's GraphSchema pack authority; and
3.  an **OWL datatype restriction** — :meth:`ValueType.to_owl` asks EG to compile an
    ``rdfs:Datatype`` defined by an ``owl:withRestrictions`` facet list
    (``xsd:pattern`` / ``xsd:minInclusive`` / … ) over the base XSD datatype, so
    the value type round-trips into the ``owl_bridge`` RDF/OWL substrate.

The module follows the import-populated-registry idiom: :data:`VALUE_TYPES` is
populated at import with real built-ins (``EmailAddress``, ``ISOCurrencyCode``,
``Percentage``, ``URL``, ``E164PhoneNumber``, ``Probability``), never an empty
shell.
"""

import datetime as _dt
import re
from collections.abc import Callable, Iterable
from decimal import Decimal, InvalidOperation
from typing import Any

from epistemic_graph.value_type_pack import (
    VALUE_TYPE_PREFIXES as SHAPES_PREFIXES,
)
from epistemic_graph.value_type_pack import (
    compile_value_type_owl,
    compile_value_type_shape,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .property_types import KG, PropertyType, parse_type_ref


class ValueConstraints(BaseModel):
    """The constraint block of a value type (CONCEPT:AU-KG.ontology.value-type-shacl-load).

    Palantir doc matched: the *constraints/metadata* a value type layers over its
    base type. Each field is optional; only the populated ones compile into the
    runtime check, the SHACL shape and the OWL datatype restriction.

    Attributes:
        pattern: A regex the (string) value must fully match.
        min_value / max_value: Inclusive numeric bounds.
        exclusive_min / exclusive_max: When True, the corresponding bound is
            treated as exclusive (``>`` / ``<`` rather than ``>=`` / ``<=``).
        min_length / max_length: Inclusive length bounds (string length, or
            element count for array base types).
        allowed_values: An explicit enumeration of permitted values; membership
            is checked after base coercion.
        unit: A unit-of-measure tag (e.g. ``percent``, ``USD``) — display/semantic
            metadata, surfaced in OWL/SHACL as an annotation.
        format: A display/format hint (e.g. ``email``, ``uri``).
        case_insensitive: When True, the regex match and enum membership ignore
            case.
    """

    model_config = ConfigDict(frozen=False)

    pattern: str | None = None
    min_value: float | int | None = None
    max_value: float | int | None = None
    exclusive_min: bool = False
    exclusive_max: bool = False
    min_length: int | None = None
    max_length: int | None = None
    allowed_values: list[Any] | None = None
    unit: str | None = None
    format: str | None = None
    case_insensitive: bool = False

    @field_validator("pattern")
    @classmethod
    def _check_pattern_compiles(cls, v: str | None) -> str | None:
        if v is not None:
            re.compile(v)  # raises re.error early on a bad pattern
        return v

    def is_empty(self) -> bool:
        """True when no constraint is declared (pure type alias)."""
        return not any(
            x is not None
            for x in (
                self.pattern,
                self.min_value,
                self.max_value,
                self.min_length,
                self.max_length,
                self.allowed_values,
            )
        )


class ValueType(BaseModel):
    """A named semantic wrapper over a base property type + constraints.

    CONCEPT:AU-KG.ontology.value-type-shacl-load — Ontology Value Types.

    Palantir doc matched: ontology *Value types*. Binds a logical, reusable name
    (e.g. ``EmailAddress``) to a base
    :class:`~...ontology.property_types.PropertyType` and a
    :class:`ValueConstraints` block, then compiles that single declaration into a
    runtime validator (:meth:`validate` / :meth:`coerce`), a SHACL property shape
    (:meth:`to_shacl`) and an OWL datatype restriction (:meth:`to_owl`).

    Attributes:
        name: The value-type name (PascalCase, used as the OWL datatype /
            SHACL shape local name).
        base_type: A property-type reference resolved through
            :func:`...property_types.parse_type_ref` (e.g. ``string``, ``double``,
            ``decimal``).
        constraints: The :class:`ValueConstraints` layered over the base type.
        description: Human/LLM-facing description (becomes ``sh:description`` /
            ``rdfs:comment``).
        examples: Illustrative conforming values (documentation only).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)

    name: str
    base_type: str = "string"
    constraints: ValueConstraints = Field(default_factory=ValueConstraints)
    description: str = ""
    examples: list[Any] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not v or not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", v):
            raise ValueError(f"value type name {v!r} must be an identifier")
        return v

    # -- base resolution ----------------------------------------------------
    @property
    def property_type(self) -> PropertyType:
        """The resolved base :class:`PropertyType`."""
        return parse_type_ref(self.base_type)

    # -- runtime validation -------------------------------------------------
    def coerce(self, value: Any) -> Any:
        """Coerce ``value`` through the base type, then enforce constraints.

        Raises:
            ValueError: if base coercion fails or any constraint is violated.
        """
        coerced = self.property_type.coerce(value)
        self._check_constraints(coerced)
        return coerced

    def validate(self, value: Any) -> bool:  # type: ignore[override]  # domain check, not pydantic's deprecated validate
        """Return True iff ``value`` coerces and satisfies every constraint."""
        try:
            self.coerce(value)
            return True
        except (ValueError, TypeError, InvalidOperation):
            return False

    def _check_constraints(self, value: Any) -> None:
        self._check_allowed_values(value)
        self._check_pattern(value)
        self._check_length_constraints(value)
        self._check_numeric_constraints(value)

    def _check_allowed_values(self, value: Any) -> None:
        """Enforce an enumeration after base-type coercion."""
        c = self.constraints
        if c.allowed_values is None:
            return
        if not self._allowed_value_matches(value):
            raise ValueError(
                f"{value!r} is not one of the allowed values for {self.name}"
            )

    def _allowed_value_matches(self, value: Any) -> bool:
        """Return whether ``value`` matches the configured enumeration."""
        c = self.constraints
        if not c.case_insensitive or not isinstance(value, str):
            return value in (c.allowed_values or [])
        allowed_values = c.allowed_values or []
        allowed = self._case_insensitive_allowed_values(allowed_values)
        return value.lower() in allowed or value in allowed_values

    @staticmethod
    def _case_insensitive_allowed_values(allowed_values: list[Any]) -> set[Any]:
        """Normalize string members while retaining non-string members."""
        allowed = {
            str(item).lower() for item in allowed_values if isinstance(item, str)
        }
        allowed.update(item for item in allowed_values if not isinstance(item, str))
        return allowed

    def _check_pattern(self, value: Any) -> None:
        """Enforce the optional regular expression on string values."""
        c = self.constraints
        if c.pattern is None:
            return
        flags = re.IGNORECASE if c.case_insensitive else 0
        if not isinstance(value, str):
            raise ValueError(
                f"{self.name} pattern applies to strings, got {type(value).__name__}"
            )
        if re.fullmatch(c.pattern, value, flags) is None:
            raise ValueError(f"{value!r} does not match pattern for {self.name}")

    def _check_length_constraints(self, value: Any) -> None:
        """Enforce optional string or collection length bounds."""
        c = self.constraints
        if c.min_length is None and c.max_length is None:
            return
        length = self._measurable_length(value)
        if length is None:
            raise ValueError(
                f"{self.name} length constraint applies to sized values, "
                f"got {type(value).__name__}"
            )
        if c.min_length is not None and length < c.min_length:
            raise ValueError(
                f"{self.name}: length {length} < min_length {c.min_length}"
            )
        if c.max_length is not None and length > c.max_length:
            raise ValueError(
                f"{self.name}: length {length} > max_length {c.max_length}"
            )

    def _check_numeric_constraints(self, value: Any) -> None:
        """Enforce optional inclusive or exclusive numeric bounds."""
        c = self.constraints
        if c.min_value is None and c.max_value is None:
            return
        num = self._as_number(value)
        self._check_numeric_bound(
            num,
            c.min_value,
            minimum=True,
            exclusive=c.exclusive_min,
        )
        self._check_numeric_bound(
            num,
            c.max_value,
            minimum=False,
            exclusive=c.exclusive_max,
        )

    def _check_numeric_bound(
        self,
        value: float,
        bound: float | int | None,
        *,
        minimum: bool,
        exclusive: bool,
    ) -> None:
        """Raise when ``value`` violates one numeric bound."""
        if bound is None:
            return
        if minimum:
            invalid = value <= bound if exclusive else value < bound
            detail = f"not > min {bound}" if exclusive else f"< min {bound}"
        else:
            invalid = value >= bound if exclusive else value > bound
            detail = f"not < max {bound}" if exclusive else f"> max {bound}"
        if invalid:
            raise ValueError(f"{self.name}: {value} {detail}")

    @staticmethod
    def _measurable_length(value: Any) -> int | None:
        if isinstance(value, str | list | tuple | bytes | set | dict):
            return len(value)
        return None

    @staticmethod
    def _as_number(value: Any) -> float:
        if isinstance(value, bool):
            raise ValueError("boolean is not a numeric value")
        if isinstance(value, int | float | Decimal):
            return float(value)
        if isinstance(value, _dt.date | _dt.datetime):
            # Allow numeric bounds on temporal values via ordinal/epoch ordering.
            if isinstance(value, _dt.datetime):
                return value.timestamp()
            return float(value.toordinal())
        if isinstance(value, str):
            return float(value)
        raise ValueError(f"{value!r} is not numeric for a min/max constraint")

    def _semantic_declaration(self) -> dict[str, Any]:
        """The typed input for EG's value-type pack compiler."""
        return {
            "name": self.name,
            "description": self.description,
            "base_iri": self.property_type.xsd_iri,
            "constraints": self.constraints.model_dump(),
        }

    def to_shacl(
        self, *, path: str | None = None, target_class: str | None = None
    ) -> str:
        """Compile this declaration as an EG-owned SHACL shape."""
        return compile_value_type_shape(
            self._semantic_declaration(), path=path, target_class=target_class
        )

    def to_owl(self) -> str:
        """Compile this declaration as an EG-owned OWL datatype restriction."""
        return compile_value_type_owl(self._semantic_declaration())


# ---------------------------------------------------------------------------
# Built-in value-type registry (populated at import — never an empty shell)
# ---------------------------------------------------------------------------
def _vt(
    name: str,
    base_type: str,
    *,
    description: str = "",
    examples: Iterable[Any] = (),
    **constraint_kwargs: Any,
) -> ValueType:
    return ValueType(
        name=name,
        base_type=base_type,
        description=description,
        examples=list(examples),
        constraints=ValueConstraints(**constraint_kwargs),
    )


# RFC-5322-pragmatic email regex (the form Foundry/most validators use).
_EMAIL_RE = r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
# http(s) URL.
_URL_RE = r"https?://[^\s/$.?#].[^\s]*"
# ISO-4217 currency code (three uppercase letters).
_ISO_CCY_RE = r"[A-Z]{3}"
# E.164 international phone number.
_E164_RE = r"\+[1-9]\d{6,14}"


VALUE_TYPES: dict[str, ValueType] = {
    "EmailAddress": _vt(
        "EmailAddress",
        "string",
        description="An RFC-5322 email address — string semantically typed as an address.",
        examples=["ops@knuckles.team"],
        pattern=_EMAIL_RE,
        max_length=254,
        format="email",
    ),
    "URL": _vt(
        "URL",
        "string",
        description="An absolute http(s) URL.",
        examples=["https://knuckles.team/kg"],
        pattern=_URL_RE,
        max_length=2048,
        format="uri",
    ),
    "ISOCurrencyCode": _vt(
        "ISOCurrencyCode",
        "string",
        description="An ISO-4217 three-letter currency code.",
        examples=["USD", "EUR", "JPY"],
        pattern=_ISO_CCY_RE,
        min_length=3,
        max_length=3,
        format="iso-4217",
    ),
    "E164PhoneNumber": _vt(
        "E164PhoneNumber",
        "string",
        description="An E.164 international phone number (leading '+', up to 15 digits).",
        examples=["+14155550123"],
        pattern=_E164_RE,
        max_length=16,
        format="e164",
    ),
    "Percentage": _vt(
        "Percentage",
        "double",
        description="A percentage in [0, 100].",
        examples=[0.0, 42.5, 100.0],
        min_value=0,
        max_value=100,
        unit="percent",
    ),
    "Probability": _vt(
        "Probability",
        "double",
        description="A probability in the closed unit interval [0, 1].",
        examples=[0.0, 0.5, 1.0],
        min_value=0,
        max_value=1,
        unit="ratio",
    ),
    # ARA — confidence attached to a /logic claim (CONCEPT:AU-KG.ontology.verified-by-implemented-by). A SHACL value
    # type so a claim's confidence is constraint-checked at the Seal's L1 gate.
    "ClaimConfidence": _vt(
        "ClaimConfidence",
        "double",
        description="Confidence a research claim is substantiated, in [0, 1].",
        examples=[0.0, 0.7, 1.0],
        min_value=0,
        max_value=1,
        unit="ratio",
    ),
    # CONCEPT:AU-KG.ontology.sampling-profile-coupling — LLM inference sampling knobs temperature top_p top_k min_p repetition_penalty max_tokens and penalties as SHACL-bounded ontology value types plus the two-surface sampling-profile tool.
    # A SamplingProfile (the InferenceProfile interface, KG-2.95) is SHACL-checked at the
    # graph write gate before the AHE-3.38 loop can promote it. Bounds mirror the
    # OpenAI/vLLM accepted ranges and the SamplingProfile pydantic field constraints.
    "Temperature": _vt(
        "Temperature",
        "double",
        description="LLM sampling temperature in [0, 2]; higher = more random.",
        examples=[0.0, 0.7, 2.0],
        min_value=0,
        max_value=2,
        unit="sampling",
    ),
    "TopP": _vt(
        "TopP",
        "double",
        description="Nucleus-sampling cumulative-probability cutoff in [0, 1].",
        examples=[0.8, 0.95, 1.0],
        min_value=0,
        max_value=1,
        unit="sampling",
    ),
    "MinP": _vt(
        "MinP",
        "double",
        description="vLLM min-p relative-probability floor in [0, 1].",
        examples=[0.0, 0.05],
        min_value=0,
        max_value=1,
        unit="sampling",
    ),
    "TopK": _vt(
        "TopK",
        "integer",
        description="vLLM top-k truncation; number of highest-probability tokens kept (>=1).",
        examples=[20, 40],
        min_value=1,
        unit="sampling",
    ),
    "RepetitionPenalty": _vt(
        "RepetitionPenalty",
        "double",
        description="vLLM repetition penalty (>0); 1.0 = no penalty.",
        examples=[1.0, 1.1, 1.5],
        min_value=0,
        exclusive_min=True,
        unit="sampling",
    ),
    "MaxTokens": _vt(
        "MaxTokens",
        "integer",
        description="Maximum tokens generated in one turn (>=1).",
        examples=[1024, 16384],
        min_value=1,
        unit="sampling",
    ),
    "PresencePenalty": _vt(
        "PresencePenalty",
        "double",
        description="OpenAI presence penalty in [-2, 2].",
        examples=[0.0, 0.3, 1.5],
        min_value=-2,
        max_value=2,
        unit="sampling",
    ),
    "FrequencyPenalty": _vt(
        "FrequencyPenalty",
        "double",
        description="OpenAI frequency penalty in [-2, 2].",
        examples=[0.0, 0.5],
        min_value=-2,
        max_value=2,
        unit="sampling",
    ),
}


def get_value_type(name: str) -> ValueType | None:
    """Return the :class:`ValueType` registered under ``name`` (or None)."""
    return VALUE_TYPES.get(name)


def register_value_type(vt: ValueType, *, overwrite: bool = False) -> ValueType:
    """Register ``vt`` in :data:`VALUE_TYPES`.

    Raises:
        ValueError: if a different value type is already registered under the
            same name and ``overwrite`` is False.
    """
    existing = VALUE_TYPES.get(vt.name)
    if existing is not None and not overwrite and existing != vt:
        raise ValueError(f"value type {vt.name!r} already registered")
    VALUE_TYPES[vt.name] = vt
    return vt


def list_value_types() -> list[str]:
    """Return all registered value-type names, sorted."""
    return sorted(VALUE_TYPES.keys())


def coerce_value_type(name: str, value: Any) -> Any:
    """Resolve ``name`` and coerce ``value`` through it in one call."""
    vt = VALUE_TYPES.get(name)
    if vt is None:
        raise ValueError(f"unknown value type {name!r}")
    return vt.coerce(value)


def validate_value_type(name: str, value: Any) -> bool:
    """Resolve ``name`` and validate ``value`` through it in one call."""
    vt = VALUE_TYPES.get(name)
    if vt is None:
        return False
    return vt.validate(value)


# CONCEPT:AU-KG.ontology.sampling-profile-coupling — the coupling from a SamplingProfile's knobs to the value types
# that bound them. The single source the governance gate (the ontology set action and
# the AHE-3.38 promotion) uses to SHACL-check a profile before it is accepted/published.
INFERENCE_KNOB_VALUE_TYPES: dict[str, str] = {
    "temperature": "Temperature",
    "top_p": "TopP",
    "min_p": "MinP",
    "top_k": "TopK",
    "repetition_penalty": "RepetitionPenalty",
    "max_tokens": "MaxTokens",
    "presence_penalty": "PresencePenalty",
    "frequency_penalty": "FrequencyPenalty",
}


def sampling_profile_violations(profile: dict[str, Any]) -> list[str]:
    """Return value-type violations for a sampling-profile dict (CONCEPT:AU-KG.ontology.sampling-profile-coupling).

    Validates each present inference knob against its bounding value type
    (:data:`INFERENCE_KNOB_VALUE_TYPES`). Empty list = the profile conforms to the
    ontology bounds and may be published. ``None`` knobs (inherit-from-base) are
    skipped. This is the ontology governance gate the profile passes before the
    evolution loop promotes it or the operator sets it.
    """
    violations: list[str] = []
    for knob, vt_name in INFERENCE_KNOB_VALUE_TYPES.items():
        value = profile.get(knob)
        if value is None:
            continue
        if not validate_value_type(vt_name, value):
            violations.append(f"{knob}={value!r} violates {vt_name}")
    return violations


def _value_types_ttl(
    value_types: Iterable[ValueType] | None,
    render: Callable[[ValueType], str],
) -> str:
    vts = (
        list(value_types)
        if value_types is not None
        else [VALUE_TYPES[name] for name in list_value_types()]
    )
    return "\n".join([SHAPES_PREFIXES, "", *(render(value_type) for value_type in vts)])


def value_types_shapes_ttl(
    value_types: Iterable[ValueType] | None = None,
) -> str:
    """Render the registry as one SHACL shapes turtle document.

    CONCEPT:AU-KG.ontology.value-type-shacl-load — concatenates the reusable ``sh:NodeShape`` fragment for
    every value type under the shared prefix header for provisioning as an
    epistemic-graph GraphSchema pack source.
    """
    return _value_types_ttl(value_types, lambda value_type: value_type.to_shacl())


def value_types_owl_ttl(
    value_types: Iterable[ValueType] | None = None,
) -> str:
    """Render the registry as one OWL datatype-restriction turtle document.

    CONCEPT:AU-KG.ontology.value-type-shacl-load — each value type becomes a named ``rdfs:Datatype`` restricted
    by its facets, under the shared prefix header, for the ``owl_bridge`` substrate.
    """
    return _value_types_ttl(value_types, ValueType.to_owl)


def write_value_shapes_ttl(target_path: str | None = None) -> str:
    """Reject AU-owned shape files; publish through the SDK/EG pack authority."""
    raise RuntimeError(
        "AU no longer writes ontology shapes; publish typed declarations "
        "through the SDK ConnectorContent pack and EG GraphSchema"
    )


# KG namespace re-export so consumers can build value-type IRIs.
VALUE_TYPE_NS = KG


__all__ = [
    "ValueConstraints",
    "ValueType",
    "VALUE_TYPES",
    "VALUE_TYPE_NS",
    "SHAPES_PREFIXES",
    "get_value_type",
    "register_value_type",
    "list_value_types",
    "coerce_value_type",
    "validate_value_type",
    "INFERENCE_KNOB_VALUE_TYPES",
    "sampling_profile_violations",
    "value_types_shapes_ttl",
    "value_types_owl_ttl",
    "write_value_shapes_ttl",
]
