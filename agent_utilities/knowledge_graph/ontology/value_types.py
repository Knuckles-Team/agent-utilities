#!/usr/bin/python
from __future__ import annotations

"""Ontology Value Types — semantic wrappers over base property types.

CONCEPT:KG-2.39 — Ontology Value Types.

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
2.  a **SHACL property shape** — :meth:`ValueType.to_shacl` emits an
    ``sh:PropertyShape`` turtle fragment (``sh:pattern``, ``sh:minInclusive`` /
    ``sh:maxInclusive``, ``sh:minLength`` / ``sh:maxLength``, ``sh:in``) so the
    existing SHACL gate
    (:class:`agent_utilities.knowledge_graph.core.shacl_validator.SHACLValidator`,
    consumed by ``security/graph_validator``) enforces the same rules at graph
    write time. :func:`write_value_shapes_ttl` materializes the whole registry
    into ``shapes/value_types.shapes.ttl`` — a file the validator loads exactly
    like ``governance.shapes.ttl``; and
3.  an **OWL datatype restriction** — :meth:`ValueType.to_owl` emits an
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
from collections.abc import Iterable
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .property_types import KG, XSD, PropertyType, parse_type_ref

# A handful of XSD constraining facets used by ``owl:withRestrictions``.
_XSD_FACETS = {
    "pattern": XSD + "pattern",
    "minInclusive": XSD + "minInclusive",
    "maxInclusive": XSD + "maxInclusive",
    "minExclusive": XSD + "minExclusive",
    "maxExclusive": XSD + "maxExclusive",
    "minLength": XSD + "minLength",
    "maxLength": XSD + "maxLength",
    "length": XSD + "length",
}


def _ttl_escape(text: str) -> str:
    """Escape a Python string for a double-quoted turtle literal."""
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _ttl_number(value: int | float | Decimal) -> str:
    """Render a numeric bound as a typed turtle literal."""
    if isinstance(value, bool):  # defensive — bool is an int subclass
        raise ValueError("numeric bound cannot be a boolean")
    if isinstance(value, int):
        return f'"{value}"^^xsd:integer'
    # float / Decimal → xsd:decimal (avoids float-repr surprises in turtle)
    return f'"{Decimal(str(value))}"^^xsd:decimal'


class ValueConstraints(BaseModel):
    """The constraint block of a value type (CONCEPT:KG-2.39).

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

    CONCEPT:KG-2.39 — Ontology Value Types.

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

    def validate(self, value: Any) -> bool:
        """Return True iff ``value`` coerces and satisfies every constraint."""
        try:
            self.coerce(value)
            return True
        except (ValueError, TypeError, InvalidOperation):
            return False

    def _check_constraints(self, value: Any) -> None:
        c = self.constraints

        # Enumeration / allowed values (compared post-coercion).
        if c.allowed_values is not None:
            if c.case_insensitive and isinstance(value, str):
                allowed = {
                    str(a).lower() for a in c.allowed_values if isinstance(a, str)
                }
                allowed |= {a for a in c.allowed_values if not isinstance(a, str)}
                ok = value.lower() in allowed or value in c.allowed_values
            else:
                ok = value in c.allowed_values
            if not ok:
                raise ValueError(
                    f"{value!r} is not one of the allowed values for {self.name}"
                )

        # Regex pattern (string-valued types).
        if c.pattern is not None:
            flags = re.IGNORECASE if c.case_insensitive else 0
            if not isinstance(value, str):
                raise ValueError(
                    f"{self.name} pattern applies to strings, got "
                    f"{type(value).__name__}"
                )
            if re.fullmatch(c.pattern, value, flags) is None:
                raise ValueError(
                    f"{value!r} does not match pattern for {self.name}"
                )

        # Length bounds (string length or array element count).
        if c.min_length is not None or c.max_length is not None:
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

        # Numeric bounds.
        if c.min_value is not None or c.max_value is not None:
            num = self._as_number(value)
            if c.min_value is not None:
                if c.exclusive_min:
                    if not num > c.min_value:
                        raise ValueError(
                            f"{self.name}: {num} not > min {c.min_value}"
                        )
                elif num < c.min_value:
                    raise ValueError(
                        f"{self.name}: {num} < min {c.min_value}"
                    )
            if c.max_value is not None:
                if c.exclusive_max:
                    if not num < c.max_value:
                        raise ValueError(
                            f"{self.name}: {num} not < max {c.max_value}"
                        )
                elif num > c.max_value:
                    raise ValueError(
                        f"{self.name}: {num} > max {c.max_value}"
                    )

    @staticmethod
    def _measurable_length(value: Any) -> int | None:
        if isinstance(value, (str, list, tuple, bytes, set, dict)):
            return len(value)
        return None

    @staticmethod
    def _as_number(value: Any) -> float:
        if isinstance(value, bool):
            raise ValueError("boolean is not a numeric value")
        if isinstance(value, (int, float, Decimal)):
            return float(value)
        if isinstance(value, (_dt.date, _dt.datetime)):
            # Allow numeric bounds on temporal values via ordinal/epoch ordering.
            if isinstance(value, _dt.datetime):
                return value.timestamp()
            return float(value.toordinal())
        if isinstance(value, str):
            return float(value)
        raise ValueError(f"{value!r} is not numeric for a min/max constraint")

    # -- SHACL property shape ----------------------------------------------
    def _shacl_property_lines(self, *, indent: str = "        ") -> list[str]:
        """Constraint lines for an ``sh:PropertyShape`` body (no path/closing)."""
        c = self.constraints
        pt = self.property_type
        lines = [f"{indent}sh:datatype <{pt.xsd_iri}> ;"]
        if c.pattern is not None:
            lines.append(f'{indent}sh:pattern "{_ttl_escape(c.pattern)}" ;')
            if c.case_insensitive:
                lines.append(f'{indent}sh:flags "i" ;')
        if c.min_value is not None:
            facet = "sh:minExclusive" if c.exclusive_min else "sh:minInclusive"
            lines.append(f"{indent}{facet} {_ttl_number(c.min_value)} ;")
        if c.max_value is not None:
            facet = "sh:maxExclusive" if c.exclusive_max else "sh:maxInclusive"
            lines.append(f"{indent}{facet} {_ttl_number(c.max_value)} ;")
        if c.min_length is not None:
            lines.append(f'{indent}sh:minLength "{c.min_length}"^^xsd:integer ;')
        if c.max_length is not None:
            lines.append(f'{indent}sh:maxLength "{c.max_length}"^^xsd:integer ;')
        if c.allowed_values is not None:
            members = " ".join(self._ttl_value(v) for v in c.allowed_values)
            lines.append(f"{indent}sh:in ( {members} ) ;")
        if c.unit:
            lines.append(f'{indent}qudt:unit "{_ttl_escape(c.unit)}" ;')
        return lines

    def _ttl_value(self, value: Any) -> str:
        """Render an enumeration member as a typed turtle literal."""
        if isinstance(value, bool):
            return f'"{str(value).lower()}"^^xsd:boolean'
        if isinstance(value, int):
            return f'"{value}"^^xsd:integer'
        if isinstance(value, float):
            return f'"{value}"^^xsd:double'
        if isinstance(value, Decimal):
            return f'"{value}"^^xsd:decimal'
        return f'"{_ttl_escape(str(value))}"'

    def to_shacl(self, *, path: str | None = None, target_class: str | None = None) -> str:
        """Emit a SHACL turtle fragment enforcing this value type.

        CONCEPT:KG-2.39 — by default emits a reusable ``sh:PropertyShape``
        (``:<Name>ValueShape``) carrying the constraints; callers attach it to a
        property via ``sh:property`` or reuse it with ``sh:node``. When ``path``
        (and optionally ``target_class``) is given it emits a node shape that
        binds the constraints to that property path — the form the SHACL gate
        validates directly.

        The emitted turtle uses the same prefixes bound in
        ``shapes/governance.shapes.ttl`` (``:`` → ``http://knuckles.team/kg#``,
        ``sh:``, ``xsd:``).
        """
        shape_iri = f":{self.name}ValueShape"
        desc = _ttl_escape(self.description or f"{self.name} value type.")
        # Constraint lines at 4-space indent for top-level predicates.
        body = self._shacl_property_lines(indent="    ")
        if path is None:
            # Reusable property-constraint shape (no sh:path). The trailing ' ;'
            # of the last constraint line is replaced by ' .' to close the shape.
            head = (
                f"{shape_iri} a sh:PropertyShape ;\n"
                f'    sh:name "{self.name}" ;\n'
                f'    sh:description "{desc}" ;\n'
            )
            constraint_block = "\n".join(body)
            assert constraint_block.endswith(" ;")
            constraint_block = constraint_block[: -len(" ;")] + " ."
            return head + constraint_block + "\n"
        node_iri = f":{self.name}Shape"
        # Re-render constraint lines at the deeper indent of the property node.
        prop_lines = "\n".join(self._shacl_property_lines(indent="        "))
        prop_block = (
            "    sh:property [\n"
            f"        sh:path :{path} ;\n"
            f"{prop_lines}\n"
            f'        sh:message "Value violates the {self.name} value type." ;\n'
            "    ] ;"
        )
        target = (
            f"    sh:targetClass :{target_class} ;\n" if target_class else ""
        )
        return (
            f"{node_iri} a sh:NodeShape ;\n"
            f'    sh:name "{self.name} Shape" ;\n'
            f'    sh:description "{desc}" ;\n'
            f"{target}"
            f"{prop_block}\n"
            f"    sh:closed false .\n"
        )

    # -- OWL datatype restriction ------------------------------------------
    def to_owl(self) -> str:
        """Emit an ``rdfs:Datatype`` defined by an OWL facet restriction.

        CONCEPT:KG-2.39 — the value type becomes a named ``rdfs:Datatype``
        (``:<Name>``) equivalent to its base XSD datatype restricted by the
        constraint facets (``owl:withRestrictions``). Enum value types instead
        emit an ``owl:oneOf`` data range. This is the RDF/OWL-substrate form
        consumed by ``owl_bridge`` reasoning.
        """
        c = self.constraints
        base_iri = self.property_type.xsd_iri
        comment = _ttl_escape(self.description or f"{self.name} value type.")
        header = (
            f":{self.name} a rdfs:Datatype ;\n"
            f'    rdfs:label "{self.name}" ;\n'
            f'    rdfs:comment "{comment}" ;\n'
        )

        # Pure enumeration → owl:oneOf data range.
        if c.allowed_values is not None and c.is_empty() is False and not (
            c.pattern or c.min_value is not None or c.max_value is not None
            or c.min_length is not None or c.max_length is not None
        ):
            members = " ".join(self._ttl_value(v) for v in c.allowed_values)
            return header + f"    owl:equivalentClass [\n        a rdfs:Datatype ;\n        owl:oneOf ( {members} )\n    ] .\n"

        facets: list[str] = []
        if c.pattern is not None:
            facets.append(f'        [ xsd:pattern "{_ttl_escape(c.pattern)}" ]')
        if c.min_value is not None:
            key = "minExclusive" if c.exclusive_min else "minInclusive"
            facets.append(f"        [ {_owl_facet(key, c.min_value)} ]")
        if c.max_value is not None:
            key = "maxExclusive" if c.exclusive_max else "maxInclusive"
            facets.append(f"        [ {_owl_facet(key, c.max_value)} ]")
        if c.min_length is not None:
            facets.append(f'        [ xsd:minLength "{c.min_length}"^^xsd:integer ]')
        if c.max_length is not None:
            facets.append(f'        [ xsd:maxLength "{c.max_length}"^^xsd:integer ]')

        if not facets:
            # No restrictable facet → plain subtype of the base datatype.
            return header + f"    owl:equivalentClass <{base_iri}> .\n"

        restriction = " ".join(facets).strip()
        return (
            header
            + "    owl:equivalentClass [\n"
            + "        a rdfs:Datatype ;\n"
            + f"        owl:onDatatype <{base_iri}> ;\n"
            + "        owl:withRestrictions (\n"
            + "\n".join(facets)
            + "\n        )\n    ] .\n"
        )


def _owl_facet(key: str, value: int | float) -> str:
    if key not in _XSD_FACETS:  # pragma: no cover - defensive
        raise ValueError(f"unknown OWL facet {key!r}")
    return f"xsd:{key} {_ttl_number(value)}"


# ---------------------------------------------------------------------------
# Turtle prefix header (matches governance.shapes.ttl bindings)
# ---------------------------------------------------------------------------
SHAPES_PREFIXES = (
    "@prefix : <http://knuckles.team/kg#> .\n"
    "@prefix sh: <http://www.w3.org/ns/shacl#> .\n"
    "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
    "@prefix qudt: <http://qudt.org/schema/qudt/> .\n"
    "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n"
)


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


def value_types_shapes_ttl(
    value_types: Iterable[ValueType] | None = None,
) -> str:
    """Render the registry as one SHACL shapes turtle document.

    CONCEPT:KG-2.39 — concatenates the reusable ``sh:PropertyShape`` fragment for
    every value type under the shared prefix header, producing a turtle file the
    SHACL gate (``SHACLValidator``) loads exactly like ``governance.shapes.ttl``.
    """
    vts = list(value_types) if value_types is not None else [
        VALUE_TYPES[n] for n in list_value_types()
    ]
    parts = [SHAPES_PREFIXES, ""]
    for vt in vts:
        parts.append(vt.to_shacl())
    return "\n".join(parts)


def value_types_owl_ttl(
    value_types: Iterable[ValueType] | None = None,
) -> str:
    """Render the registry as one OWL datatype-restriction turtle document.

    CONCEPT:KG-2.39 — each value type becomes a named ``rdfs:Datatype`` restricted
    by its facets, under the shared prefix header, for the ``owl_bridge`` substrate.
    """
    vts = list(value_types) if value_types is not None else [
        VALUE_TYPES[n] for n in list_value_types()
    ]
    parts = [SHAPES_PREFIXES, ""]
    for vt in vts:
        parts.append(vt.to_owl())
    return "\n".join(parts)


def write_value_shapes_ttl(target_path: str | None = None) -> str:
    """Materialize the value-type SHACL shapes to ``shapes/value_types.shapes.ttl``.

    CONCEPT:KG-2.39 — the live consumer hook. Writes the combined SHACL document
    next to ``governance.shapes.ttl`` so the existing SHACL gate picks it up. The
    file location is returned. Idempotent: re-running rewrites the same content.

    Args:
        target_path: Override path; defaults to the package ``shapes`` directory.
    """
    from pathlib import Path

    if target_path is None:
        shapes_dir = Path(__file__).resolve().parent.parent / "shapes"
        shapes_dir.mkdir(parents=True, exist_ok=True)
        path = shapes_dir / "value_types.shapes.ttl"
    else:
        path = Path(target_path)
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value_types_shapes_ttl(), encoding="utf-8")
    return str(path)


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
    "value_types_shapes_ttl",
    "value_types_owl_ttl",
    "write_value_shapes_ttl",
]
