"""The record shape a connector delivers: field -> JSON types + presence.

A shape is the unit schema drift is measured in (AU-SEC-R004). It is inferred from a
drained batch of records or restored from the approved contract, and it has a
stable content digest so a drift report can name exactly what was compared.

Only top-level fields are shaped: a nested object is one ``object`` field. The
type lattice is JSON's, with ``integer`` a subtype of ``number`` (see
:func:`covers`); ``null`` is a type like any other, so "became nullable" is a
type change the classifier can see.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

#: Domain separator of :func:`shape_digest`.
SHAPE_DIGEST_DOMAIN = "au/source-record-shape/v1"

#: The JSON types a field may carry, in canonical order.
JSON_TYPES = ("null", "boolean", "integer", "number", "string", "array", "object")

_PYTHON_TYPES: tuple[tuple[type | tuple[type, ...], str], ...] = (
    (bool, "boolean"),
    (int, "integer"),
    (float, "number"),
    (str, "string"),
    ((list, tuple), "array"),
    (Mapping, "object"),
)


@dataclass(frozen=True, slots=True)
class FieldShape:
    """One field: the JSON types seen for it and whether every record had it."""

    types: frozenset[str]
    required: bool

    def to_json(self) -> dict[str, Any]:
        ordered = [name for name in JSON_TYPES if name in self.types]
        return {"types": ordered, "required": self.required}


@dataclass(frozen=True, slots=True)
class RecordShape:
    """Field name -> :class:`FieldShape`, kept sorted by name."""

    fields: tuple[tuple[str, FieldShape], ...]

    @classmethod
    def of(cls, fields: Mapping[str, FieldShape]) -> RecordShape:
        return cls(tuple(sorted(fields.items())))

    def as_dict(self) -> dict[str, FieldShape]:
        return dict(self.fields)

    def to_json(self) -> dict[str, Any]:
        return {name: shape.to_json() for name, shape in self.fields}

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> RecordShape:
        return cls.of(
            {
                str(name): FieldShape(
                    types=frozenset(str(t) for t in spec.get("types", ())),
                    required=bool(spec.get("required")),
                )
                for name, spec in payload.items()
                if isinstance(spec, Mapping)
            }
        )


def json_type(value: Any) -> str:
    """The JSON type name of one decoded value."""
    if value is None:
        return "null"
    for python_type, name in _PYTHON_TYPES:
        if isinstance(value, python_type):
            return name
    return "string"


def infer_shape(records: Iterable[Mapping[str, Any]]) -> RecordShape:
    """The shape of a batch: a field is ``required`` when every record has it."""
    seen: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    total = 0
    for record in records:
        total += 1
        for name, value in record.items():
            seen.setdefault(str(name), set()).add(json_type(value))
            counts[str(name)] = counts.get(str(name), 0) + 1
    return RecordShape.of(
        {
            name: FieldShape(frozenset(types), counts[name] == total)
            for name, types in seen.items()
        }
    )


def covers(allowed: frozenset[str], observed: str) -> bool:
    """Whether a field declared as ``allowed`` accepts a value of ``observed``."""
    return observed in allowed or (observed == "integer" and "number" in allowed)


def merge_shapes(base: RecordShape, extra: RecordShape) -> RecordShape:
    """The union contract: every field of both, types joined, presence met."""
    merged = base.as_dict()
    for name, shape in extra.fields:
        known = merged.get(name)
        merged[name] = (
            shape
            if known is None
            else FieldShape(
                known.types | shape.types, known.required and shape.required
            )
        )
    return RecordShape.of(merged)


def shape_digest(shape: RecordShape) -> str:
    """SHA-256 over the canonical JSON of ``shape`` under its domain."""
    payload = json.dumps(shape.to_json(), sort_keys=True, separators=(",", ":"))
    framed = f"{SHAPE_DIGEST_DOMAIN}\0{payload}".encode()
    return hashlib.sha256(framed).hexdigest()


__all__ = [
    "JSON_TYPES",
    "SHAPE_DIGEST_DOMAIN",
    "FieldShape",
    "RecordShape",
    "covers",
    "infer_shape",
    "json_type",
    "merge_shapes",
    "shape_digest",
]
