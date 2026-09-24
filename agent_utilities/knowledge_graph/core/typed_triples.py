"""Typed triples for Epistemic Graph validation (EH-472).

Agent Utilities owns no RDF syntax or RDF library. A caller describes the facts it
wants validated as plain ``(subject, predicate, object)`` values and sends them in
EG's ``ShaclValidate.data_triples``; the engine alone turns them into RDF terms and
validates them against its committed GraphSchema shapes. Python scalars map onto
the XSD datatypes the engine expects (``bool`` -> ``xsd:boolean``, ``int`` ->
``xsd:integer``, ``float`` -> ``xsd:double``, ``str`` -> a plain string).
"""

from __future__ import annotations

from typing import Any

KG_NS = "http://knuckles.team/kg#"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"

TypedTriple = dict[str, Any]

_SCALAR_DATATYPES: tuple[tuple[type, str], ...] = (
    (bool, "boolean"),
    (int, "integer"),
    (float, "double"),
)


def kg(local: str) -> str:
    """The ``kg#`` IRI of a local name."""
    return f"{KG_NS}{local}"


def iri(value: str) -> dict[str, str]:
    """An IRI object."""
    return {"kind": "iri", "iri": value}


def literal(value: Any, datatype: str | None = None) -> dict[str, Any]:
    """A literal object; ``datatype`` (an XSD local name) overrides the inferred one."""
    inferred = next(
        (name for kind, name in _SCALAR_DATATYPES if isinstance(value, kind)), None
    )
    lexical = str(value).lower() if isinstance(value, bool) else str(value)
    chosen = datatype or inferred
    return {
        "kind": "literal",
        "lexical": lexical,
        "datatype": f"{XSD_NS}{chosen}" if chosen else None,
        "language": None,
    }


def triple(subject: str, predicate: str, obj: dict[str, Any]) -> TypedTriple:
    """One typed triple."""
    return {"subject": subject, "predicate": predicate, "object": obj}


def typed(subject: str, class_iri: str) -> TypedTriple:
    """``subject rdf:type class_iri``."""
    return triple(subject, RDF_TYPE, iri(class_iri))


__all__ = [
    "KG_NS",
    "RDF_TYPE",
    "TypedTriple",
    "XSD_NS",
    "iri",
    "kg",
    "literal",
    "triple",
    "typed",
]
