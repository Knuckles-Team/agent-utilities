"""A schema-repair candidate: the new record contract as SHACL (EH-403).

The candidate is what an approval approves. It is rendered deterministically
from the proposed record shape as one SHACL shapes document under the source's
own namespace, keyed ``approved:<source>`` in the graph's schema sources, and
identified by :func:`approved_candidate_digest` -- byte-for-byte the digest EG
recomputes when the candidate is attached (``eg_types::graph_schema::approval``).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from .shape import FieldShape, RecordShape

#: Key prefix EG reserves for approval-bound schema sources.
APPROVED_SOURCE_PREFIX = "approved:"
#: Domain separator EG frames the candidate digest with.
APPROVED_CANDIDATE_DOMAIN = "eg/approved-schema-candidate/v1"

_XSD = {
    "string": "xsd:string",
    "boolean": "xsd:boolean",
    "integer": "xsd:integer",
    "number": "xsd:decimal",
}
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def approved_candidate_digest(
    source_id: str, shapes_ttl: str | None, ontology_ttl: str | None
) -> str:
    """EG's candidate identity: key and document digests, NUL-terminated."""

    def document(body: str | None) -> str:
        return "" if body is None else hashlib.sha256(body.encode()).hexdigest()

    parts = (
        APPROVED_CANDIDATE_DOMAIN,
        source_id,
        document(shapes_ttl),
        document(ontology_ttl),
    )
    return hashlib.sha256("".join(f"{part}\0" for part in parts).encode()).hexdigest()


def _local(name: str) -> str:
    """An IRI-safe local name (every other character percent-encoded)."""
    return _UNSAFE.sub(lambda m: "".join(f"%{b:02X}" for b in m.group().encode()), name)


def _property_shape(source_ns: str, name: str, shape: FieldShape) -> str:
    lines = [f"  sh:property [ sh:path <{source_ns}{_local(name)}>"]
    if shape.required and "null" not in shape.types:
        lines.append("    ; sh:minCount 1")
    scalars = sorted(shape.types - {"null"})
    if len(scalars) == 1 and scalars[0] in _XSD:
        lines.append(f"    ; sh:datatype {_XSD[scalars[0]]}")
    return "\n".join(lines) + " ]"


def render_shapes(source: str, shape: RecordShape) -> str:
    """The SHACL document of ``shape`` for ``source`` (deterministic)."""
    source_ns = f"urn:au:source:{_local(source)}#"
    header = (
        "@prefix sh: <http://www.w3.org/ns/shacl#> .\n"
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n"
        f"<{source_ns}RecordShape> a sh:NodeShape ;\n"
        f"  sh:targetClass <{source_ns}Record>"
    )
    properties = [_property_shape(source_ns, name, spec) for name, spec in shape.fields]
    return " ;\n".join([header, *properties]) + " .\n"


@dataclass(frozen=True, slots=True)
class RepairCandidate:
    """The proposed contract, its SHACL rendering and its identity."""

    source: str
    shape: RecordShape
    shapes_ttl: str
    renames: Mapping[str, str] = field(default_factory=dict)

    @property
    def source_id(self) -> str:
        return f"{APPROVED_SOURCE_PREFIX}{self.source}"

    @property
    def digest(self) -> str:
        return approved_candidate_digest(self.source_id, self.shapes_ttl, None)


def build_candidate(
    source: str, shape: RecordShape, renames: Mapping[str, str]
) -> RepairCandidate:
    return RepairCandidate(source, shape, render_shapes(source, shape), dict(renames))


__all__ = [
    "APPROVED_CANDIDATE_DOMAIN",
    "APPROVED_SOURCE_PREFIX",
    "RepairCandidate",
    "approved_candidate_digest",
    "build_candidate",
    "render_shapes",
]
