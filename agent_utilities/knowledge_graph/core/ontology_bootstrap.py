#!/usr/bin/python
from __future__ import annotations

"""Self-bootstrapping ontology agent (schema-free KG construction).

CONCEPT:AU-KG.enrichment.entity-claim-extraction — Entity/Claim Extraction (bootstrapped ontology)

Distilled from the Product-KG research (`.specify/specs/research-evolution-20260606/`
plan b7-05): instead of a fixed hand-authored ontology, *derive* the ontology from
a sample corpus — classes from record types, properties (with inferred datatype +
domain) from their attributes — with **plateau-based stopping** (stop once new
samples stop adding ontology elements), and populate only **grounded** triples
(explicit values, unit-normalised, no inference). Deterministic, no LLM.

Gated behind a config flag in the ingest path (the existing fixed `ontology.ttl`
remains the default); the SHACL gate still validates the grounded triples.

Concept: ontology-bootstrap
"""

import re
from typing import Any

from pydantic import BaseModel, Field

_XSD = {
    bool: "xsd:boolean",
    int: "xsd:integer",
    float: "xsd:decimal",
    str: "xsd:string",
}
_UNIT_SUFFIX = re.compile(r"\s*(usd|kg|g|mg|ml|l|cm|mm|m|km|%|ms|s)\b", re.IGNORECASE)


def _datatype(value: Any) -> str:
    # bool is a subclass of int — check it first
    for py_t, xsd in _XSD.items():
        if isinstance(value, py_t):
            return xsd
    return "xsd:string"


def _normalize_value(value: Any) -> str:
    """Unit-normalise a grounded value (strip trailing units, collapse whitespace)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    s = _UNIT_SUFFIX.sub("", str(value)).strip()
    return re.sub(r"\s+", " ", s)


class OntologyProperty(BaseModel):
    name: str
    domain: str
    datatype: str = "xsd:string"
    sample_count: int = 0


class OntologyClass(BaseModel):
    name: str
    properties: list[str] = Field(default_factory=list)
    instance_count: int = 0


class BootstrapResult(BaseModel):
    classes: list[OntologyClass] = Field(default_factory=list)
    properties: list[OntologyProperty] = Field(default_factory=list)
    samples_seen: int = 0
    plateaued: bool = False


class OntologyBootstrapper:
    """Incrementally derive an ontology from sample records (CONCEPT:AU-KG.enrichment.entity-claim-extraction)."""

    def __init__(self, *, plateau_patience: int = 5) -> None:
        self.plateau_patience = max(1, plateau_patience)
        self._classes: dict[str, OntologyClass] = {}
        self._properties: dict[tuple[str, str], OntologyProperty] = {}
        self._no_new_streak = 0
        self._samples = 0

    def observe(self, record: dict[str, Any], *, class_key: str = "type") -> int:
        """Learn classes/properties from one record; returns # new elements added."""
        self._samples += 1
        cls = str(record.get(class_key) or "Thing")
        new = 0
        oc = self._classes.get(cls)
        if oc is None:
            oc = OntologyClass(name=cls)
            self._classes[cls] = oc
            new += 1
        oc.instance_count += 1
        for key, value in record.items():
            if key == class_key or value is None or value == "":
                continue
            pk = (cls, key)
            prop = self._properties.get(pk)
            if prop is None:
                self._properties[pk] = OntologyProperty(
                    name=key, domain=cls, datatype=_datatype(value), sample_count=1
                )
                if key not in oc.properties:
                    oc.properties.append(key)
                new += 1
            else:
                prop.sample_count += 1
        self._no_new_streak = 0 if new else self._no_new_streak + 1
        return new

    @property
    def plateaued(self) -> bool:
        return self._no_new_streak >= self.plateau_patience

    def bootstrap(
        self, samples: list[dict[str, Any]], *, class_key: str = "type"
    ) -> BootstrapResult:
        """Observe samples until the ontology plateaus or samples are exhausted."""
        for record in samples:
            self.observe(record, class_key=class_key)
            if self.plateaued:
                break
        return self.result()

    def result(self) -> BootstrapResult:
        return BootstrapResult(
            classes=list(self._classes.values()),
            properties=list(self._properties.values()),
            samples_seen=self._samples,
            plateaued=self.plateaued,
        )

    def to_turtle(self) -> str:
        """Emit the derived ontology as RDF/Turtle (classes + typed properties)."""
        lines = [
            "@prefix : <http://agent-utilities/onto#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            "",
        ]
        for oc in self._classes.values():
            lines.append(f":{oc.name} a rdfs:Class .")
        for prop in self._properties.values():
            lines.append(
                f":{prop.name} a rdf:Property ; rdfs:domain :{prop.domain} ; "
                f"rdfs:range {prop.datatype} ."
            )
        return "\n".join(lines)

    def grounded_triples(
        self,
        record: dict[str, Any],
        *,
        subject_key: str = "id",
        class_key: str = "type",
    ) -> list[tuple[str, str, str]]:
        """Emit ``(subject, predicate, value)`` triples for explicit values only.

        Anti-hallucination: a triple is emitted only when the record carries an
        explicit non-empty value; values are unit-normalised. No inference.
        """
        subject = str(record.get(subject_key) or "")
        if not subject:
            return []
        triples: list[tuple[str, str, str]] = []
        cls = record.get(class_key)
        if cls:
            triples.append((subject, "rdf:type", str(cls)))
        for key, value in record.items():
            if key in (subject_key, class_key) or value is None or value == "":
                continue
            triples.append((subject, key, _normalize_value(value)))
        return triples
