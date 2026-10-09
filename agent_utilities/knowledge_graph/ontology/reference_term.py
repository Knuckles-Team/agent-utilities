"""Typed model for an OBO-sourced ReferenceTerm individual.

CONCEPT:AU-RETIRE.ontology.obo-reference-term-ingestion

AU-RETIRE-R003 requires that AU ingest OBO ontology terms (GO biological
process, UBERON/CL, ENVO, PATO, ChEBI) as ``ReferenceTerm`` individuals keyed
by their CURIE, aggregate GO annotations into a ``taxonCapableOf`` relation,
and type ChEBI nutrient-role entries as ``kg:Nutrient``.

This is the ``.1`` slice for that row: a typed model that refuses a term
lacking a valid, source-matching CURIE. Loading the real OBO sources and
the ``taxonCapableOf``/``kg:Nutrient`` aggregation is the remaining behavior
(AU-RETIRE-R003.2+); see ``specs/au-engine-duplicate-retirement/tasks.md``
for the recorded split.
"""

from __future__ import annotations

from pydantic import BaseModel, model_validator

_SOURCE_PREFIXES: dict[str, tuple[str, ...]] = {
    "GO": ("GO",),
    "UBERON": ("UBERON",),
    "CL": ("CL",),
    "ENVO": ("ENVO",),
    "PATO": ("PATO",),
    "CHEBI": ("CHEBI",),
}


class ReferenceTerm(BaseModel):
    """One OBO ontology term, keyed by CURIE.

    Construction refuses a ``curie`` that is not ``PREFIX:local_id`` shaped,
    or whose prefix does not match the declared ``source``, per
    AU-RETIRE-R003.
    """

    curie: str
    source: str
    label: str

    @model_validator(mode="after")
    def _refuse_malformed_or_mismatched_curie(self) -> ReferenceTerm:
        if ":" not in self.curie:
            raise ValueError(
                f"ReferenceTerm refuses a non-CURIE identifier: {self.curie!r} "
                "(AU-RETIRE-R003)"
            )
        prefix, _, local_id = self.curie.partition(":")
        if not prefix or not local_id:
            raise ValueError(
                f"ReferenceTerm refuses a malformed CURIE: {self.curie!r} "
                "(AU-RETIRE-R003)"
            )
        allowed = _SOURCE_PREFIXES.get(self.source.upper())
        if allowed is None or prefix.upper() not in allowed:
            raise ValueError(
                f"ReferenceTerm refuses CURIE prefix {prefix!r} that does not "
                f"match declared source {self.source!r} (AU-RETIRE-R003)"
            )
        return self
