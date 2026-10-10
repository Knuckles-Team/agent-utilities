"""Typed model for an OBO-sourced ReferenceTerm individual.

CONCEPT:AU-RETIRE.ontology.obo-reference-term-ingestion

AU-RETIRE-R003 requires that AU ingest OBO ontology terms (GO biological
process, UBERON/CL, ENVO, PATO, ChEBI) as ``ReferenceTerm`` individuals keyed
by their CURIE, aggregate GO annotations into a ``taxonCapableOf`` relation,
and type ChEBI nutrient-role entries as ``kg:Nutrient``.

The ``.1`` slice delivered the typed model that refuses a term lacking a
valid, source-matching CURIE. This is the ``.2`` slice: a real ingest
entry point, ``ingest_reference_terms``, that turns an iterable of raw OBO
term records into ``ReferenceTerm`` individuals keyed by CURIE, deduplicating
repeated CURIEs deterministically (first record wins) instead of letting a
caller re-validate and re-key records by hand. The GO ``taxonCapableOf``
aggregation and the ChEBI ``kg:Nutrient`` typing remain the next slice
(AU-RETIRE-R003.3+); see ``specs/au-engine-duplicate-retirement/tasks.md``
for the recorded split.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

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


def ingest_reference_terms(
    records: Iterable[Mapping[str, str]],
) -> dict[str, ReferenceTerm]:
    """Build ``ReferenceTerm`` individuals from raw OBO term records.

    Each record is a mapping with ``curie``, ``source`` and ``label`` keys,
    as read from an OBO source file or loader. A record is validated through
    ``ReferenceTerm`` (AU-RETIRE-R003), so a non-CURIE identifier or a CURIE
    prefix mismatched with its declared source raises ``ValidationError``
    before ingest completes. Repeated CURIEs are deduplicated: the first
    record for a given CURIE wins and later duplicates are dropped, so
    ingest is deterministic regardless of source ordering (AU-RETIRE-R003.2).

    Returns the individuals keyed by CURIE.
    """
    terms: dict[str, ReferenceTerm] = {}
    for record in records:
        term = ReferenceTerm(
            curie=record["curie"],
            source=record["source"],
            label=record["label"],
        )
        if term.curie not in terms:
            terms[term.curie] = term
    return terms
