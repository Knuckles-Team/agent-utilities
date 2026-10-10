"""Tests for the AU-RETIRE-R003.1/.2 ReferenceTerm ingest path."""

import pytest
from pydantic import ValidationError

from agent_utilities.knowledge_graph.ontology.reference_term import (
    ReferenceTerm,
    ingest_reference_terms,
)


def test_accepts_valid_go_curie() -> None:
    term = ReferenceTerm(curie="GO:0008150", source="GO", label="biological_process")
    assert term.curie == "GO:0008150"


def test_refuses_non_curie_identifier() -> None:
    """AU-RETIRE-R003: a term lacking CURIE shape is refused."""
    with pytest.raises(ValidationError, match="AU-RETIRE-R003"):
        ReferenceTerm(curie="0008150", source="GO", label="biological_process")


def test_refuses_curie_prefix_mismatched_with_source() -> None:
    """AU-RETIRE-R003: a CURIE prefix that disagrees with source is refused."""
    with pytest.raises(ValidationError, match="AU-RETIRE-R003"):
        ReferenceTerm(curie="CHEBI:15377", source="GO", label="water")


@pytest.mark.spec("AU-RETIRE-R003.2")
def test_ingest_reference_terms_keys_individuals_by_curie() -> None:
    records = [
        {"curie": "GO:0008150", "source": "GO", "label": "biological_process"},
        {"curie": "CHEBI:15377", "source": "CHEBI", "label": "water"},
    ]
    terms = ingest_reference_terms(records)
    assert set(terms) == {"GO:0008150", "CHEBI:15377"}
    assert terms["CHEBI:15377"].label == "water"


@pytest.mark.spec("AU-RETIRE-R003.2")
def test_ingest_reference_terms_deduplicates_repeated_curie_first_wins() -> None:
    records = [
        {"curie": "GO:0008150", "source": "GO", "label": "first"},
        {"curie": "GO:0008150", "source": "GO", "label": "second"},
    ]
    terms = ingest_reference_terms(records)
    assert len(terms) == 1
    assert terms["GO:0008150"].label == "first"


@pytest.mark.spec("AU-RETIRE-R003.2")
def test_ingest_reference_terms_rejects_invalid_record() -> None:
    records = [{"curie": "0008150", "source": "GO", "label": "biological_process"}]
    with pytest.raises(ValidationError, match="AU-RETIRE-R003"):
        ingest_reference_terms(records)
