"""Tests for the AU-RETIRE-R003.1 ReferenceTerm typed model."""

import pytest
from pydantic import ValidationError

from agent_utilities.knowledge_graph.ontology.reference_term import ReferenceTerm


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
