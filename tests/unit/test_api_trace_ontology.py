"""Public API exports trace_id (AU-BOUNDARY-R013.5)."""

from __future__ import annotations

import pytest

from agent_utilities.api import trace_ontology
from agent_utilities.observability import trace_ontology as internal


@pytest.mark.spec("AU-BOUNDARY-R013.5")
def test_trace_id_is_the_internal_definition() -> None:
    assert trace_ontology.trace_id is internal.trace_id
    assert trace_ontology.__all__ == ["trace_id"]
