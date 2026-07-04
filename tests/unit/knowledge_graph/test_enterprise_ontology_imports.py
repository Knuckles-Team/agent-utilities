"""Guard that the enterprise-architecture ontology modules stay in the TBox.

CONCEPT:AU-KG.ontology.batch-actions-executor — the ArchiMate and LeanIX TTL modules ship as part of the
vendor-neutral enterprise ontology; they must be in ``ontology.ttl``'s
``owl:imports`` closure or the reasoner never sees their classes (a shipped-but-
dead TTL). This pins them to the import list so a future edit can't silently drop
them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.core.ontology_loader import OntologyLoader

_ONTOLOGY = (
    Path(__file__).resolve().parents[3]
    / "agent_utilities"
    / "knowledge_graph"
    / "ontology.ttl"
)


@pytest.mark.concept("AU-KG.ontology.batch-actions-executor")
def test_archimate_and_leanix_are_in_the_import_closure():
    graph = OntologyLoader().load_with_imports(_ONTOLOGY)
    serialized = graph.serialize(format="turtle").lower()
    # Both EA modules must contribute triples to the merged TBox.
    assert "archimate" in serialized, "ArchiMate ontology not in import closure"
    assert "leanix" in serialized, "LeanIX ontology not in import closure"
