"""Research ontology objects are registered + discoverable (CONCEPT:KG-2.76)."""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.ontology.interfaces import (
    DEFAULT_INTERFACE_REGISTRY,
)
from agent_utilities.knowledge_graph.ontology.links import DEFAULT_LINK_REGISTRY
from agent_utilities.knowledge_graph.ontology.research_objects import (
    register_research_ontology,
)
from agent_utilities.models.knowledge_graph import RegistryEdgeType

pytestmark = pytest.mark.concept("KG-2.76")


def test_research_interfaces_registered_by_default():
    # populated at import (side-effect of importing the ontology package)
    import agent_utilities.knowledge_graph.ontology  # noqa: F401

    assert DEFAULT_INTERFACE_REGISTRY.get("ResearchPaper") is not None
    assert DEFAULT_INTERFACE_REGISTRY.get("ResearchConcept") is not None


def test_research_links_registered_with_correct_edge_types():
    import agent_utilities.knowledge_graph.ontology  # noqa: F401

    addresses = DEFAULT_LINK_REGISTRY.get("paper_addresses")
    satisfied = DEFAULT_LINK_REGISTRY.get("paper_satisfied_by")
    relates = DEFAULT_LINK_REGISTRY.get("paper_relates_to")
    assert addresses is not None and addresses.edge_type == RegistryEdgeType.ADDRESSES
    assert (
        satisfied is not None and satisfied.edge_type == RegistryEdgeType.SATISFIED_BY
    )
    assert relates is not None and relates.edge_type == RegistryEdgeType.RELATES_TO


def test_register_research_ontology_is_idempotent():
    # re-running against the already-populated registries must not raise
    register_research_ontology(DEFAULT_INTERFACE_REGISTRY, DEFAULT_LINK_REGISTRY)
    register_research_ontology(DEFAULT_INTERFACE_REGISTRY, DEFAULT_LINK_REGISTRY)
    assert DEFAULT_INTERFACE_REGISTRY.get("ResearchPaper") is not None
