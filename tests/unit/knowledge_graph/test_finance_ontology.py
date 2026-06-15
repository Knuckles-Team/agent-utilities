"""Finance microstructure ontology objects are registered + discoverable (KG-2.81)."""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.ontology.finance_objects import (
    register_finance_ontology,
)
from agent_utilities.knowledge_graph.ontology.interfaces import (
    DEFAULT_INTERFACE_REGISTRY,
)
from agent_utilities.knowledge_graph.ontology.links import DEFAULT_LINK_REGISTRY
from agent_utilities.models.knowledge_graph import RegistryEdgeType

pytestmark = pytest.mark.concept("KG-2.81")


def test_finance_interfaces_registered_by_default():
    # populated at import (side-effect of importing the ontology package)
    import agent_utilities.knowledge_graph.ontology  # noqa: F401

    assert DEFAULT_INTERFACE_REGISTRY.get("MicrostructureSignal") is not None
    surveillance = DEFAULT_INTERFACE_REGISTRY.get("SurveillanceSignal")
    assert surveillance is not None
    # SurveillanceSignal specializes MicrostructureSignal (→ rdfs:subClassOf).
    assert "MicrostructureSignal" in surveillance.extends


def test_finance_links_registered_with_correct_edge_types():
    import agent_utilities.knowledge_graph.ontology  # noqa: F401

    grounded = DEFAULT_LINK_REGISTRY.get("signal_grounded_in_paper")
    relates = DEFAULT_LINK_REGISTRY.get("signal_relates_to_concept")
    assert grounded is not None and grounded.edge_type == RegistryEdgeType.GROUNDED_IN
    # grounded_in carries its inverse so reasoning can walk both directions.
    assert grounded.inverse_edge_type == RegistryEdgeType.SUPPORTS
    assert relates is not None and relates.edge_type == RegistryEdgeType.RELATES_TO


def test_surveillance_signal_emits_owl_subclass():
    surveillance = DEFAULT_INTERFACE_REGISTRY.get("SurveillanceSignal")
    owl = surveillance.to_owl(registry=DEFAULT_INTERFACE_REGISTRY)
    blob = owl if isinstance(owl, str) else str(owl)
    assert "MicrostructureSignal" in blob


def test_register_finance_ontology_is_idempotent():
    register_finance_ontology(DEFAULT_INTERFACE_REGISTRY, DEFAULT_LINK_REGISTRY)
    register_finance_ontology(DEFAULT_INTERFACE_REGISTRY, DEFAULT_LINK_REGISTRY)
    assert DEFAULT_INTERFACE_REGISTRY.get("MicrostructureSignal") is not None
