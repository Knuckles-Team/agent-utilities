#!/usr/bin/python
from __future__ import annotations

"""Tests for ontology Interfaces (CONCEPT:AU-KG.ontology.conformance-check).

Covers: defining an interface, implementing it with a conforming and a
non-conforming type (gap detection), interface inheritance (extends), and the
programmatic-targeting feature (find_implementers / conforms / resolve_target).
"""

import pytest

from agent_utilities.knowledge_graph.ontology.interfaces import (
    DEFAULT_INTERFACE_REGISTRY,
    ImplementationReport,
    Interface,
    InterfaceLinkConstraint,
    InterfaceProperty,
    InterfaceRegistry,
    target_object_types,
)
from agent_utilities.models.knowledge_graph import (
    RegistryEdgeType,
    RegistryNodeType,
)


def _registry() -> InterfaceRegistry:
    reg = InterfaceRegistry()
    reg.register(
        Interface(
            name="Locatable",
            properties=[
                InterfaceProperty(name="lat", type_ref="double"),
                InterfaceProperty(name="lon", type_ref="double"),
            ],
        )
    )
    return reg


def test_define_and_owl_emission():
    reg = _registry()
    iface = reg.get("Locatable")
    assert iface is not None
    ttl = iface.to_owl(registry=reg)
    # owl:Class + SHACL NodeShape with both required interface-properties.
    assert ":Locatable a owl:Class" in ttl
    assert ":LocatableShape a sh:NodeShape" in ttl
    assert "sh:targetClass :Locatable" in ttl
    assert "sh:path :lat" in ttl
    assert "sh:path :lon" in ttl
    assert "sh:minCount 1" in ttl


def test_conforming_and_nonconforming_implementation_gaps():
    reg = _registry()

    # Conforming type: declares both lat+lon as double.
    reg.declare_type_shape("geo_place", properties={"lat": "double", "lon": "double"})
    ok_report = reg.implement("geo_place", "Locatable")
    assert isinstance(ok_report, ImplementationReport)
    assert ok_report.ok is True
    assert ok_report.gaps == []

    # Non-conforming type: missing lon, and lat is the wrong type (string).
    reg.declare_type_shape("bad_place", properties={"lat": "string"})
    bad_report = reg.implement("bad_place", "Locatable")
    assert bad_report.ok is False
    # Both a missing-property gap and a type-mismatch gap are reported.
    joined = " | ".join(bad_report.gaps)
    assert "lon" in joined  # missing required property
    assert "lat" in joined  # wrong type (double declared as string)


def test_required_property_missing_value_at_instance_level():
    reg = _registry()
    iface = reg.get("Locatable")
    assert iface is not None
    # Instance missing lon entirely.
    gaps = iface.gaps_for({"lat": 1.0}, registry=reg)
    assert any("lon" in g for g in gaps)
    # Instance with a non-double lat (not coercible) fails the type check.
    gaps2 = iface.gaps_for({"lat": "not-a-number", "lon": 2.0}, registry=reg)
    assert any("lat" in g for g in gaps2)
    # Fully valid instance conforms.
    assert iface.conforms({"lat": 1.0, "lon": 2.0}, registry=reg)


def test_interface_inheritance_extends():
    reg = _registry()
    reg.register(
        Interface(
            name="HasProvenance",
            properties=[InterfaceProperty(name="timestamp", type_ref="timestamp")],
        )
    )
    reg.register(
        Interface(
            name="GeoEvent",
            extends=["Locatable", "HasProvenance"],
            link_constraints=[
                InterfaceLinkConstraint(
                    name="when",
                    edge_type=RegistryEdgeType.HAS_TEMPORAL_EXTENT,
                    min_count=1,
                )
            ],
        )
    )
    geo = reg.get("GeoEvent")
    assert geo is not None
    # Inheritance-resolved shape includes parents' properties.
    all_props = geo.all_properties(reg)
    assert {"lat", "lon", "timestamp"} <= set(all_props)

    # A conforming instance must satisfy inherited props AND own link constraint.
    conforming = {
        "lat": 1.0,
        "lon": 2.0,
        "timestamp": "2020-01-01T00:00:00Z",
        "links": [{"type": "has_temporal_extent", "target": "t1"}],
    }
    assert geo.conforms(conforming, registry=reg)

    # Missing the required temporal link => gap.
    missing_link = dict(conforming)
    missing_link.pop("links")
    gaps = geo.gaps_for(missing_link, registry=reg)
    assert any("has_temporal_extent" in g for g in gaps)

    # Registering an interface extending an unknown parent is rejected.
    with pytest.raises(ValueError):
        reg.register(Interface(name="Orphan", extends=["DoesNotExist"]))


def test_find_implementers_and_targeting():
    reg = _registry()
    reg.declare_type_shape("geo_place", properties={"lat": "double", "lon": "double"})
    reg.declare_type_shape("geo_sensor", properties={"lat": "double", "lon": "double"})
    reg.implement("geo_place", "Locatable")
    reg.implement("geo_sensor", "Locatable")

    implementers = reg.find_implementers("Locatable")
    assert set(implementers) == {"geo_place", "geo_sensor"}

    # resolve_target: an interface name expands to its implementers; a plain
    # concrete type name passes through unchanged (programmatic targeting).
    assert set(reg.resolve_target("Locatable")) == {"geo_place", "geo_sensor"}
    assert reg.resolve_target("some_concrete_type") == ["some_concrete_type"]

    # conforms() used as the runtime targeting gate.
    assert reg.conforms({"lat": 1.0, "lon": 2.0}, "Locatable")
    assert not reg.conforms({"lat": 1.0}, "Locatable")

    # Unknown interface raises.
    with pytest.raises(ValueError):
        reg.find_implementers("Nope")


def test_transitive_targeting_through_subinterface():
    reg = _registry()
    reg.register(Interface(name="PreciseLocatable", extends=["Locatable"]))
    reg.declare_type_shape("gps_fix", properties={"lat": "double", "lon": "double"})
    reg.implement("gps_fix", "PreciseLocatable")
    # An implementer of the sub-interface is a transitive implementer of the parent.
    assert "gps_fix" in reg.find_implementers("Locatable")


def test_default_registry_live_builtins():
    # The import-populated default registry ships real interfaces + implementers.
    assert "HasProvenance" in DEFAULT_INTERFACE_REGISTRY
    assert "Locatable" in DEFAULT_INTERFACE_REGISTRY
    assert "GeoTracked" in DEFAULT_INTERFACE_REGISTRY

    doc_impls = DEFAULT_INTERFACE_REGISTRY.find_implementers("HasProvenance")
    assert RegistryNodeType.DOCUMENT.value in doc_impls
    place_impls = DEFAULT_INTERFACE_REGISTRY.find_implementers("Locatable")
    assert RegistryNodeType.PLACE.value in place_impls

    # GeoTracked extends Locatable + HasProvenance => inherits lat/lon/timestamp.
    geo = DEFAULT_INTERFACE_REGISTRY.get("GeoTracked")
    assert geo is not None
    props = geo.all_properties(DEFAULT_INTERFACE_REGISTRY)
    assert {"lat", "lon", "timestamp"} <= set(props)

    # Module-level targeting convenience resolves an interface to its types.
    assert RegistryNodeType.PLACE.value in target_object_types("Locatable")
    # And passes a plain concrete-type name through unchanged.
    assert target_object_types("document") == ["document"]


def test_registry_to_owl_emits_implements_assertions():
    reg = _registry()
    reg.declare_type_shape("geo_place", properties={"lat": "double", "lon": "double"})
    reg.implement("geo_place", "Locatable")
    ttl = reg.to_owl()
    assert ":Locatable a owl:Class" in ttl
    # implementing type emitted rdfs:subClassOf + sh:node the interface shape.
    assert "rdfs:subClassOf :Locatable" in ttl
    assert "sh:node :LocatableShape" in ttl
