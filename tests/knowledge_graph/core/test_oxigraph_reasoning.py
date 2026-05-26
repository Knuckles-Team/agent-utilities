"""Unit tests for Rust-compiled Oxigraph OWL-RL Datalog reasoning engine.

CONCEPT:KG-2.6
"""

import pytest
import os
import tempfile
from agent_utilities.knowledge_graph.backends.owl import OxigraphDatalogBackend

def test_oxigraph_subclass_transitivity():
    """Verify that OxigraphDatalogBackend correctly reasons subclass transitivity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ontology_path = os.path.join(tmpdir, "test_subclass.ttl")

        # Write a simple Turtle ontology defining:
        # :A rdfs:subClassOf :B .
        # :B rdfs:subClassOf :C .
        # :x a :A .
        ttl_content = """
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix : <http://example.org/> .

        :A rdfs:subClassOf :B .
        :B rdfs:subClassOf :C .
        :x a :A .
        """
        with open(ontology_path, "w") as f:
            f.write(ttl_content)

        backend = OxigraphDatalogBackend(ontology_path=ontology_path)

        # Verify initial types & asserted subclasses
        # x is asserted as type A
        assert backend.is_subclass_of("http://example.org/A", "http://example.org/C")
        assert backend.is_subclass_of("http://example.org/A", "http://example.org/B")
        assert backend.is_subclass_of("http://example.org/B", "http://example.org/C")

        # Query individuals of type C (should infer x via subClassOf transitivity)
        c_instances = backend.get_instances_of("http://example.org/C")
        assert "http://example.org/x" in c_instances

        # Query individuals of type B
        b_instances = backend.get_instances_of("http://example.org/B")
        assert "http://example.org/x" in b_instances


def test_oxigraph_symmetric_properties():
    """Verify symmetric property reasoning in OxigraphDatalogBackend."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ontology_path = os.path.join(tmpdir, "test_symmetric.ttl")

        # Define :partnerOf as symmetric, and :alice :partnerOf :bob
        ttl_content = """
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix : <http://example.org/> .

        :partnerOf a owl:SymmetricProperty .
        :alice :partnerOf :bob .
        """
        with open(ontology_path, "w") as f:
            f.write(ttl_content)

        backend = OxigraphDatalogBackend(ontology_path=ontology_path)

        # Verify symmetric inference: bob is partnerOf alice
        partners = backend.get_property_values("http://example.org/bob", "http://example.org/partnerOf")
        assert "http://example.org/alice" in partners


def test_oxigraph_inverse_properties():
    """Verify inverse property reasoning in OxigraphDatalogBackend."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ontology_path = os.path.join(tmpdir, "test_inverse.ttl")

        # Define :childOf as inverse of :parentOf, and :bob :childOf :alice
        ttl_content = """
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix : <http://example.org/> .

        :childOf owl:inverseOf :parentOf .
        :bob :childOf :alice .
        """
        with open(ontology_path, "w") as f:
            f.write(ttl_content)

        backend = OxigraphDatalogBackend(ontology_path=ontology_path)

        # Verify inverse inference: alice is parentOf bob
        parents = backend.get_property_values("http://example.org/alice", "http://example.org/parentOf")
        assert "http://example.org/bob" in parents


def test_oxigraph_transitive_properties():
    """Verify transitive property reasoning in OxigraphDatalogBackend."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ontology_path = os.path.join(tmpdir, "test_transitive.ttl")

        # Define :dependsOn as transitive, and A -> B -> C
        ttl_content = """
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix : <http://example.org/> .

        :dependsOn a owl:TransitiveProperty .
        :A :dependsOn :B .
        :B :dependsOn :C .
        """
        with open(ontology_path, "w") as f:
            f.write(ttl_content)

        backend = OxigraphDatalogBackend(ontology_path=ontology_path)

        # Verify transitive inference: A dependsOn C
        deps = backend.get_property_values("http://example.org/A", "http://example.org/dependsOn")
        assert "http://example.org/C" in deps
        assert "http://example.org/B" in deps


def test_oxigraph_legal_transitivity():
    """Verify that OxigraphDatalogBackend correctly reasons subclass transitivity for LegalTrust and LLCFormationFiling."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ontology_path = os.path.abspath(os.path.join(
        current_dir, "..", "..", "..", "agent_utilities", "knowledge_graph", "ontology_legal.ttl"
    ))

    backend = OxigraphDatalogBackend(ontology_path=ontology_path)

    # Verify that LLCFormationFiling is a subclass of RegulatoryFiling
    assert backend.is_subclass_of("http://knuckles.team/kg#LLCFormationFiling", "http://knuckles.team/kg#RegulatoryFiling")

    # Verify that LLCFormationFiling is a subclass of bfo:0000015 (RegulatoryFiling's superclass)
    assert backend.is_subclass_of("http://knuckles.team/kg#LLCFormationFiling", "http://purl.obolibrary.org/obo/BFO_0000015")

    # Verify that LegalTrust is a subclass of bfo:0000031
    assert backend.is_subclass_of("http://knuckles.team/kg#LegalTrust", "http://purl.obolibrary.org/obo/BFO_0000031")


def test_oxigraph_harness_transitivity():
    """Verify that OxigraphDatalogBackend correctly reasons subclass transitivity for new Developer Harness classes."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ontology_path = os.path.abspath(os.path.join(
        current_dir, "..", "..", "..", "agent_utilities", "knowledge_graph", "ontology.ttl"
    ))

    backend = OxigraphDatalogBackend(ontology_path=ontology_path)

    # Verify that LSPServer is a subclass of ApplicationLayer
    assert backend.is_subclass_of("http://knuckles.team/kg#LSPServer", "http://knuckles.team/kg#ApplicationLayer")

    # Verify that LSPServer is a subclass of ArchiMateElement (ApplicationLayer's superclass)
    assert backend.is_subclass_of("http://knuckles.team/kg#LSPServer", "http://knuckles.team/kg#ArchiMateElement")

    # Verify that ContextFile is a subclass of TechnologyLayer
    assert backend.is_subclass_of("http://knuckles.team/kg#ContextFile", "http://knuckles.team/kg#TechnologyLayer")

    # Verify that OptimizationPattern is a subclass of MotivationLayer
    assert backend.is_subclass_of("http://knuckles.team/kg#OptimizationPattern", "http://knuckles.team/kg#MotivationLayer")
