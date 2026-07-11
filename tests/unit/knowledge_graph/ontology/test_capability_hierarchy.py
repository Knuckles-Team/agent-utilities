"""Tests for the ontology-native capability subsumption reader (X-4).

Exercises the real bundled ``ontology_capability.ttl`` (no mocking of the
ontology itself — the point is to prove the dependency-free scan reads the
actual ``rdfs:subClassOf`` axioms correctly) plus a synthetic fixture for
edge cases (multi-parent, cycles, missing files).
"""

from __future__ import annotations

import textwrap

from agent_utilities.knowledge_graph.ontology.capability_hierarchy import (
    CapabilityHierarchy,
    get_default_hierarchy,
    load_capability_hierarchy,
)

# ---------------------------------------------------------------------------
# Bundled ontology — real rdfs:subClassOf axioms from ontology_capability.ttl
# ---------------------------------------------------------------------------
def test_direct_subclass_is_recognized():
    h = load_capability_hierarchy()
    assert "ServiceCapability" in h.parents_of("DNSCapability")
    assert h.is_subtype_of("DNSCapability", "ServiceCapability")


def test_two_level_chain_ancestors_include_both_levels():
    h = load_capability_hierarchy()
    # EncryptedTransport ⊑ TransportCapability ⊑ ServiceCapability.
    ancestors = h.ancestors("EncryptedTransport")
    assert "TransportCapability" in ancestors
    assert "ServiceCapability" in ancestors
    assert h.is_subtype_of("EncryptedTransport", "ServiceCapability")
    assert h.is_subtype_of("EncryptedTransport", "TransportCapability")


def test_warm_fork_fanout_two_level_chain_via_restriction_block():
    """WarmForkFanoutCapability's subClassOf list mixes a bare parent with no
    restriction — but SandboxExecutionCapability's own definition sits alongside
    other classes that DO have bracketed owl:Restriction blocks; this proves the
    bracket-stripping regex does not eat a real named parent."""
    h = load_capability_hierarchy()
    assert h.is_subtype_of("WarmForkFanoutCapability", "SandboxExecutionCapability")
    assert h.is_subtype_of("WarmForkFanoutCapability", "ServiceCapability")


def test_restriction_bracket_never_becomes_a_false_superclass():
    """DNSCapability's subClassOf list has a bracketed ``owl:Restriction`` (with
    ``owl:onProperty :providedBy``) ALONGSIDE the bare ``:ServiceCapability``
    parent. The restriction's filler classes (``:providedBy``) must never leak
    into the parent set."""
    h = load_capability_hierarchy()
    parents = h.parents_of("DNSCapability")
    assert parents == {"ServiceCapability"}
    assert "providedBy" not in parents


def test_subtype_is_not_a_supertype_of_its_own_parent():
    h = load_capability_hierarchy()
    assert not h.is_subtype_of("ServiceCapability", "DNSCapability")
    assert not h.is_subtype_of("TransportCapability", "EncryptedTransport")


def test_bfo_root_parent_is_ignored_not_misread_as_a_local_class():
    """``rdfs:subClassOf bfo:0000031`` (a foreign-prefixed URI) must not be
    misread as a local class literally named "0000031"."""
    h = load_capability_hierarchy()
    assert "0000031" not in h.parents_of("ServiceCapability")
    assert "0000031" not in h.known_classes()


def test_unrelated_classes_are_not_subtypes_of_each_other():
    h = load_capability_hierarchy()
    assert not h.is_subtype_of("VPNCapability", "DNSCapability")
    assert not h.is_subtype_of("MailingCapability", "CRMCapability")


def test_subsumption_path_returns_the_full_chain():
    h = load_capability_hierarchy()
    path = h.subsumption_path("WarmForkFanoutCapability", "ServiceCapability")
    assert path == [
        "WarmForkFanoutCapability",
        "SandboxExecutionCapability",
        "ServiceCapability",
    ]
    # Identity path.
    assert h.subsumption_path("DNSCapability", "DNSCapability") == ["DNSCapability"]
    # No relationship -> None.
    assert h.subsumption_path("DNSCapability", "CRMCapability") is None


def test_descendants_is_the_inverse_of_ancestors():
    h = load_capability_hierarchy()
    assert "EncryptedTransport" in h.descendants("TransportCapability")
    assert "PlaintextTransport" in h.descendants("TransportCapability")
    assert "TransportCapability" in h.ancestors("EncryptedTransport")


def test_default_hierarchy_singleton_is_cached():
    a = get_default_hierarchy()
    b = get_default_hierarchy()
    assert a is b


# ---------------------------------------------------------------------------
# Synthetic fixtures — edge cases the bundled ontology doesn't happen to hit
# ---------------------------------------------------------------------------
def _write(tmp_path, name: str, body: str):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_missing_file_degrades_to_an_empty_hierarchy(tmp_path):
    h = CapabilityHierarchy.from_files([tmp_path / "does_not_exist.ttl"])
    assert h.known_classes() == frozenset()
    assert h.ancestors("Anything") == frozenset()
    assert not h.is_subtype_of("A", "B")
    assert h.is_subtype_of("A", "A")  # identity always holds


def test_multi_parent_class_ancestors_union_both_branches(tmp_path):
    ttl = """
    @prefix : <http://example.org/kg#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :Root a owl:Class .

    :BranchA a owl:Class ;
        rdfs:subClassOf :Root .

    :BranchB a owl:Class ;
        rdfs:subClassOf :Root .

    :Diamond a owl:Class ;
        rdfs:subClassOf :BranchA, :BranchB .
    """
    p = _write(tmp_path, "diamond.ttl", ttl)
    h = CapabilityHierarchy.from_files([p])
    assert h.parents_of("Diamond") == {"BranchA", "BranchB"}
    assert h.ancestors("Diamond") == {"BranchA", "BranchB", "Root"}
    assert h.is_subtype_of("Diamond", "Root")


def test_cyclic_definition_never_infinite_loops(tmp_path):
    ttl = """
    @prefix : <http://example.org/kg#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :A a owl:Class ;
        rdfs:subClassOf :B .

    :B a owl:Class ;
        rdfs:subClassOf :A .
    """
    p = _write(tmp_path, "cycle.ttl", ttl)
    h = CapabilityHierarchy.from_files([p])
    # Terminates and reports the (degenerate, mutually-subsuming) ancestry.
    assert "B" in h.ancestors("A")
    assert "A" in h.ancestors("B")
    assert h.is_subtype_of("A", "B")
    assert h.is_subtype_of("B", "A")


def test_malformed_ttl_file_is_skipped_not_raised(tmp_path):
    p = _write(tmp_path, "broken.ttl", "not even close to turtle {{{ :::")
    # Must not raise — a bad ontology file never breaks routing.
    h = CapabilityHierarchy.from_files([p])
    assert h.known_classes() == frozenset()
