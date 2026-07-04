"""Tests for ontology grounding of extracted facts (CONCEPT:AU-KG.backend.mirror-health-repair)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.extraction.ontology_grounding import (
    CLASS_SYNONYMS,
    ground_fact,
    ground_facts,
)


class _Fact:
    """Minimal stand-in for an ExtractedFact (subject/predicate/object)."""

    def __init__(self, subject: str, predicate: str, obj: str) -> None:
        self.subject = subject
        self.predicate = predicate
        self.object = obj


def test_cross_modal_org_synonyms_converge():
    """A vendor, a supplier and a company all ground onto ``organization``."""
    vendor = ground_fact("Acme", "issued", "vendor")
    supplier = ground_fact("Acme", "listed_as", "supplier")
    company = ground_fact("Acme", "is_a", "company")
    assert vendor["object_type"] == "organization"
    assert supplier["object_type"] == "organization"
    assert company["object_type"] == "organization"


def test_subject_and_object_grounded():
    g = ground_fact("buyer", "purchased_from", "supplier")
    assert g["subject_type"] == "person"
    assert g["object_type"] == "organization"
    assert g["predicate"] == "purchased_from"


def test_predicate_passthrough():
    g = ground_fact("x", "some_precise_relation", "y")
    assert g["predicate"] == "some_precise_relation"


def test_head_noun_phrase_resolution():
    """A short noun phrase grounds on its head noun."""
    g = ground_fact("purchase invoice", "has_total", "100 USD")
    assert g["subject_type"] == "document"


def test_company_suffix_word_hint():
    g = ground_fact("Acme Corp Ltd", "located_in", "Berlin")
    assert g["subject_type"] == "organization"


def test_currency_value_type_grounding():
    """An ISO currency code object grounds to the ISOCurrencyCode value type."""
    g = ground_fact("invoice", "denominated_in", "USD")
    assert g["object_type"] == "ISOCurrencyCode"
    assert g.get("object_value") == "USD"


def test_email_value_type_grounding():
    g = ground_fact("contact", "has_email", "ops@knuckles.team")
    assert g["object_type"] == "EmailAddress"
    assert g.get("object_value") == "ops@knuckles.team"


def test_url_value_type_grounding():
    g = ground_fact("company", "website", "https://knuckles.team/kg")
    assert g["object_type"] == "URL"


def test_unresolved_surface_is_none_not_error():
    g = ground_fact("zxqyfoobar12345", "frobnicates", "quux99zzz")
    assert g["subject_type"] is None
    assert g["object_type"] is None


def test_interface_name_grounding():
    """Naming a registered interface grounds onto that interface."""
    g = ground_fact("thing", "is", "Locatable")
    assert g["object_type"] == "Locatable"


def test_interface_shape_conformance_grounding():
    """An object dict carrying lat/lon conforms to the Locatable interface."""
    g = ground_fact(
        "reading",
        "at",
        "sensor-7",
        object_props={"lat": 52.5, "lon": 13.4},
    )
    assert g["object_type"] == "Locatable"


def test_ground_facts_batch_returns_pairs():
    # The headline feature is keyword convergence: vendor/supplier/company all
    # ground to "organization", buyer to "person". Entities outside the table
    # (e.g. a bare proper name) ground to None — best-effort, not guessed.
    facts = [
        _Fact("Acme", "is_a", "vendor"),
        _Fact("the buyer", "pays", "the supplier"),
    ]
    pairs = ground_facts(facts)
    assert len(pairs) == 2
    f0, g0 = pairs[0]
    assert f0 is facts[0]
    assert g0["object_type"] == "organization"  # vendor -> organization
    _f1, g1 = pairs[1]
    assert g1["subject_type"] == "person"  # buyer -> person
    assert g1["object_type"] == "organization"  # supplier -> organization


def test_ground_facts_accepts_dicts():
    facts = [{"subject": "city", "predicate": "in", "object": "country"}]
    pairs = ground_facts(facts)
    (_fact, g) = pairs[0]
    assert g["subject_type"] == "place"
    assert g["object_type"] == "place"


def test_ground_facts_never_raises_on_bad_input():
    pairs = ground_facts([object()])  # no subject/predicate/object attrs
    assert len(pairs) == 1
    (_fact, g) = pairs[0]
    assert g["subject_type"] is None


def test_lexicon_targets_are_real_node_types():
    """Every grounding target resolves to a live RegistryNodeType value."""
    from agent_utilities.models.knowledge_graph import RegistryNodeType

    valid = {t.value for t in RegistryNodeType}
    for target in set(CLASS_SYNONYMS.values()):
        assert target in valid
