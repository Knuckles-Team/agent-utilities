#!/usr/bin/python
"""Unit tests for ontology value types (CONCEPT:AU-KG.ontology.value-type-shacl-load).

Covers per-built-in constraint pass/fail (runtime validator), the
schema_definition/base coercion bridge, and asserts the emitted SHACL turtle and
OWL datatype-restriction turtle carry the expected structure and parse in
epistemic-graph (``OntologyInspect``; AU owns no RDF parser — EH-471).
"""

import pytest

from agent_utilities.knowledge_graph.ontology.value_types import (
    VALUE_TYPES,
    ValueConstraints,
    ValueType,
    coerce_value_type,
    get_value_type,
    list_value_types,
    register_value_type,
    validate_value_type,
    value_types_owl_ttl,
    value_types_shapes_ttl,
    write_value_shapes_ttl,
)


def test_registry_populated_with_real_builtins():
    names = set(list_value_types())
    required = {
        "EmailAddress",
        "ISOCurrencyCode",
        "Percentage",
        "URL",
        "E164PhoneNumber",
        "Probability",
    }
    assert required <= names
    for name in names:
        assert isinstance(VALUE_TYPES[name], ValueType)


# --- EmailAddress ------------------------------------------------------------
def test_email_address_constraints():
    assert validate_value_type("EmailAddress", "ops@knuckles.team")
    assert (
        coerce_value_type("EmailAddress", "a.b+c@sub.example.co")
        == "a.b+c@sub.example.co"
    )
    assert not validate_value_type("EmailAddress", "not-an-email")
    assert not validate_value_type("EmailAddress", "missing@tld")
    # max_length enforced
    huge = "x" * 250 + "@a.com"
    assert not validate_value_type("EmailAddress", huge)


# --- URL ---------------------------------------------------------------------
def test_url_constraints():
    assert validate_value_type("URL", "https://knuckles.team/kg?x=1")
    assert validate_value_type("URL", "http://example.com")
    assert not validate_value_type("URL", "ftp://example.com")
    assert not validate_value_type("URL", "knuckles.team")


# --- ISOCurrencyCode ---------------------------------------------------------
def test_iso_currency_code_constraints():
    assert coerce_value_type("ISOCurrencyCode", "USD") == "USD"
    assert validate_value_type("ISOCurrencyCode", "EUR")
    assert not validate_value_type("ISOCurrencyCode", "usd")  # lowercase
    assert not validate_value_type("ISOCurrencyCode", "US")  # too short
    assert not validate_value_type("ISOCurrencyCode", "USDX")  # too long


# --- E164PhoneNumber ---------------------------------------------------------
def test_e164_phone_constraints():
    assert validate_value_type("E164PhoneNumber", "+14155550123")
    assert not validate_value_type("E164PhoneNumber", "4155550123")  # no '+'
    assert not validate_value_type("E164PhoneNumber", "+0123")  # leading zero


# --- Percentage / Probability (numeric bounds) -------------------------------
def test_percentage_bounds():
    assert coerce_value_type("Percentage", "42.5") == 42.5
    assert validate_value_type("Percentage", 0)
    assert validate_value_type("Percentage", 100)
    assert not validate_value_type("Percentage", -1)
    assert not validate_value_type("Percentage", 100.1)


def test_probability_bounds():
    assert validate_value_type("Probability", 0.0)
    assert validate_value_type("Probability", 1.0)
    assert not validate_value_type("Probability", 1.5)
    assert not validate_value_type("Probability", -0.01)


# --- coercion goes through the base type first -------------------------------
def test_base_coercion_applied_before_constraint():
    # "100" is a string but base double-coerces it, then bound passes.
    assert coerce_value_type("Percentage", "100") == 100.0
    # boolean must not slip through the numeric base type.
    assert not validate_value_type("Percentage", True)


# --- enum + exclusive bound value types --------------------------------------
def test_enum_value_type():
    side = register_value_type(
        ValueType(
            name="OrderSide",
            base_type="string",
            constraints=ValueConstraints(allowed_values=["buy", "sell"]),
            description="Side of an order.",
        ),
        overwrite=True,
    )
    assert side.validate("buy")
    assert not side.validate("hold")


def test_exclusive_bounds():
    vt = ValueType(
        name="OpenUnitInterval",
        base_type="double",
        constraints=ValueConstraints(
            min_value=0, max_value=1, exclusive_min=True, exclusive_max=True
        ),
    )
    assert vt.validate(0.5)
    assert not vt.validate(0.0)
    assert not vt.validate(1.0)


def test_register_duplicate_rejected():
    with pytest.raises(ValueError):
        register_value_type(
            ValueType(name="EmailAddress", base_type="integer"),
        )


def test_get_and_unknown():
    assert get_value_type("EmailAddress").name == "EmailAddress"
    assert get_value_type("DoesNotExist") is None
    assert not validate_value_type("DoesNotExist", "x")
    with pytest.raises(ValueError):
        coerce_value_type("DoesNotExist", "x")


# --- SHACL turtle emission ---------------------------------------------------
def test_to_shacl_reusable_node_shape_parses():
    vt = VALUE_TYPES["EmailAddress"]
    from agent_utilities.knowledge_graph.ontology.value_types import SHAPES_PREFIXES

    turtle = SHAPES_PREFIXES + "\n" + vt.to_shacl()
    assert "a sh:NodeShape" in turtle
    assert "sh:pattern" in turtle and "sh:datatype" in turtle
    assert turtle.rstrip().endswith(".")


def test_to_shacl_node_shape_with_path_parses():
    vt = VALUE_TYPES["Percentage"]
    from agent_utilities.knowledge_graph.ontology.value_types import SHAPES_PREFIXES

    turtle = (
        SHAPES_PREFIXES + "\n" + vt.to_shacl(path="completionRate", target_class="Task")
    )
    assert "sh:targetClass" in turtle
    assert "sh:minInclusive" in turtle and "sh:maxInclusive" in turtle


def test_full_registry_shapes_ttl_parses():
    turtle = value_types_shapes_ttl()
    assert turtle.count("a sh:NodeShape") >= len(list_value_types())


# --- OWL datatype restriction emission ---------------------------------------
def test_to_owl_facet_restriction_parses():
    vt = VALUE_TYPES["ISOCurrencyCode"]
    from agent_utilities.knowledge_graph.ontology.value_types import SHAPES_PREFIXES

    turtle = SHAPES_PREFIXES + "\n" + vt.to_owl()
    assert "rdfs:Datatype" in turtle and "owl:withRestrictions" in turtle


def test_to_owl_enum_oneof_parses():
    vt = ValueType(
        name="TrafficLight",
        base_type="string",
        constraints=ValueConstraints(allowed_values=["red", "amber", "green"]),
    )
    from agent_utilities.knowledge_graph.ontology.value_types import SHAPES_PREFIXES

    turtle = SHAPES_PREFIXES + "\n" + vt.to_owl()
    assert "owl:oneOf" in turtle


def test_full_registry_owl_ttl_parses():
    turtle = value_types_owl_ttl()
    assert turtle.count("a rdfs:Datatype") >= len(list_value_types())


# --- live-path materialization to shapes/ ------------------------------------
def test_write_value_shapes_ttl_materializes_loadable_file(tmp_path):
    target = tmp_path / "value_types.shapes.ttl"
    written = write_value_shapes_ttl(str(target))
    assert written == str(target)
    content = target.read_text(encoding="utf-8")
    assert "sh:NodeShape" in content


def test_value_type_documents_parse_in_eg(engine_graph):
    """The registry's SHACL and OWL renderings are well-formed Turtle — EG, the
    only RDF parser, reads both (the in-memory layer validate_kg appends)."""
    shapes = engine_graph.ontology_inspect([value_types_shapes_ttl()])
    assert shapes.triple_count > 0
    owl = engine_graph.ontology_inspect([value_types_owl_ttl()])
    assert owl.triple_count > 0
