#!/usr/bin/python
"""Unit tests for the ontology PropertyType registry (CONCEPT:AU-KG.ontology.ontology-property-types).

Exercises coercion + validation for every base scalar type and the complex
types (array, struct, vector, marking, geo, timeseries, attachment,
media_reference), plus the schema_definition column-type mapping that ties the
registry back into TableDefinition.columns.
"""

import datetime as _dt
from decimal import Decimal

import pytest

from agent_utilities.knowledge_graph.ontology.property_types import (
    DEFAULT_VECTOR_DIM,
    PROPERTY_TYPES,
    PropertyType,
    coerce_value,
    column_type_for,
    get_property_type,
    list_property_types,
    parse_type_ref,
    validate_value,
)


def test_registry_populated_with_all_palantir_types():
    names = set(list_property_types())
    required = {
        "string",
        "boolean",
        "byte",
        "short",
        "integer",
        "long",
        "float",
        "double",
        "decimal",
        "date",
        "timestamp",
        "geohash",
        "geoshape",
        "geotimeseries",
        "timeseries",
        "struct",
        "attachment",
        "media_reference",
        "marking",
        "vector",
        "embedding",
    }
    assert required <= names
    for name in names:
        assert isinstance(PROPERTY_TYPES[name], PropertyType)


# --- Scalar coercion ---------------------------------------------------------
@pytest.mark.parametrize(
    "type_name,raw,expected",
    [
        ("string", 123, "123"),
        ("boolean", "yes", True),
        ("boolean", "0", False),
        ("byte", "127", 127),
        ("short", 30000, 30000),
        ("integer", "42", 42),
        ("long", 2**40, 2**40),
        ("float", "3.14", 3.14),
        ("double", 2, 2.0),
        ("decimal", "1.10", Decimal("1.10")),
    ],
)
def test_scalar_coercion(type_name, raw, expected):
    assert coerce_value(type_name, raw) == expected


def test_bounded_int_overflow_rejected():
    assert not validate_value("byte", 999)
    assert not validate_value("short", 2**20)
    with pytest.raises(ValueError):
        coerce_value("byte", 1000)


def test_bool_is_not_an_integer():
    assert not validate_value("integer", True)
    assert not validate_value("float", False)


def test_date_and_timestamp():
    d = coerce_value("date", "2026-06-09")
    assert d == _dt.date(2026, 6, 9)
    ts = coerce_value("timestamp", "2026-06-09T12:00:00Z")
    assert ts.tzinfo is not None
    assert ts.year == 2026 and ts.hour == 12
    # epoch + naive get a UTC tz attached
    assert coerce_value("timestamp", 0).tzinfo is not None


# --- Geo types ---------------------------------------------------------------
def test_geohash():
    assert coerce_value("geohash", "9q8yyk8ytpxr") == "9q8yyk8ytpxr"
    assert not validate_value("geohash", "ABC!")  # 'a','l' etc excluded + '!'
    assert not validate_value("geohash", "io")  # 'i','o' not in base-32 alphabet


def test_geoshape_geojson():
    poly = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
    assert coerce_value("geoshape", poly)["type"] == "Polygon"
    assert (
        coerce_value("geoshape", '{"type":"Point","coordinates":[1,2]}')["type"]
        == "Point"
    )
    assert not validate_value("geoshape", {"type": "Blob"})


def test_timeseries_and_geotimeseries():
    series = coerce_value(
        "timeseries", [["2026-06-09T00:00:00Z", 1.5], ["2026-06-09T01:00:00Z", 2]]
    )
    assert len(series["points"]) == 2
    assert series["points"][0][1] == 1.5
    # reference-id form
    assert coerce_value("timeseries", "ri.series.123")["series_id"] == "ri.series.123"

    gts = coerce_value(
        "geotimeseries",
        [["2026-06-09T00:00:00Z", {"lat": 37.0, "lon": -122.0}]],
    )
    assert gts["points"][0]["geo"] == {"lat": 37.0, "lon": -122.0}
    # latitude out of range rejected
    assert not validate_value(
        "geotimeseries", [["2026-06-09T00:00:00Z", {"lat": 200, "lon": 0}]]
    )


# --- Complex / reference types ----------------------------------------------
def test_struct():
    assert coerce_value("struct", {"a": 1}) == {"a": 1}
    assert coerce_value("struct", '{"b": 2}') == {"b": 2}
    assert not validate_value("struct", 5)


def test_attachment_and_media_reference():
    att = coerce_value(
        "attachment", {"rid": "ri.att.1", "filename": "x.pdf", "size": "10"}
    )
    assert att == {"rid": "ri.att.1", "filename": "x.pdf", "size": 10}
    assert coerce_value("attachment", "ri.att.2")["rid"] == "ri.att.2"
    assert not validate_value("attachment", {"filename": "no-rid.pdf"})

    media = coerce_value(
        "media_reference",
        {
            "media_set_rid": "ri.set.1",
            "media_item_rid": "ri.item.1",
            "media_type": "image/png",
        },
    )
    assert media["media_item_rid"] == "ri.item.1"
    assert media["media_set_rid"] == "ri.set.1"


def test_marking():
    m = coerce_value("marking", ["SECRET", "PII", "SECRET"])
    assert m == {"marking_ids": ["SECRET", "PII"]}
    assert coerce_value("marking", "TOPSECRET") == {"marking_ids": ["TOPSECRET"]}
    assert not validate_value("marking", [])


# --- Array<T> ----------------------------------------------------------------
def test_array_of_scalars():
    assert coerce_value("array<integer>", ["1", 2, 3]) == [1, 2, 3]
    assert coerce_value("array<string>", [1, "x"]) == ["1", "x"]
    assert coerce_value("list<long>", "[10, 20]") == [10, 20]
    # element validation propagates
    assert not validate_value("array<byte>", [1, 999])


# --- Vector / embedding ------------------------------------------------------
def test_vector_default_dim():
    pt = get_property_type("vector")
    assert pt.dimension == DEFAULT_VECTOR_DIM
    vec = coerce_value("vector", [0.0] * DEFAULT_VECTOR_DIM)
    assert len(vec) == DEFAULT_VECTOR_DIM
    assert all(isinstance(x, float) for x in vec)


def test_vector_dim_mismatch_rejected():
    assert not validate_value("vector", [1.0, 2.0, 3.0])


def test_vector_parameterized_dim():
    pt = parse_type_ref("vector<4>")
    assert pt.dimension == 4
    assert pt.storage_hint == "FLOAT[4]"
    assert coerce_value("vector<4>", [1, 2, 3, 4]) == [1.0, 2.0, 3.0, 4.0]
    assert not validate_value("vector<4>", [1, 2, 3])


def test_embedding_from_numpy_like():
    class _FakeArray:
        def __init__(self, data):
            self._data = data

        def tolist(self):
            return self._data

    pt = parse_type_ref(f"embedding<{3}>")
    assert pt.coerce(_FakeArray([1.0, 2.0, 3.0])) == [1.0, 2.0, 3.0]


# --- schema_definition column-type bridge ------------------------------------
@pytest.mark.parametrize(
    "type_ref,expected",
    [
        ("string", "STRING"),
        ("boolean", "BOOLEAN"),
        ("long", "INT64"),
        ("double", "DOUBLE"),
        ("float", "FLOAT"),
        ("timestamp", "TIMESTAMP"),
        ("array<string>", "STRING[]"),
        ("marking", "STRING[]"),
        ("vector", f"FLOAT[{DEFAULT_VECTOR_DIM}]"),
        ("vector<16>", "FLOAT[16]"),
    ],
)
def test_column_type_mapping(type_ref, expected):
    assert column_type_for(type_ref) == expected


def test_column_types_are_valid_schema_strings():
    """Every storage hint must be a known schema_definition column form."""
    from agent_utilities.models.knowledge_graph import TableDefinition

    cols = {name: column_type_for(name) for name in list_property_types()}
    # Constructing a TableDefinition validates the dict[str,str] contract.
    td = TableDefinition(
        name="PropTypeProbe", columns={"id": "STRING PRIMARY KEY", **cols}
    )
    assert td.columns["timestamp"] == "TIMESTAMP"
    assert td.columns["embedding"].startswith("FLOAT[")


def test_aliases_resolve():
    assert get_property_type("datetime").name == "timestamp"
    assert get_property_type("int").name == "integer"
    assert get_property_type("json").name == "struct"
    assert get_property_type("nonsense-type") is None


def test_xsd_iris_present():
    assert PROPERTY_TYPES["timestamp"].xsd_iri.endswith("#dateTime")
    assert PROPERTY_TYPES["integer"].xsd_iri.endswith("#int")
    assert PROPERTY_TYPES["string"].xsd_iri.endswith("#string")
