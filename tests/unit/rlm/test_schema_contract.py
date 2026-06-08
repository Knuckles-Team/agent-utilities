"""CONCEPT:ORCH-1.12 — SchemaContract normalizes every supported output-spec form.

Unit coverage for the structured-output contract used by RLM subagent fan-out:
a Pydantic model, a primitive (`int`/`bool`), a typing generic (`list[Model]`),
and a raw JSON-Schema dict (`{"type": "boolean"}`), including the degraded
shallow-check fallback used when `jsonschema` is absent.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent_utilities.rlm.schema import SchemaContract


class _Person(BaseModel):
    name: str
    age: int


@pytest.mark.concept(id="ORCH-1.12")
def test_pydantic_model_spec_validates_and_coerces():
    c = SchemaContract.from_spec(_Person)
    assert "properties" in c.json_schema
    ok, coerced, err = c.validate({"name": "Saltram", "age": 40})
    assert ok and err is None
    assert isinstance(coerced, _Person) and coerced.name == "Saltram"

    bad_ok, _, bad_err = c.validate({"name": "x"})  # missing required 'age'
    assert bad_ok is False and "age" in bad_err


@pytest.mark.concept(id="ORCH-1.12")
def test_primitive_int_spec_coerces_string():
    c = SchemaContract.from_spec(int)
    assert c.json_schema.get("type") == "integer"
    ok, coerced, err = c.validate("5")
    assert ok and coerced == 5 and err is None
    bad_ok, _, bad_err = c.validate("not-a-number")
    assert bad_ok is False and bad_err


@pytest.mark.concept(id="ORCH-1.12")
def test_primitive_bool_spec():
    c = SchemaContract.from_spec(bool)
    assert c.json_schema.get("type") == "boolean"
    ok, coerced, _ = c.validate(True)
    assert ok and coerced is True


@pytest.mark.concept(id="ORCH-1.12")
def test_generic_list_of_models_spec():
    c = SchemaContract.from_spec(list[_Person])
    assert c.json_schema.get("type") == "array"
    ok, coerced, err = c.validate([{"name": "a", "age": 1}, {"name": "b", "age": 2}])
    assert ok and err is None
    assert len(coerced) == 2 and all(isinstance(p, _Person) for p in coerced)


@pytest.mark.concept(id="ORCH-1.12")
def test_raw_jsonschema_dict_spec():
    c = SchemaContract.from_spec({"type": "boolean"})
    assert c.json_schema == {"type": "boolean"}
    ok, _, err = c.validate(True)
    assert ok and err is None
    bad_ok, _, bad_err = c.validate("nope")
    assert bad_ok is False and bad_err


@pytest.mark.concept(id="ORCH-1.12")
def test_raw_dict_shallow_fallback_when_jsonschema_absent(monkeypatch):
    """When jsonschema is unavailable we fall back to a shallow type check — never a silent pass."""
    import agent_utilities.rlm.schema as schema_mod

    monkeypatch.setattr(schema_mod, "_HAS_JSONSCHEMA", False)
    c = SchemaContract.from_spec({"type": "boolean"})
    ok, _, _ = c.validate(True)
    assert ok
    bad_ok, _, bad_err = c.validate("not-bool")
    assert bad_ok is False and "boolean" in bad_err


@pytest.mark.concept(id="ORCH-1.12")
def test_bad_spec_raises_typeerror():
    with pytest.raises(TypeError):
        SchemaContract.from_spec(object())  # not a type, dict, or model
