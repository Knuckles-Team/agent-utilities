"""Characterization tests for ``CapabilityDescriptor.from_node_properties`` (CX-AU-08).

Pins observed behaviour before refactor, including edge cases not covered by
``tests/unit/knowledge_graph/test_capability_descriptor.py``:

* the three-way ``capability_type``/``type``/``node_type`` fallback order,
* ``cost_estimate``/``latency_ms_estimate`` being ``None`` (not ``0.0``) when
  absent, distinguishing "unset" from "zero",
* ``reliability`` using an explicit ``is not None`` check on
  ``capability_reward`` -- so a reward of exactly ``0`` yields
  ``reliability == 0.0``, NOT the 0.5 "unproven" default,
* a scalar string value for a tuple-typed field (e.g. ``side_effects="read"``)
  being wrapped as a 1-tuple rather than iterated character-by-character.

CX-AU-08 owns only ``agent_utilities/knowledge_graph/retrieval``; this test
file is added in isolation (commit 1) and must be byte-identical across
commit 2 (the refactor).
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.retrieval.capability_descriptor import (
    CapabilityDescriptor,
)


def test_empty_props_yields_documented_defaults():
    d = CapabilityDescriptor.from_node_properties("tool:bare", {})
    assert d.capability_type == ""
    assert d.version == "1.0.0"
    assert d.input_schema == {}
    assert d.output_schema == {}
    assert d.side_effects == ()
    assert d.required_data_types == ()
    assert d.required_resource_types == ()
    assert d.tenant_scopes == ()
    assert d.authz_scopes == ()
    assert d.cost_estimate is None
    assert d.latency_ms_estimate is None
    assert d.locality is None
    assert d.policy_class == "standard"
    assert d.reliability == 0.5
    assert d.success_count == 0
    assert d.updated_at is None


def test_capability_type_prefers_capability_type_over_type_and_node_type():
    d = CapabilityDescriptor.from_node_properties(
        "x", {"capability_type": "A", "type": "B", "node_type": "C"}
    )
    assert d.capability_type == "A"


def test_capability_type_falls_back_to_type_when_capability_type_absent():
    d = CapabilityDescriptor.from_node_properties("x", {"type": "B", "node_type": "C"})
    assert d.capability_type == "B"


def test_capability_type_falls_back_to_node_type_last():
    d = CapabilityDescriptor.from_node_properties("x", {"node_type": "C"})
    assert d.capability_type == "C"


def test_cost_estimate_absent_is_none_not_zero():
    d = CapabilityDescriptor.from_node_properties("x", {})
    assert d.cost_estimate is None


def test_cost_estimate_explicit_zero_is_preserved_as_zero_not_none():
    d = CapabilityDescriptor.from_node_properties("x", {"cost_estimate": 0})
    assert d.cost_estimate == 0.0


def test_latency_estimate_explicit_zero_is_preserved():
    d = CapabilityDescriptor.from_node_properties("x", {"latency_ms_estimate": 0})
    assert d.latency_ms_estimate == 0.0


def test_reliability_defaults_to_point_five_when_reward_absent():
    d = CapabilityDescriptor.from_node_properties("x", {})
    assert d.reliability == 0.5


def test_reliability_zero_reward_is_zero_not_the_default():
    """capability_reward=0 is a real observed reward of zero, distinct from
    'no reward recorded yet' -- pins the `is not None` check, not truthiness."""
    d = CapabilityDescriptor.from_node_properties("x", {"capability_reward": 0})
    assert d.reliability == 0.0


def test_reliability_reads_from_capability_reward_property():
    d = CapabilityDescriptor.from_node_properties("x", {"capability_reward": 0.72})
    assert d.reliability == 0.72


def test_success_count_reads_capability_reward_count():
    d = CapabilityDescriptor.from_node_properties("x", {"capability_reward_count": 7})
    assert d.success_count == 7


def test_success_count_defaults_to_zero_when_absent():
    d = CapabilityDescriptor.from_node_properties("x", {})
    assert d.success_count == 0


def test_tuple_field_scalar_string_is_wrapped_not_iterated():
    """side_effects="read" must become ("read",), not ("r","e","a","d")."""
    d = CapabilityDescriptor.from_node_properties("x", {"side_effects": "read"})
    assert d.side_effects == ("read",)


def test_tuple_field_list_is_stringified_elementwise():
    d = CapabilityDescriptor.from_node_properties(
        "x", {"required_data_types": ["a", 1, None]}
    )
    assert d.required_data_types == ("a", "1", "None")


def test_tuple_field_absent_is_empty_tuple():
    d = CapabilityDescriptor.from_node_properties("x", {})
    assert d.required_resource_types == ()


def test_version_defaults_when_absent():
    d = CapabilityDescriptor.from_node_properties("x", {})
    assert d.version == "1.0.0"


def test_version_reads_capability_version_property():
    d = CapabilityDescriptor.from_node_properties("x", {"capability_version": "3.2.1"})
    assert d.version == "3.2.1"


def test_policy_class_defaults_to_standard():
    d = CapabilityDescriptor.from_node_properties("x", {})
    assert d.policy_class == "standard"


def test_policy_class_reads_explicit_value():
    d = CapabilityDescriptor.from_node_properties("x", {"policy_class": "restricted"})
    assert d.policy_class == "restricted"


def test_approval_class_passthrough_when_present():
    d = CapabilityDescriptor.from_node_properties("x", {"approval_class": "auto"})
    assert d.approval_class == "auto"


def test_approval_class_absent_falls_back_to_post_init_default():
    # approval_class not in props -> None passed to the dataclass constructor,
    # so __post_init__'s own fail-closed defaulting takes over.
    d = CapabilityDescriptor.from_node_properties("x", {})
    assert d.approval_class == "auto"


def test_locality_passthrough():
    d = CapabilityDescriptor.from_node_properties("x", {"locality": "us-east"})
    assert d.locality == "us-east"


def test_updated_at_passthrough():
    d = CapabilityDescriptor.from_node_properties(
        "x", {"descriptor_updated_at": "2026-01-01T00:00:00.000000Z"}
    )
    assert d.updated_at == "2026-01-01T00:00:00.000000Z"


def test_input_output_schema_are_copied_dicts():
    schema = {"a": 1}
    d = CapabilityDescriptor.from_node_properties("x", {"input_schema": schema})
    assert d.input_schema == {"a": 1}
    assert d.input_schema is not schema
