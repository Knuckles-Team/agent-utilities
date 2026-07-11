"""Tests for the versioned capability descriptor (X-4).

Covers the descriptor dataclass round-trip to/from engine node properties, the
AU-side in-process registry, and durable persist/load against a fake Cypher
backend (mirrors the pattern in ``tests/unit/graph/test_capability_designation.py``).
"""

from __future__ import annotations

import re
import types
from typing import Any

import pytest

from agent_utilities.knowledge_graph.retrieval.capability_descriptor import (
    APPROVAL_AUTO,
    APPROVAL_HUMAN_REQUIRED,
    SIDE_EFFECT_DESTRUCTIVE,
    CapabilityDescriptor,
    CapabilityDescriptorRegistry,
    load_capability_descriptor,
    persist_capability_descriptor,
)


_RETURN_ALIAS_RE = re.compile(r"n\.(\w+)\s+AS\s+(\w+)")
_SET_ASSIGN_RE = re.compile(r"n\.(\w+)\s*=\s*\$(\w+)")


class _FakeCypherBackend:
    """A minimal in-memory Cypher executor honouring ``SET n.x = $y`` assignments
    and ``RETURN n.x AS y`` aliases by NAME, not by coincidental param-key match.

    Faithful enough to serve BOTH ``durable_outcome_store`` (SET params named
    ``r``/``c``/``ts``, aliased back out as ``reward``/``count``) and
    ``capability_descriptor`` (SET/RETURN params named after the property itself)
    against the SAME in-memory node store — exactly what a real engine's property
    projection does, and exactly why ``capability_descriptor`` never needs its own
    reward properties: they live on the identical node.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}

    def execute(self, query: str, params: dict[str, Any] | None = None):
        params = params or {}
        nid = str(params.get("id"))
        if "SET" in query:
            node = self.nodes.setdefault(nid, {})
            for prop, param in _SET_ASSIGN_RE.findall(query):
                node[prop] = params.get(param)
            return []
        node = self.nodes.get(nid, {})
        row = {alias: node.get(prop) for prop, alias in _RETURN_ALIAS_RE.findall(query)}
        return [row]


def _make_engine(backend: Any = None) -> Any:
    return types.SimpleNamespace(backend=backend)


# ---------------------------------------------------------------------------
# Dataclass round-trip
# ---------------------------------------------------------------------------
def test_default_approval_class_is_auto_for_a_non_destructive_capability():
    d = CapabilityDescriptor(id="tool:x", capability_type="DNSCapability")
    assert d.approval_class == APPROVAL_AUTO


def test_destructive_side_effect_defaults_to_human_approval_required():
    d = CapabilityDescriptor(
        id="tool:x", capability_type="DNSCapability", side_effects=(SIDE_EFFECT_DESTRUCTIVE,)
    )
    assert d.approval_class == APPROVAL_HUMAN_REQUIRED


def test_explicit_approval_class_overrides_the_default():
    d = CapabilityDescriptor(
        id="tool:x",
        capability_type="DNSCapability",
        side_effects=(SIDE_EFFECT_DESTRUCTIVE,),
        approval_class=APPROVAL_AUTO,
    )
    assert d.approval_class == APPROVAL_AUTO


def test_to_from_node_properties_round_trips_every_declared_field():
    d = CapabilityDescriptor(
        id="tool:dns",
        capability_type="DNSCapability",
        version="2.1.0",
        input_schema={"query": "string"},
        output_schema={"records": "list"},
        side_effects=("read", "external_call"),
        required_data_types=("dns_zone",),
        required_resource_types=("network",),
        tenant_scopes=("tenant-a",),
        authz_scopes=("dns:read",),
        cost_estimate=0.002,
        latency_ms_estimate=45.0,
        locality="local",
        policy_class="restricted",
    )
    props = d.to_node_properties()
    # Reward-related properties are never written by this module.
    assert "capability_reward" not in props
    assert "reliability" not in props

    reconstructed = CapabilityDescriptor.from_node_properties(
        "tool:dns", {**props, "capability_reward": 0.72, "capability_reward_count": 4}
    )
    assert reconstructed.capability_type == "DNSCapability"
    assert reconstructed.version == "2.1.0"
    assert reconstructed.input_schema == {"query": "string"}
    assert reconstructed.side_effects == ("read", "external_call")
    assert reconstructed.required_data_types == ("dns_zone",)
    assert reconstructed.tenant_scopes == ("tenant-a",)
    assert reconstructed.authz_scopes == ("dns:read",)
    assert reconstructed.cost_estimate == pytest.approx(0.002)
    assert reconstructed.latency_ms_estimate == pytest.approx(45.0)
    assert reconstructed.locality == "local"
    assert reconstructed.policy_class == "restricted"
    # Reliability/success history is derived from the bandit's OWN properties.
    assert reconstructed.reliability == pytest.approx(0.72)
    assert reconstructed.success_count == 4


def test_from_node_properties_on_empty_props_yields_safe_defaults():
    d = CapabilityDescriptor.from_node_properties("tool:bare", {})
    assert d.capability_type == ""
    assert d.version == "1.0.0"
    assert d.reliability == 0.5
    assert d.success_count == 0
    assert d.side_effects == ()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_registry_register_get_remove():
    reg = CapabilityDescriptorRegistry()
    d = CapabilityDescriptor(id="tool:a", capability_type="DNSCapability")
    reg.register(d)
    assert len(reg) == 1
    assert "tool:a" in reg
    assert reg.get("tool:a") is d
    assert reg.get("missing") is None
    assert reg.remove("tool:a") is True
    assert "tool:a" not in reg
    assert reg.remove("tool:a") is False


def test_registry_all_returns_every_registered_descriptor():
    reg = CapabilityDescriptorRegistry()
    reg.register(CapabilityDescriptor(id="a", capability_type="DNSCapability"))
    reg.register(CapabilityDescriptor(id="b", capability_type="CRMCapability"))
    assert {d.id for d in reg.all()} == {"a", "b"}


def test_registry_hydrate_from_engine_reads_declared_properties():
    engine = types.SimpleNamespace(
        graph=types.SimpleNamespace(
            _get_node_properties=lambda nid: {
                "tool:a": {"capability_type": "DNSCapability", "capability_version": "3.0.0"},
            }.get(nid, {})
        )
    )
    reg = CapabilityDescriptorRegistry()
    count = reg.hydrate_from_engine(engine, ["tool:a", "tool:unknown"])
    assert count == 2
    assert reg.get("tool:a").capability_type == "DNSCapability"
    assert reg.get("tool:a").version == "3.0.0"
    # An id with no descriptor properties still yields a minimal descriptor.
    assert reg.get("tool:unknown") is not None
    assert reg.get("tool:unknown").capability_type == ""


def test_registry_hydrate_from_engine_never_raises_without_a_graph_surface():
    engine = types.SimpleNamespace()
    reg = CapabilityDescriptorRegistry()
    count = reg.hydrate_from_engine(engine, ["tool:a"])
    assert count == 1
    assert reg.get("tool:a") is not None


# ---------------------------------------------------------------------------
# Durable persistence
# ---------------------------------------------------------------------------
def test_persist_then_load_round_trips_through_a_fake_backend():
    engine = _make_engine(_FakeCypherBackend())
    d = CapabilityDescriptor(
        id="tool:dns",
        capability_type="DNSCapability",
        version="1.2.0",
        side_effects=("read",),
        cost_estimate=0.01,
        locality="local",
    )
    assert persist_capability_descriptor(engine, d) is True

    loaded = load_capability_descriptor(engine, "tool:dns")
    assert loaded is not None
    assert loaded.capability_type == "DNSCapability"
    assert loaded.version == "1.2.0"
    assert loaded.side_effects == ("read",)
    assert loaded.cost_estimate == pytest.approx(0.01)
    assert loaded.locality == "local"


def test_persist_never_writes_the_bandit_owned_reward_properties():
    engine = _make_engine(_FakeCypherBackend())
    d = CapabilityDescriptor(id="tool:dns", capability_type="DNSCapability")
    persist_capability_descriptor(engine, d)
    stored = engine.backend.nodes["tool:dns"]
    assert "capability_reward" not in stored
    assert "capability_reward_count" not in stored


def test_persist_without_a_backend_returns_false_not_raise():
    engine = _make_engine(None)
    d = CapabilityDescriptor(id="tool:dns", capability_type="DNSCapability")
    assert persist_capability_descriptor(engine, d) is False


def test_load_without_a_backend_returns_none_not_raise():
    engine = _make_engine(None)
    assert load_capability_descriptor(engine, "tool:dns") is None


def test_load_unknown_id_returns_a_minimal_descriptor_not_none():
    engine = _make_engine(_FakeCypherBackend())
    loaded = load_capability_descriptor(engine, "no_such_tool")
    # The fake backend always returns a (possibly empty) row for MATCH...RETURN,
    # so this is a minimal/default descriptor, not a hard failure.
    assert loaded is not None
    assert loaded.capability_type == ""


def test_reliability_reflects_the_durable_bandit_reward_after_outcome_persist():
    """A CapabilityDescriptor reloaded after the durable bandit records an outcome
    (via durable_outcome_store, reusing the SAME node) reflects the live reward —
    proving the descriptor never carries a second, driftable reward copy."""
    from agent_utilities.knowledge_graph.retrieval.durable_outcome_store import (
        persist_capability_reward,
    )

    engine = _make_engine(_FakeCypherBackend())
    d = CapabilityDescriptor(id="tool:dns", capability_type="DNSCapability")
    persist_capability_descriptor(engine, d)

    persist_capability_reward(engine, "tool:dns", success=True, alpha=1.0)

    reloaded = load_capability_descriptor(engine, "tool:dns")
    assert reloaded.reliability == pytest.approx(1.0)
    assert reloaded.success_count == 1
    # The descriptor's own fields survive the reward write untouched.
    assert reloaded.capability_type == "DNSCapability"
