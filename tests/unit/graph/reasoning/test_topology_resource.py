"""Unit tests for the versioned, graph-addressable topology resource."""

from agent_utilities.graph.reasoning.cot import COT_SPEC
from agent_utilities.graph.reasoning.rap import RAP_SPEC
from agent_utilities.graph.reasoning.topology import (
    TopologySpec,
    register_topology,
)


def test_digest_is_stable_across_calls():
    assert COT_SPEC.digest == COT_SPEC.digest


def test_digest_differs_across_topologies():
    assert COT_SPEC.digest != RAP_SPEC.digest


def test_digest_changes_when_node_contracts_change():
    other = TopologySpec(
        name=COT_SPEC.name,
        version=COT_SPEC.version,
        node_contracts=(*COT_SPEC.node_contracts, "extra"),
        loop_budget=COT_SPEC.loop_budget,
    )
    assert other.digest != COT_SPEC.digest


def test_topology_id_is_graph_addressable():
    assert COT_SPEC.topology_id == f"topology:cot:{COT_SPEC.digest}"


def test_to_node_carries_budgets_and_termination_conditions():
    node = RAP_SPEC.to_node()
    assert node.artifact_kind == "reasoning_topology"
    assert node.artifact_id == "rap"
    assert node.version_hash == RAP_SPEC.digest
    assert node.loop_budget == RAP_SPEC.loop_budget
    assert set(node.termination_conditions) == set(RAP_SPEC.termination_conditions)
    assert node.checkpoint_semantics


class _StubBackend:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def execute(self, query, params):
        self.calls.append((query, params))


class _StubEngine:
    def __init__(self):
        self.nodes: dict[str, tuple[str, dict]] = {}
        self.backend = _StubBackend()

    def add_node(self, node_id, node_type, props):
        self.nodes[node_id] = (node_type, props)


def test_register_topology_writes_a_typed_node():
    engine = _StubEngine()
    register_topology(engine, COT_SPEC)
    node_type, props = engine.nodes[COT_SPEC.topology_id]
    assert node_type == "reasoning_topology_version"
    assert props["artifact_id"] == "cot"
    assert props["version_hash"] == COT_SPEC.digest


def test_register_topology_never_raises_without_an_engine():
    register_topology(None, COT_SPEC)  # must not raise


def test_the_resource_keeps_no_outcome_store():
    """EH-474: registration is the only write; no reward/task_count is ever set."""
    engine = _StubEngine()
    register_topology(engine, COT_SPEC)
    _node_type, props = engine.nodes[COT_SPEC.topology_id]
    assert props["reward"] is None
    assert props["task_count"] == 0
    assert engine.backend.calls == []
    import agent_utilities.graph.reasoning as reasoning

    assert not hasattr(reasoning, "record_topology_outcome")
