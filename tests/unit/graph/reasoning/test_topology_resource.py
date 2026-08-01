"""Unit tests for the versioned, graph-addressable topology resource."""

import pytest

from agent_utilities.graph.reasoning.budgets import TerminationProof, TerminationReason
from agent_utilities.graph.reasoning.cot import COT_SPEC
from agent_utilities.graph.reasoning.rap import RAP_SPEC
from agent_utilities.graph.reasoning.topology import (
    TopologySpec,
    record_topology_outcome,
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


def test_record_topology_outcome_never_scores_a_degraded_run_as_success():
    engine = _StubEngine()
    proof = TerminationProof(
        reason=TerminationReason.LOOP_BUDGET_EXHAUSTED, success=True, degraded=True
    )
    record_topology_outcome(
        engine, COT_SPEC.topology_id, success=True, quality_score=0.9, proof=proof
    )
    # Two calls: a bounded read to resolve the prior reward/task_count (D-W2C-5
    # -- the write's SET values must be literals, so the EMA math moved to
    # Python), then the literal-valued write. A degraded run's EMA target is
    # score=0.0, so starting from the default prior reward (0.5) the new
    # reward must move DOWN, never reflect a clean-success update.
    assert len(engine.backend.calls) == 2
    _read_query, read_params = engine.backend.calls[0]
    assert read_params == {"tid": COT_SPEC.topology_id}
    _write_query, write_params = engine.backend.calls[1]
    assert write_params["reward"] < 0.5
    assert write_params["task_count"] == 1


def test_record_topology_outcome_scores_clean_success():
    engine = _StubEngine()
    record_topology_outcome(
        engine, COT_SPEC.topology_id, success=True, quality_score=0.9
    )
    _write_query, write_params = engine.backend.calls[-1]
    # EMA from the default prior reward (0.5): 0.5*(1-0.15) + 0.15*0.9 = 0.56.
    assert write_params["reward"] == pytest.approx(0.56)
    assert write_params["task_count"] == 1


def test_record_topology_outcome_never_raises_without_a_backend():
    record_topology_outcome(None, COT_SPEC.topology_id, success=True)  # no raise

    class _NoBackendEngine:
        backend = None

    record_topology_outcome(
        _NoBackendEngine(), COT_SPEC.topology_id, success=True
    )  # no raise
