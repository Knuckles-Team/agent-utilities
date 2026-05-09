import numpy as np
import pytest

from agent_utilities.knowledge_graph.core.markov_transitions import (
    MarkovTransitionModel,
)


def test_markov_transition_model_ingest():
    model = MarkovTransitionModel()
    trace = ["A", "B", "A", "C", "C"]
    model.ingest_trace(trace)

    assert "A" in model.states
    assert "B" in model.states
    assert "C" in model.states

    # A -> B (1), A -> C (1). Total from A = 2.
    assert model.get_transition_probability("A", "B") == 0.5
    assert model.get_transition_probability("A", "C") == 0.5

    # B -> A (1). Total from B = 1.
    assert model.get_transition_probability("B", "A") == 1.0

    # C -> C (1). Total from C = 1.
    assert model.get_transition_probability("C", "C") == 1.0


def test_stationary_distribution():
    model = MarkovTransitionModel()

    # Trace where it eventually gets stuck in "SINK"
    trace1 = ["START", "PROCESS", "PROCESS", "SINK", "SINK", "SINK"]
    trace2 = ["START", "SINK", "SINK"]

    model.ingest_trace(trace1)
    model.ingest_trace(trace2)

    stat_dist = model.stationary_distribution()

    # Since SINK is an absorbing state (it only transitions to itself),
    # the stationary distribution should have all probability mass on SINK.
    assert "SINK" in stat_dist
    assert stat_dist["SINK"] > 0.99
    assert stat_dist["START"] < 0.01


def test_predict_sink_nodes():
    model = MarkovTransitionModel()
    model.ingest_trace(["A", "B", "C", "C", "C"])

    sinks = model.predict_sink_nodes(threshold=0.5)
    assert len(sinks) == 1
    assert sinks[0][0] == "C"
    assert sinks[0][1] > 0.99
