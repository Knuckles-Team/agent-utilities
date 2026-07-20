#!/usr/bin/python
"""Tests for the Multi-Agent Social System swarm model (b2-01 MASS).

CONCEPT:AU-ORCH.dispatch.kg-governed-agent-swarm
"""

import pytest

from agent_utilities.graph.social_system import MultiAgentSocialSystem

pytestmark = pytest.mark.concept("AU-ORCH.dispatch.kg-governed-agent-swarm")


def _sys():
    m = MultiAgentSocialSystem()
    m.add_agent("a", archetype="explorer", latent_state=1.0)
    m.add_agent("b", archetype="critic", latent_state=2.0)
    m.add_agent("c", archetype="critic", latent_state=3.0)
    return m


# --- F3: local observability ----------------------------------------------
def test_observable_messages_restricted_to_neighborhood():
    m = _sys()
    m.add_edge("a", "b")  # a sees b (and itself), not c
    msgs = {"a": "m_a", "b": "m_b", "c": "m_c"}
    obs = m.observable_messages("a", msgs)
    assert set(obs) == {"a", "b"}
    assert "c" not in obs


# --- F2: strategic heterogeneity ------------------------------------------
def test_archetype_distribution_and_heterogeneity():
    m = _sys()
    assert m.archetype_distribution() == {"explorer": 1, "critic": 2}
    assert 0.0 < m.heterogeneity() <= 1.0


def test_heterogeneity_zero_when_homogeneous():
    m = MultiAgentSocialSystem()
    for i in range(3):
        m.add_agent(f"x{i}", archetype="worker")
    assert m.heterogeneity() == 0.0  # single archetype → collapsed


# --- F4: co-evolution ------------------------------------------------------
def test_co_evolve_adds_edges_and_centrality():
    m = _sys()
    cent = m.co_evolve([("a", "b"), ("b", "c")])
    assert m.degree("b") == 2  # b now interacts with a and c
    assert cent["b"] > cent["a"]  # b is more central


# --- F6: P1–P4 swarm health ------------------------------------------------
def test_swarm_health_reports_p1_to_p4():
    m = _sys()
    m.co_evolve([("a", "b"), ("b", "c")])
    health = m.swarm_health(prev_states=[1.0, 1.0, 1.0])
    assert health["agents"] == 3
    assert 0.0 <= health["heterogeneity"] <= 1.0
    assert health["topology_variance"] >= 0.0
    assert "coevolution_slope" in health
    assert health["w1_drift"] > 0.0  # states moved from the prev distribution


def test_swarm_health_no_drift_without_prev():
    m = _sys()
    assert m.swarm_health()["w1_drift"] == 0.0


# --- live path: ParallelEngine snapshot ------------------------------------
def test_parallel_engine_social_health_live():
    from agent_utilities.graph.parallel_engine import ParallelEngine
    from agent_utilities.models.execution_manifest import (
        AgentExecutionResult,
        AgentSpec,
        ExecutionManifest,
        SynthesisSpec,
    )

    manifest = ExecutionManifest(
        name="mass-test",
        agents=[
            AgentSpec(agent_id="researcher", role="researcher", task_template="t"),
            AgentSpec(
                agent_id="auditor",
                role="auditor",
                task_template="t",
                depends_on=["researcher"],
            ),
        ],
        execution_mode="parallel",
        query="analyze",
        synthesis=SynthesisSpec(strategy="flat"),
    )
    results = [
        AgentExecutionResult(
            agent_id="researcher", role="researcher", output="a" * 20, success=True
        ),
        AgentExecutionResult(
            agent_id="auditor", role="auditor", output="b" * 5, success=True
        ),
    ]
    engine = ParallelEngine()
    health = engine._social_swarm_health(results, manifest)
    assert health["agents"] == 2
    assert set(health["archetypes"]) == {"researcher", "auditor"}
    # the depends_on edge formed an interaction link
    assert health["degree_centrality"]["auditor"] > 0.0


def test_parallel_engine_social_health_single_agent_noop():
    from agent_utilities.graph.parallel_engine import ParallelEngine
    from agent_utilities.models.execution_manifest import AgentExecutionResult

    engine = ParallelEngine()
    out = engine._social_swarm_health(
        [AgentExecutionResult(agent_id="solo", output="x", success=True)],
        type("M", (), {"agents": []})(),
    )
    assert out == {}
