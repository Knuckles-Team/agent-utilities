"""Tests for the curated trading TeamConfig seed — CONCEPT:EE-039."""

from agent_utilities.graph.trading_team_seed import (
    TRADING_TEAM_ID,
    build_trading_team_config,
    seed_trading_team,
)
from agent_utilities.models.knowledge_graph import RegistryNodeType


def test_build_trading_team_config_shape():
    tc = build_trading_team_config()
    assert tc.id == TRADING_TEAM_ID
    assert tc.type == RegistryNodeType.TEAM_CONFIG
    # Reusable as a proven team: success_rate above the reuse threshold.
    assert tc.success_rate >= tc.reuse_threshold
    # Execution tools concentrated on exactly one specialist (paper-gate point).
    holders = [
        sid
        for sid, tools in tc.tool_assignments.items()
        if "emerald_orders" in tools
    ]
    assert holders == ["execution-specialist"]
    # Risk manager gets the adversarial critic capability.
    assert "critic" in tc.capability_overrides["risk-manager"]


def test_seed_trading_team_writes_to_graph():
    class _FakeGraph:
        def __init__(self):
            self.nodes = {}

        def add_node(self, node_id, **attrs):
            self.nodes[node_id] = attrs

    class _FakeEngine:
        def __init__(self):
            self.graph = _FakeGraph()
            self.backend = None

    engine = _FakeEngine()
    nid = seed_trading_team(engine)
    assert nid == TRADING_TEAM_ID
    assert TRADING_TEAM_ID in engine.graph.nodes
    assert engine.graph.nodes[TRADING_TEAM_ID]["task_pattern"]


def test_seed_trading_team_no_graph_noop():
    class _Bare:
        graph = None

    assert seed_trading_team(_Bare()) is None
