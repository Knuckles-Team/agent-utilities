#!/usr/bin/python
from __future__ import annotations

"""Tests for CONCEPT:AU-KG.memory.tiered-memory-caching — Persistent Self-Model."""


from dataclasses import dataclass, field

from agent_utilities.knowledge_graph.retrieval.memory_retriever import (
    SELF_MODEL_ANCHOR,
    MemoryRetriever,
)
from agent_utilities.models.knowledge_graph import (
    RegistryNodeType,
)


class FakeEngine:  # type: ignore
    """Minimal mock engine for self-model tests."""

    def __init__(self):
        from agent_utilities.knowledge_graph.core.graph_compute import (
            GraphComputeEngine,
        )

        self.graph = GraphComputeEngine(backend_type="rust")
        self.backend = None

    def _upsert_node(self, label, node_id, props):
        self.graph.add_node(node_id, label=label, **props)

    def link_nodes(self, src, tgt, rel_type, props=None):
        self.graph.add_edge(src, tgt, type=rel_type, **(props or {}))


@dataclass
class FakeTaskList:
    tasks: list = field(default_factory=list)


@dataclass
class FakeGraphState:  # type: ignore
    """Minimal mock GraphState for session aggregation tests."""

    session_id: str = "sess:test"
    routed_domain: str = "gitlab"
    error: str | None = None
    node_history: list[str] = field(default_factory=list)
    task_list: FakeTaskList = field(default_factory=FakeTaskList)
    routing_confidence_log: dict = field(default_factory=dict)


class TestMemoryRetrieverCreation:
    def test_get_or_create_initializes(self):
        engine = FakeEngine()  # type: ignore
        sm = MemoryRetriever(engine)  # type: ignore

        node = sm.get_or_create()
        assert node is not None
        assert node.version == 1
        assert node.type == RegistryNodeType.SELF_MODEL
        assert SELF_MODEL_ANCHOR in engine.graph

    def test_get_or_create_returns_existing(self):
        engine = FakeEngine()  # type: ignore
        sm = MemoryRetriever(engine)  # type: ignore

        first = sm.get_or_create()
        second = sm.get_or_create()
        assert first.id == second.id

    def test_get_current_returns_none_initially(self):
        engine = FakeEngine()  # type: ignore
        sm = MemoryRetriever(engine)  # type: ignore

        current = sm.get_current()
        assert current is None


class TestMemoryRetrieverSnapshots:
    def test_create_snapshot_increments_version(self):
        engine = FakeEngine()  # type: ignore
        sm = MemoryRetriever(engine)  # type: ignore

        v1 = sm.get_or_create()
        v2 = sm.create_snapshot(session_id="sess:1")

        assert v2.version == v1.version + 1

    def test_create_snapshot_carries_forward_data(self):
        engine = FakeEngine()  # type: ignore
        sm = MemoryRetriever(engine)  # type: ignore

        v1 = sm.get_or_create()
        # Modify v1 data
        v1.domain_success_rates = {"gitlab": 0.9}
        v1.total_sessions = 5
        sm.ogm.upsert(v1)

        v2 = sm.create_snapshot()
        assert v2.domain_success_rates == {"gitlab": 0.9}
        assert v2.total_sessions == 5

    def test_create_snapshot_moves_current_pointer(self):
        engine = FakeEngine()  # type: ignore
        sm = MemoryRetriever(engine)  # type: ignore

        sm.get_or_create()  # initializes KG; return value needed only for side-effect
        v2 = sm.create_snapshot()

        current = sm.get_current()
        assert current is not None
        assert current.id == v2.id


class TestSessionAggregation:
    def test_update_after_session_increments_counters(self):
        engine = FakeEngine()  # type: ignore
        sm = MemoryRetriever(engine)  # type: ignore
        sm.get_or_create()

        session = FakeGraphState(routed_domain="gitlab")
        sm.update_after_session(session)  # type: ignore[arg-type]

        current = sm.get_current()
        assert current is not None
        assert current.total_sessions == 1

    def test_update_after_session_tracks_domain(self):
        engine = FakeEngine()  # type: ignore
        sm = MemoryRetriever(engine)  # type: ignore
        sm.get_or_create()

        session = FakeGraphState(routed_domain="gitlab")
        sm.update_after_session(session)  # type: ignore[arg-type]

        current = sm.get_current()
        assert current is not None
        assert "gitlab" in current.domain_success_rates
        assert current.domain_success_rates["gitlab"] > 0

    def test_update_after_session_tracks_failures(self):
        engine = FakeEngine()  # type: ignore
        sm = MemoryRetriever(engine)  # type: ignore
        sm.get_or_create()

        session = FakeGraphState(
            routed_domain="gitlab",
            error="Connection timeout",
        )
        sm.update_after_session(session)  # type: ignore[arg-type]

        current = sm.get_current()
        assert current is not None
        assert "Connection timeout" in current.known_failure_patterns

    def test_update_after_session_tracks_node_history(self):
        engine = FakeEngine()  # type: ignore
        sm = MemoryRetriever(engine)  # type: ignore
        sm.get_or_create()

        session = FakeGraphState(
            node_history=["router", "planner", "gitlab_agent"],
        )
        sm.update_after_session(session)  # type: ignore[arg-type]

        current = sm.get_current()
        assert current is not None
        assert current.tool_proficiency.get("gitlab_agent", 0) > 0


class TestMemoryRetrieverQuery:
    def test_query_capabilities_empty(self):
        engine = FakeEngine()  # type: ignore
        sm = MemoryRetriever(engine)  # type: ignore

        caps = sm.query_capabilities("unknown_domain")
        assert caps == {"success_rate": 0.0, "confidence": 0.0, "proficiency": 0.0}

    def test_query_capabilities_with_data(self):
        engine = FakeEngine()  # type: ignore
        sm = MemoryRetriever(engine)  # type: ignore
        v1 = sm.get_or_create()
        v1.domain_success_rates = {"gitlab": 0.85}
        v1.capability_confidence = {"gitlab": 0.9}
        v1.tool_proficiency = {"gitlab": 0.7}
        sm.ogm.upsert(v1)

        caps = sm.query_capabilities("gitlab")
        assert caps["success_rate"] == 0.85

    def test_explain_self_returns_markdown(self):
        engine = FakeEngine()  # type: ignore
        sm = MemoryRetriever(engine)  # type: ignore
        v1 = sm.get_or_create()
        v1.domain_success_rates = {"gitlab": 0.9}
        v1.total_sessions = 10
        sm.ogm.upsert(v1)

        explanation = sm.explain_self()
        assert "Agent Self-Model" in explanation
        assert "gitlab" in explanation

    def test_explain_self_without_model(self):
        engine = FakeEngine()  # type: ignore
        sm = MemoryRetriever(engine)  # type: ignore

        explanation = sm.explain_self()
        assert "No self-model available" in explanation


class TestTemporalTrend:
    def test_temporal_trend_empty(self):
        engine = FakeEngine()  # type: ignore
        sm = MemoryRetriever(engine)  # type: ignore

        trend = sm.temporal_trend("gitlab")
        assert trend == []

    def test_temporal_trend_single_version(self):
        engine = FakeEngine()  # type: ignore
        sm = MemoryRetriever(engine)  # type: ignore
        v1 = sm.get_or_create()
        v1.domain_success_rates = {"gitlab": 0.8}
        sm.ogm.upsert(v1)

        trend = sm.temporal_trend("gitlab", lookback=3)
        assert len(trend) == 1
        assert trend[0] == 0.8
