#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:ORCH-1.28 & OS-5.5 — Reactive Event-Sourcing & Budget Guardrail Test Suite."""

import asyncio
import time

import pytest

from agent_utilities.graph.reactive import (
    BehaviorDispatcher,
    BudgetGuard,
    BudgetTrippedException,
    EventLedger,
    reactive_behavior,
)
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.models.knowledge_graph import EventNode
from agent_utilities.observability.token_tracker import (
    TokenUsageRecord,
    TokenUsageTracker,
)

# ── Mock Engine & Context setup ──────────────────────────────────────


class MockEngine:
    """Minimal mock engine satisfying active engine references."""

    _instance = None

    def __init__(self):
        self.graph = GraphComputeEngine(backend_type="rust")
        self.backend = None

    def _upsert_node(self, label: str, node_id: str, props: dict):
        self.graph.add_node(node_id, type="event", **props)

    def link_nodes(self, src: str, tgt: str, rel_type: str, props: dict | None = None):
        self.graph.add_edge(src, tgt, type=rel_type.upper(), **(props or {}))

    @classmethod
    def get_active(cls):
        if cls._instance is None:
            cls._instance = MockEngine()
        return cls._instance


@pytest.fixture(autouse=True)
def mock_engine_singleton(monkeypatch):
    """Bypass singleton retrieving of active IntelligenceGraphEngine in EventLedger."""
    engine = MockEngine()
    monkeypatch.setattr(IntelligenceGraphEngine, "get_active", lambda: engine)
    return engine


# ── EventLedger Tests ────────────────────────────────────────────────


class TestEventLedger:
    def test_append_event_persists_and_links(self, mock_engine_singleton):
        ledger = EventLedger(engine=mock_engine_singleton)
        run_id = "run:test:1"

        # 1. Append first event
        evt1 = ledger.append_event(
            run_id=run_id,
            node_id="agent_alpha",
            event_type="task.proposed",
            payload={"task": "do research"},
            severity="info",
        )

        assert evt1.id.startswith("evt:")
        assert evt1.event_type == "task.proposed"
        assert evt1.episode_id == run_id
        assert evt1.id in mock_engine_singleton.graph

        # Verify OCCURRED_DURING edge
        assert mock_engine_singleton.graph.has_edge(evt1.id, run_id)
        edge_data = mock_engine_singleton.graph.get_edge_data(evt1.id, run_id)[0]
        assert edge_data["type"] == "OCCURRED_DURING"

        # 2. Append second event to check chronological linkage
        evt2 = ledger.append_event(
            run_id=run_id,
            node_id="agent_alpha",
            event_type="task.completed",
            payload={"status": "success"},
            severity="info",
        )

        # Verify WAS_DERIVED_FROM edge linking event 2 to event 1
        assert mock_engine_singleton.graph.has_edge(evt2.id, evt1.id)
        edge_data_lineage = mock_engine_singleton.graph.get_edge_data(evt2.id, evt1.id)[
            0
        ]
        assert edge_data_lineage["type"] == "WAS_DERIVED_FROM"

    def test_get_run_events_sorted(self, mock_engine_singleton):
        ledger = EventLedger(engine=mock_engine_singleton)
        run_id = "run:test:2"

        evt1 = ledger.append_event(run_id=run_id, node_id="a", event_type="step.1")
        evt2 = ledger.append_event(run_id=run_id, node_id="a", event_type="step.2")

        events = ledger.get_run_events(run_id)
        assert len(events) == 2
        assert events[0].id == evt1.id
        assert events[1].id == evt2.id

    def test_fork_run_time_travel(self, mock_engine_singleton):
        ledger = EventLedger(engine=mock_engine_singleton)
        run_id = "run:test:3"

        evt1 = ledger.append_event(run_id=run_id, node_id="a", event_type="init")
        evt2 = ledger.append_event(run_id=run_id, node_id="a", event_type="middle")
        ledger.append_event(run_id=run_id, node_id="a", event_type="end")

        # Fork execution up to 'middle' event
        forked = ledger.fork_run(run_id, evt2.id)
        assert len(forked) == 2
        assert forked[0].id == evt1.id
        assert forked[1].id == evt2.id


# ── BehaviorDispatcher Tests ──────────────────────────────────────────


@pytest.mark.asyncio
class TestBehaviorDispatcher:
    async def test_async_and_sync_behaviors_concurrency(self):
        dispatcher = BehaviorDispatcher()
        received_events = []

        # 1. Register async listener
        @reactive_behavior("test.event", dispatcher=dispatcher)
        async def on_test_async(event: EventNode):
            await asyncio.sleep(0.01)
            received_events.append(("async", event.id))

        # 2. Register sync listener
        @reactive_behavior("test.event", dispatcher=dispatcher)
        def on_test_sync(event: EventNode):
            received_events.append(("sync", event.id))

        # Create target event node
        evt = EventNode(
            id="evt:test:async",
            name="Test",
            event_type="test.event",
            timestamp="2026-05-24T00:00:00Z",
            episode_id="run_async",
        )

        # Dispatch and await complete concurrency
        await dispatcher.dispatch_event(evt)

        # Assert both async and sync callbacks fired successfully
        assert len(received_events) == 2
        types = [item[0] for item in received_events]
        assert "async" in types
        assert "sync" in types


# ── BudgetGuard Tests ──────────────────────────────────────────────────


class TestBudgetGuard:
    def test_time_limit_breached(self, mock_engine_singleton):
        ledger = EventLedger(engine=mock_engine_singleton)
        run_id = "run:budget:1"

        # Initialize guard starting in the past to immediately trigger time limits
        guard = BudgetGuard(max_time_seconds=0.001)
        time.sleep(0.002)

        with pytest.raises(BudgetTrippedException) as exc_info:
            guard.check_limits(run_id=run_id, ledger=ledger)

        assert exc_info.value.limit_type == "time"
        assert exc_info.value.limit_value == 0.001

        # Assert critical budget.tripped event recorded in Ledger
        events = ledger.get_run_events(run_id)
        assert len(events) == 1
        assert events[0].event_type == "budget.tripped"
        assert events[0].severity == "critical"
        assert events[0].payload["limit_type"] == "time"

    def test_token_limit_breached(self, mock_engine_singleton):
        ledger = EventLedger(engine=mock_engine_singleton)
        tracker = TokenUsageTracker()
        run_id = "run:budget:2"

        guard = BudgetGuard(max_tokens=100, token_tracker=tracker)

        # Record high token usage record
        rec = TokenUsageRecord(
            agent_name="agent_heavy",
            session_id=run_id,
            prompt_tokens=80,
            response_tokens=30,  # Total = 110 (breaches limit 100)
        )
        tracker.record(rec)

        with pytest.raises(BudgetTrippedException) as exc_info:
            guard.check_limits(run_id=run_id, ledger=ledger)

        assert exc_info.value.limit_type == "tokens"
        assert exc_info.value.limit_value == 100

        # Assert logged to ledger
        events = ledger.get_run_events(run_id)
        assert len(events) == 1
        assert events[0].event_type == "budget.tripped"
        assert events[0].payload["limit_type"] == "tokens"

    def test_spend_limit_breached(self, mock_engine_singleton):
        ledger = EventLedger(engine=mock_engine_singleton)
        tracker = TokenUsageTracker()
        run_id = "run:budget:3"

        # Limit to $0.01 max cost
        guard = BudgetGuard(
            max_cost_usd=0.01,
            token_tracker=tracker,
            prompt_cost_per_token=0.001,  # $1.00 per 1000 tokens
            response_cost_per_token=0.005,  # $5.00 per 1000 tokens
        )

        # Record token usage
        rec = TokenUsageRecord(
            agent_name="agent_heavy",
            session_id=run_id,
            prompt_tokens=5,  # Cost = 5 * 0.001 = $0.005
            response_tokens=2,  # Cost = 2 * 0.005 = $0.010 (Total = $0.015 > $0.010 limit)
        )
        tracker.record(rec)

        with pytest.raises(BudgetTrippedException) as exc_info:
            guard.check_limits(run_id=run_id, ledger=ledger)

        assert exc_info.value.limit_type == "cost"
        assert exc_info.value.limit_value == 0.01

        # Assert logged to ledger
        events = ledger.get_run_events(run_id)
        assert len(events) == 1
        assert events[0].event_type == "budget.tripped"
        assert events[0].payload["limit_type"] == "cost"
