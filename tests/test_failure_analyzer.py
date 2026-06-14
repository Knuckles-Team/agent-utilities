"""Failure-driven evolution analyzer (CONCEPT:AHE-3.18).

Covers: deterministic signature clustering, materialization of the dormant
telemetry schema (ExecutionSummary / PerformanceAnomaly) + synthetic
``failure_gap`` Concept topics, and — the contract that makes the whole loop
work — that a materialized gap is selected by the golden loop's
``unresolved_topics`` intake with NO golden-loop change.

@pytest.mark.concept("AHE-3.18")
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.adaptation.failure_analyzer import (
    FailureAnalyzer,
    FailureRecord,
    cluster_failures,
)
from agent_utilities.knowledge_graph.adaptation.topic_resolver import unresolved_topics

pytestmark = pytest.mark.concept("AHE-3.18")


class _FakeEngine:
    """Minimal in-memory engine honoring add_node/link_nodes/query_cypher.

    query_cypher implements just the two shapes ``unresolved_topics`` issues, so
    we can assert the real intake contract against materialized gap nodes.
    """

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(self, node_id, node_type, properties=None):
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def link_nodes(self, source_id, target_id, rel_type, properties=None):
        self.edges.append((source_id, target_id, rel_type.upper()))

    def query_cypher(self, query, params=None):
        if "ADDRESSED_BY" in query:
            addressed = {s for s, _t, r in self.edges if r == "ADDRESSED_BY"}
            # query returns the *concept* side (source) of ADDRESSED_BY
            return [{"id": nid} for nid in addressed]
        # "MATCH (c:Concept) RETURN c.id, c.name"
        return [
            {"id": n["id"], "name": n.get("name")}
            for n in self.nodes.values()
            if n["type"] == "Concept"
        ]


class _FakeBackend:
    """Trace backend returning canned failure telemetry."""

    def __init__(self, errors=None, lows=None, anomalies=None):
        self._errors = errors or []
        self._lows = lows or []
        self._anomalies = anomalies or []

    async def get_error_observations(self, **k):
        return self._errors

    async def get_low_score_traces(self, **k):
        return self._lows

    async def get_cost_latency_anomalies(self, **k):
        return self._anomalies


def _analyzer(engine, backend, *, min_occurrences=2, **kw):
    return FailureAnalyzer(
        engine,
        trace_backend=backend,
        feedback=None,
        latency_budget_ms=1000.0,
        min_occurrences=min_occurrences,
        **kw,
    )


class TestClustering:
    def test_same_failure_different_ids_collapse(self):
        recs = [
            FailureRecord(
                "error",
                "loop",
                "Timeout after 3000ms id=0xAB12CD34",
                "ERROR_RATE",
                "t1",
            ),
            FailureRecord(
                "error",
                "loop",
                "Timeout after 8000ms id=0x99FF00AA",
                "ERROR_RATE",
                "t2",
            ),
            FailureRecord("error", "loop", "Connection refused", "ERROR_RATE", "t3"),
        ]
        pats = cluster_failures(recs)
        assert len(pats) == 2
        assert pats[0].count == 2  # most frequent first
        assert sorted(pats[0].trace_ids) == ["t1", "t2"]


class TestMaterialization:
    def test_materializes_nodes_and_gap_topic(self):
        engine = _FakeEngine()
        backend = _FakeBackend(
            errors=[
                {
                    "traceId": "t1",
                    "name": "loop",
                    "statusMessage": "Timeout after 3000ms",
                },
                {
                    "traceId": "t2",
                    "name": "loop",
                    "statusMessage": "Timeout after 7000ms",
                },
            ],
            anomalies=[
                {
                    "name": "loop",
                    "p95_latency_ms": 9000,
                    "over_latency": True,
                    "over_cost": False,
                },
                {
                    "name": "loop",
                    "p95_latency_ms": 9000,
                    "over_latency": True,
                    "over_cost": False,
                },
            ],
        )
        report = _analyzer(engine, backend).run_once()

        types = [n["type"] for n in engine.nodes.values()]
        assert "PerformanceAnomaly" in types
        assert "ExecutionSummary" in types
        assert any(
            n["type"] == "Concept" and n.get("kind") == "failure_gap"
            for n in engine.nodes.values()
        )
        assert report["gap_concepts"], report
        # ExecutionSummary success_rate < 1.0 so maintainer picks it up
        es = next(n for n in engine.nodes.values() if n["type"] == "ExecutionSummary")
        assert es["success_rate"] < 1.0
        assert es["workflow_id"] == "loop"
        # PerformanceAnomaly carries the fields maintainer queries
        pa = next(n for n in engine.nodes.values() if n["type"] == "PerformanceAnomaly")
        assert pa["anomaly_type"] in {"ERROR_RATE", "TIMEOUT"}
        assert pa["target_node_id"] == "loop"
        # provenance edge wired
        assert any(r == "EVIDENCES" for _s, _t, r in engine.edges)

    def test_one_off_below_min_occurrences_is_ignored(self):
        engine = _FakeEngine()
        backend = _FakeBackend(
            errors=[{"traceId": "t1", "name": "loop", "statusMessage": "rare blip"}]
        )
        report = _analyzer(engine, backend, min_occurrences=2).run_once()
        assert report["gap_concepts"] == []
        assert not any(n["type"] == "Concept" for n in engine.nodes.values())


class TestDaemonRegistration:
    """failure_ingest is a maintenance job gated on KG_FAILURE_EVOLUTION."""

    def test_registered_only_when_enabled(self, monkeypatch):
        from unittest.mock import MagicMock

        from agent_utilities.core import config as cfg_mod
        from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin

        ms = MagicMock()
        monkeypatch.setattr(cfg_mod.config, "kg_failure_evolution", False)
        names = [n for n, _i, _t in TaskManagerMixin._maintenance_jobs(ms)]
        assert "failure_ingest" not in names

        monkeypatch.setattr(cfg_mod.config, "kg_failure_evolution", True)
        monkeypatch.setattr(cfg_mod.config, "kg_failure_evolution_interval", 1234.0)
        jobs = TaskManagerMixin._maintenance_jobs(ms)
        match = [(i, t) for n, i, t in jobs if n == "failure_ingest"]
        assert match and match[0][0] == 1234.0
        assert match[0][1] == ms._tick_failure_ingest


class TestRunFailureIngest:
    """run_failure_ingest (shared by the daemon tick + the MCP action)."""

    def test_run_once_works_inside_running_event_loop(self):
        """The MCP action runs the tick inside the server's event loop; run_once
        must not raise 'asyncio.run() cannot be called from a running event loop'."""
        import asyncio

        async def _in_loop():
            a = FailureAnalyzer(
                _FakeEngine(), trace_backend=_FakeBackend(), feedback=None
            )
            return a.run_once()  # sync call from within a running loop

        rep = asyncio.run(_in_loop())
        assert rep["records_pulled"] == 0

    def test_no_failures_is_clean_noop(self):
        from agent_utilities.knowledge_graph.adaptation.failure_analyzer import (
            run_failure_ingest,
        )

        # No telemetry -> no gaps -> no remediation cycle attempted.
        eng = _FakeEngine()
        import agent_utilities.knowledge_graph.adaptation.failure_analyzer as fa

        orig = fa.FailureAnalyzer.from_engine
        fa.FailureAnalyzer.from_engine = staticmethod(
            lambda engine: fa.FailureAnalyzer(
                engine, trace_backend=_FakeBackend(), feedback=None
            )
        )
        try:
            rep = run_failure_ingest(eng)
        finally:
            fa.FailureAnalyzer.from_engine = orig
        assert rep["gap_concepts"] == []
        assert "remediation" not in rep


class TestLoopTopicsOverride:
    """The failure tick addresses its just-created gaps directly, not via the
    generic (limited, arbitrarily-ordered) unresolved_topics scan."""

    def test_supplied_topics_bypass_generic_scan(self, monkeypatch):
        import agent_utilities.knowledge_graph.enrichment.cards as cards_mod
        import agent_utilities.knowledge_graph.enrichment.synthesize as synth_mod
        import agent_utilities.knowledge_graph.research.loop_controller as gl
        from agent_utilities.knowledge_graph.enrichment.orchestration import TeamSpec
        from agent_utilities.knowledge_graph.research.loop_controller import (
            LoopController,
        )

        captured = {}

        def fake_synth(goal, *a, **k):
            captured["goal"] = goal
            return TeamSpec(name="T", goal=goal, lead="L", members=["a", "b"]), []

        monkeypatch.setattr(synth_mod, "synthesize_team", fake_synth)
        monkeypatch.setattr(synth_mod, "persist_synthesis", lambda *a, **k: (1, 0))
        monkeypatch.setattr(cards_mod, "make_lite_llm_fn", lambda *a, **k: (lambda p: "{}"))

        def _boom(*a, **k):  # must never run when topics are supplied
            raise AssertionError("generic active_loops scan should be skipped")

        monkeypatch.setattr(gl, "active_loops", _boom)

        from unittest.mock import MagicMock

        eng = MagicMock()
        eng.backend.semantic_search = lambda *a, **k: []
        ctrl = LoopController(eng)
        ctrl._capability_search = lambda: (lambda q, top_k=5: [])  # type: ignore[assignment]

        rep = ctrl.run_one_cycle(
            topics=[{"id": "failure_gap:x", "name": "Failure: timeout in loop"}],
            assimilate=False,
        )
        assert rep["topics_intake"] == 1
        assert "timeout in loop" in captured["goal"]


class TestIntakeContract:
    def test_gap_is_selected_by_unresolved_topics(self):
        """The materialized failure_gap must ride the golden loop's intake unchanged."""
        engine = _FakeEngine()
        # Same normalized signature (differ only in a number) so they cluster.
        backend = _FakeBackend(
            errors=[
                {"traceId": "t1", "name": "loop", "statusMessage": "boom error code 1"},
                {"traceId": "t2", "name": "loop", "statusMessage": "boom error code 2"},
            ]
        )
        _analyzer(engine, backend).run_once()
        topics = unresolved_topics(engine, limit=10)
        gap_ids = [t["id"] for t in topics]
        assert any(g.startswith("failure_gap:") for g in gap_ids), gap_ids
