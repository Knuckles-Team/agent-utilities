"""PerformanceAnomaly consumer (CONCEPT:AU-AHE.optimization.performance-anomaly-consumer).

PerformanceAnomaly nodes were written (kg-report-persister skill,
ExecutionSummary flows, the failure analyzer) but had no consumer. The
consumer pass scans unconsumed anomalies, clusters by (target, type), files
failure_gap topics via the failure analyzer's shared gap-topic path, and
stamps them consumed; a daemon tick runs it (KG_ANOMALY_CONSUMER, default ON).

@pytest.mark.concept("AU-AHE.optimization.performance-anomaly-consumer")
"""

from __future__ import annotations

import re

import pytest

from agent_utilities.knowledge_graph.adaptation.anomaly_consumer import (
    consume_anomalies,
)

pytestmark = pytest.mark.concept("AU-AHE.optimization.performance-anomaly-consumer")


class _Backend:
    def __init__(self, engine):
        self._engine = engine

    def execute(self, query, params=None):
        params = params or {}
        if "SET a.consumed" in query:
            node = self._engine.nodes.get(params.get("id"))
            if node is not None:
                node["consumed"] = params.get("ts")
                node["consumed_by"] = "anomaly_consumer"
        return []


class _Engine:
    """Fake engine honoring the consumer's query/write surface."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str]] = []
        self.backend = _Backend(self)

    def add_node(self, node_id, node_type, properties=None):
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def link_nodes(self, source_id, target_id, rel_type, properties=None):
        self.edges.append((source_id, target_id, rel_type.upper()))

    def query_cypher(self, query, params=None):
        if "a.consumed IS NULL" in query:
            m = re.search(r"LIMIT (\d+)", query)
            limit = int(m.group(1)) if m else 1000
            rows = [
                {"a": dict(n)}
                for n in self.nodes.values()
                if n["type"] == "PerformanceAnomaly" and "consumed" not in n
            ]
            return rows[:limit]
        if "EVIDENCES" in query:
            return [
                {"id": s}
                for s, _t, r in self.edges
                if r == "EVIDENCES"
                and s in self.nodes
                and self.nodes[s]["type"] == "PerformanceAnomaly"
            ]
        return []


def _seed_anomaly(engine, aid, target="wf_alpha", anomaly_type="TIMEOUT"):
    engine.add_node(
        aid,
        "PerformanceAnomaly",
        properties={
            "target_node_id": target,
            "anomaly_type": anomaly_type,
            "threshold_exceeded": 9000.0,
            "baseline": 1000.0,
            "metadata": "p95 latency exceeds budget",
            "timestamp": "2026-06-10T00:00:00Z",
        },
    )
    return aid


def _gaps(engine):
    return [
        n
        for n in engine.nodes.values()
        if n["type"] == "Concept" and n.get("kind") == "failure_gap"
    ]


class TestConsumeAnomalies:
    def test_files_gap_and_marks_consumed(self):
        engine = _Engine()
        _seed_anomaly(engine, "pa:1")
        report = consume_anomalies(engine)
        assert report["scanned"] == 1
        assert report["gaps_filed"] == 1
        assert report["consumed"] == 1
        (gap,) = _gaps(engine)
        assert gap["source"] == "anomaly_consumer"
        assert ("pa:1", gap["id"], "EVIDENCES") in engine.edges
        assert engine.nodes["pa:1"]["consumed_by"] == "anomaly_consumer"

    def test_clusters_same_target_and_type_into_one_gap(self):
        engine = _Engine()
        _seed_anomaly(engine, "pa:1")
        _seed_anomaly(engine, "pa:2")
        _seed_anomaly(engine, "pa:3", target="wf_beta", anomaly_type="ERROR_RATE")
        report = consume_anomalies(engine)
        assert report["gaps_filed"] == 2  # one per (target, type) cluster
        gaps = _gaps(engine)
        cluster_gap = next(g for g in gaps if "wf_alpha" in g["name"])
        assert cluster_gap["occurrences"] == 2
        # every clustered anomaly evidences the shared gap
        assert ("pa:1", cluster_gap["id"], "EVIDENCES") in engine.edges
        assert ("pa:2", cluster_gap["id"], "EVIDENCES") in engine.edges

    def test_second_pass_is_idempotent(self):
        engine = _Engine()
        _seed_anomaly(engine, "pa:1")
        consume_anomalies(engine)
        second = consume_anomalies(engine)
        assert second["scanned"] == 0
        assert len(_gaps(engine)) == 1

    def test_already_evidencing_anomalies_skip_refiling(self):
        engine = _Engine()
        # A failure_analyzer-born anomaly already EVIDENCES its gap.
        _seed_anomaly(engine, "pa:fa")
        engine.add_node("failure_gap:x", "Concept", properties={"kind": "failure_gap"})
        engine.link_nodes("pa:fa", "failure_gap:x", "EVIDENCES")
        before = len(_gaps(engine))
        report = consume_anomalies(engine)
        assert report["already_evidenced"] == 1
        assert report["gaps_filed"] == 0
        assert len(_gaps(engine)) == before
        # ...but it IS marked consumed so it never rescans.
        assert "consumed" in engine.nodes["pa:fa"]

    def test_scan_limit_bounds_the_pass(self):
        engine = _Engine()
        for i in range(5):
            _seed_anomaly(engine, f"pa:{i}", target=f"wf_{i}")
        report = consume_anomalies(engine, limit=2)
        assert report["scanned"] == 2

    def test_gap_topic_reaches_loop_intake(self):
        from agent_utilities.knowledge_graph.adaptation.topic_resolver import (
            unresolved_topics,
        )

        class _IntakeEngine(_Engine):
            def query_cypher(self, query, params=None):
                if "ADDRESSED_BY" in query:
                    return []
                if "Concept" in query and "RETURN" in query and "c." in query:
                    return [
                        {"id": n["id"], "name": n.get("name")}
                        for n in self.nodes.values()
                        if n["type"] == "Concept"
                    ]
                return super().query_cypher(query, params)

        engine = _IntakeEngine()
        _seed_anomaly(engine, "pa:1")
        consume_anomalies(engine)
        topics = unresolved_topics(engine)
        assert any(t["id"].startswith("failure_gap:") for t in topics)


class TestDaemonRegistration:
    def _jobs(self, monkeypatch, enabled: bool):
        from agent_utilities.core import schedule_engine as _se
        from agent_utilities.core.config import config as cfg
        from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
            EpistemicGraphBackend,
        )
        from agent_utilities.knowledge_graph.core.engine_tasks import (
            TaskManagerMixin,
        )

        monkeypatch.setattr(cfg, "kg_anomaly_consumer", enabled)
        inst = TaskManagerMixin.__new__(TaskManagerMixin)  # type: ignore[type-abstract]
        inst.backend = EpistemicGraphBackend()
        inst._register_maintenance_schedules()
        return {s.name for s in _se._load_all(inst) if s.enabled}

    def test_tick_registered_by_default_flag(self, monkeypatch):
        assert "anomaly_consumer" in self._jobs(monkeypatch, True)

    def test_flag_off_unregisters_tick(self, monkeypatch):
        assert "anomaly_consumer" not in self._jobs(monkeypatch, False)

    def test_default_flag_is_on(self):
        from agent_utilities.core.config import AgentConfig

        assert AgentConfig().kg_anomaly_consumer is True
