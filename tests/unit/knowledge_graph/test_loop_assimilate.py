#!/usr/bin/python
"""Golden-loop assimilation stage + watermark idempotency (VU-7).

CONCEPT:KG-2.7
"""

import pytest

from agent_utilities.knowledge_graph.research.loop_controller import LoopController

pytestmark = pytest.mark.concept("KG-2.7")


class _Graph:
    def __init__(self, nodes):
        self._n = dict(nodes)
        self._out: dict = {}
        self._in: dict = {}

    def nodes(self, data=False):
        return list(self._n.items()) if data else list(self._n)

    def add_node(self, node_id, attrs):
        self._n[node_id] = attrs

    def add_edge(self, src, dst, props):
        self._out.setdefault(src, []).append((src, dst, props))
        self._in.setdefault(dst, []).append((src, dst, props))

    def out_edges(self, nid, data=False):
        e = self._out.get(nid, [])
        return e if data else [(s, t) for s, t, _ in e]

    def in_edges(self, nid, data=False):
        e = self._in.get(nid, [])
        return e if data else [(s, t) for s, t, _ in e]


class _Engine:
    def __init__(self, nodes):
        self.graph = _Graph(nodes)
        self.backend = None
        self.submitted: list[str] = []

    def add_node(self, node_id, node_type, properties=None, ephemeral=False):
        self.graph.add_node(node_id, {**(properties or {}), "type": node_type})

    def link_nodes(self, src, dst, rel_type, properties=None, ephemeral=False):
        self.graph.add_edge(src, dst, properties or {})

    def query_cypher(self, *a, **k):
        return []  # no unresolved topics → cycle stops after assimilate

    def submit_task(self, *, target_path, **k):
        self.submitted.append(target_path)
        return f"job:{len(self.submitted)}"


def _graph_nodes():
    return {
        "f1": {
            "type": "capability",
            "embedding": [1.0, 0.0],
            "concept_ids": ["KG-2.1"],
            "research_sources": ["p1", "p2"],
            "importance_score": 0.5,
            "status": "open",
        },
        "f2": {
            "type": "capability",
            "embedding": [1.0, 0.0],
            "concept_ids": ["KG-2.1"],
            "importance_score": 0.4,
            "status": "open",
        },  # duplicate of f1
        "f3": {
            "type": "capability",
            "embedding": [0.0, 1.0],
            "concept_ids": ["ORCH-1.2"],
            "research_sources": ["p3"],
            "importance_score": 0.7,
            "status": "open",
        },
        "c1": {"type": "concept", "embedding": [0.2, 0.2]},
    }


def test_run_assimilate_dedups_and_ranks():
    ctl = LoopController(_Engine(_graph_nodes()))
    rep = ctl._run_assimilate()
    assert rep["skipped"] is False
    assert rep["duplicates_superseded"] >= 1  # f1 supersedes f2
    assert rep["open_gaps"] == 2  # f1, f3 (f2 superseded)
    assert {g["feature_id"] for g in rep["ranked_gaps"]} == {"f1", "f3"}
    assert rep["watermark"]


def test_run_assimilate_idempotent_skip():
    engine = _Engine(_graph_nodes())
    ctl = LoopController(engine)
    first = ctl._run_assimilate()
    assert first["skipped"] is False
    second = ctl._run_assimilate()  # nothing changed
    assert second["skipped"] is True and second["reason"] == "unchanged"


def test_force_overrides_watermark_skip():
    engine = _Engine(_graph_nodes())
    ctl = LoopController(engine)
    ctl._run_assimilate()
    forced = ctl._run_assimilate(force=True)
    assert forced["skipped"] is False


def test_run_one_cycle_includes_assimilate():
    ctl = LoopController(_Engine(_graph_nodes()))
    report = ctl.run_one_cycle()
    assert report["assimilate"] is not None
    assert report["assimilate"]["skipped"] is False
    assert report["topics_intake"] == 0  # no topics → stops after assimilate


def test_cycle_metrics_and_evolution_node_persisted():
    engine = _Engine(_graph_nodes())
    report = LoopController(engine).run_one_cycle()
    # monitoring: per-stage timings + error count + duration
    m = report["metrics"]
    assert "duration_ms" in m and m["error_count"] == 0
    assert "assimilate" in m["stage_ms"] and "intake" in m["stage_ms"]
    # a queryable EvolutionCycle node was persisted under the shared id/type
    # convention used by the daemon tick (engine_tasks._tick_evolution)
    cycles = [nid for nid in engine.graph.nodes() if str(nid).startswith("evo_cycle_")]
    assert len(cycles) == 1


def test_breadth_stage_runs_when_configured(tmp_path, monkeypatch):
    (tmp_path / "memory-os").mkdir()
    (tmp_path / "memory-os" / "pyproject.toml").write_text("x", encoding="utf-8")
    monkeypatch.setenv("KG_BREADTH_LIBRARY_ROOTS", str(tmp_path))
    # Isolate from any ambient/config-configured roots so only the test's tmp lib
    # is scanned. _run_breadth reads a FRESH AgentConfig() (KG-2.7) whose
    # settings use env_ignore_empty=True AND fall back to the repo's .env FILE —
    # so neither delenv nor an empty env var isolates on a deployed checkout
    # with a real .env. Point the roots at an existing-but-empty directory.
    empty_repos = tmp_path / "no-repos"
    empty_repos.mkdir()
    monkeypatch.setenv("KG_BREADTH_REPO_ROOTS", str(empty_repos))
    from agent_utilities.core.config import config as _cfg

    monkeypatch.setattr(_cfg, "kg_breadth_repo_roots", "", raising=False)
    monkeypatch.setattr(_cfg, "kg_breadth_library_roots", "", raising=False)
    engine = _Engine(_graph_nodes())
    report = LoopController(engine).run_one_cycle(breadth=True)
    assert report["breadth"] is not None and report["breadth"]["projects"] == 1
    assert engine.submitted == [
        str(tmp_path / "memory-os")
    ]  # codebase ingest submitted
    assert "breadth" in report["metrics"]["stage_ms"]
