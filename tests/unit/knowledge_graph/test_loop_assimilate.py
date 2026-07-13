#!/usr/bin/python
"""Golden-loop assimilation stage + watermark idempotency (VU-7).

CONCEPT:AU-KG.query.vendor-agnostic-traversal
"""

import pytest

from agent_utilities.knowledge_graph.research.loop_controller import LoopController

pytestmark = pytest.mark.concept("AU-KG.query.vendor-agnostic-traversal")


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

    def query_cypher(self, cypher, params=None):
        # Answer the two BOUNDED watermark-stage query shapes for real (matching a
        # live engine); everything else (e.g. topic-detection queries) resolves to
        # [] -> "no unresolved topics", preserving cycle-stops-after-assimilate.
        params = params or {}
        if "RETURN count(n)" in cypher:
            types = {
                t.strip().strip("'").lower()
                for t in cypher.split("[", 1)[1].split("]", 1)[0].split(",")
            }
            n = sum(
                1
                for attrs in self.graph._n.values()
                if str(attrs.get("type", "")).lower() in types
            )
            return [{"c": n}]
        if "n.id = $id" in cypher and "n.hash" in cypher:
            attrs = self.graph._n.get(params.get("id"))
            if attrs is None or "hash" not in attrs:
                return []
            return [{"hash": attrs["hash"]}]
        return []  # no unresolved topics → cycle stops after assimilate

    def submit_task(self, *, target_path, **k):
        self.submitted.append(target_path)
        return f"job:{len(self.submitted)}"


def _graph_nodes():
    return {
        "f1": {
            "type": "capability",
            "embedding": [1.0, 0.0],
            "concept_ids": ["AU-KG.memory.tiered-memory-caching"],
            "research_sources": ["p1", "p2"],
            "importance_score": 0.5,
            "status": "open",
        },
        "f2": {
            "type": "capability",
            "embedding": [1.0, 0.0],
            "concept_ids": ["AU-KG.memory.tiered-memory-caching"],
            "importance_score": 0.4,
            "status": "open",
        },  # duplicate of f1
        "f3": {
            "type": "capability",
            "embedding": [0.0, 1.0],
            "concept_ids": ["AU-ORCH.adapter.hot-cache-invalidation"],
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


class _TooLargeError(Exception):
    """Mirrors the live engine's ``RESULT_TOO_LARGE`` response guard."""


class _GuardedGraph(_Graph):
    """A graph whose unscoped ``nodes(data=True)`` behaves like the live
    ``epistemic-graph`` engine at ecosystem scale: refused outright once the
    graph outgrows the response cap (live-reproduced at 139,657 nodes > the
    50,000 cap, ``RESULT_TOO_LARGE: GetNodes would return ... nodes``).
    ``get_nodes_by_label`` — the bounded per-label surface — stays available,
    since that's what the assimilate path must use instead.
    """

    def nodes(self, data=False):
        raise _TooLargeError(
            "RESULT_TOO_LARGE: GetNodes would return 139657 nodes (> cap 50000)"
        )

    def get_nodes_by_label(self, label, limit=0):
        wanted = label.lower()
        rows = [
            (nid, attrs)
            for nid, attrs in self._n.items()
            if str(attrs.get("type", "")).lower() == wanted
        ]
        return rows if not limit else rows[:limit]


class _GuardedEngine(_Engine):
    """An engine backed by ``_GuardedGraph`` — inherits ``_Engine.query_cypher``'s
    real answers for the two BOUNDED watermark-stage shapes, but any accidental
    unscoped ``graph.nodes()`` call (the bug this regression test guards against)
    raises ``_TooLargeError`` instead of silently succeeding.
    """

    def __init__(self, nodes):
        super().__init__(nodes)
        self.graph = _GuardedGraph(nodes)


def test_assimilate_watermark_stages_never_dump_whole_graph():
    """Bug fix regression (RESULT_TOO_LARGE): ``_state_watermark``/``_load_watermark``
    run on every ``assimilate`` call, before dedup/gap/synergy/rank even start, so
    they must use bounded per-id / per-label queries — never an unscoped
    ``graph.nodes(data=True)`` whole-graph dump. ``_GuardedGraph.nodes()`` raises if
    that ever happens; a passing run proves the bounded path is what's really used.
    """
    engine = _GuardedEngine(_graph_nodes())
    ctl = LoopController(engine)
    first = ctl._run_assimilate()
    assert first["skipped"] is False
    assert first["watermark"]
    # Second call re-reads the persisted watermark via the same bounded id lookup
    # (not a dump) and correctly detects "unchanged".
    second = ctl._run_assimilate()
    assert second["skipped"] is True and second["reason"] == "unchanged"


def test_state_watermark_fallback_is_bounded_per_label():
    """When the cheap Cypher count is unavailable, the hash-based fallback must
    still use the bounded ``get_nodes_by_label`` surface (via ``iter_typed_nodes``),
    not an unscoped ``graph.nodes(data=True)`` scan.
    """

    class _NoCountEngine(_GuardedEngine):
        def query_cypher(self, cypher, params=None):
            if "RETURN count(n)" in cypher:
                return []  # forces _cheap_input_count() -> None -> fallback path
            return super().query_cypher(cypher, params)

    engine = _NoCountEngine(_graph_nodes())
    ctl = LoopController(engine)
    wm = ctl._state_watermark()
    assert wm  # a real hash, computed without touching graph.nodes()


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
