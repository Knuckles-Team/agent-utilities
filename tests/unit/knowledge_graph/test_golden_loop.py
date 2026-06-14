"""Unit tests for the propose-only self-evolution golden loop (CONCEPT:KG-2.7)."""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.adaptation.topic_resolver import (
    mark_addressed,
    unresolved_topics,
)
from agent_utilities.knowledge_graph.research.golden_loop import GoldenLoopController


class _StubEngine:
    """Minimal engine: canned cypher results + records link_nodes calls."""

    def __init__(self, concepts, addressed):
        self._concepts = concepts  # list[(id, name)]
        self._addressed = set(addressed)  # ids with ADDRESSED_BY
        self.links: list[tuple[str, str, str]] = []
        self.backend = object()  # no semantic_search → acquire returns []

    def query_cypher(self, q: str, params: dict | None = None) -> list[dict[str, Any]]:
        if "ADDRESSED_BY" in q and "RETURN c.id AS id" in q and "name" not in q:
            return [{"id": i} for i in self._addressed]
        if "MATCH (c:Concept) RETURN c.id AS id, c.name AS name" in q:
            return [{"id": i, "name": n} for i, n in self._concepts]
        return []

    def link_nodes(self, source_id, target_id, rel_type, properties=None):
        self.links.append((source_id, target_id, rel_type))


def test_unresolved_topics_subtracts_addressed():
    eng = _StubEngine(
        concepts=[("c:1", "A"), ("c:2", "B"), ("c:3", "C")],
        addressed=["c:2"],
    )
    topics = unresolved_topics(eng, limit=10)
    ids = {t["id"] for t in topics}
    assert ids == {"c:1", "c:3"}  # c:2 is already addressed → excluded


def test_mark_addressed_writes_both_directions():
    eng = _StubEngine([], [])
    n = mark_addressed(eng, "c:1", ["src:a", "src:b", "c:1"], source="t")
    assert n == 2  # self-link (c:1) skipped
    rels = {(s, t, r) for s, t, r in eng.links}
    assert ("src:a", "c:1", "ADDRESSES") in rels
    assert ("c:1", "src:a", "ADDRESSED_BY") in rels


def test_run_breadth_self_configures_from_workspace_yml(monkeypatch):
    """Live path: with no KG_BREADTH_* roots, breadth auto-discovers the ecosystem
    from the XDG workspace.yml — so assimilate always has a codebase to compare
    research against, zero-config (CONCEPT:KG-2.7)."""
    import agent_utilities.core.workspace_config as wc
    import agent_utilities.knowledge_graph.assimilation as assim
    from agent_utilities.knowledge_graph.assimilation.breadth_ingest import (
        BreadthReport,
    )

    monkeypatch.delenv("KG_BREADTH_LIBRARY_ROOTS", raising=False)
    monkeypatch.delenv("KG_BREADTH_REPO_ROOTS", raising=False)
    monkeypatch.setattr(
        wc, "workspace_project_roots", lambda *a, **k: ["/eco/repo-a", "/eco/repo-b"]
    )
    captured: dict = {}

    def fake_run(engine, *, library_roots=None, repo_roots=None, **kw):
        captured["repos"] = repo_roots
        return BreadthReport()

    monkeypatch.setattr(assim, "run_breadth_ingest", fake_run)

    rep = GoldenLoopController(_StubEngine([], []))._run_breadth()
    assert captured["repos"] == ["/eco/repo-a", "/eco/repo-b"]
    assert not rep.get("skipped")


def test_run_one_cycle_intake_only_propose_only():
    eng = _StubEngine(concepts=[("c:1", "A"), ("c:2", "B")], addressed=[])
    # acquire returns [] (no semantic_search) → resolve does nothing, but the
    # cycle must complete cleanly and stay propose-only.
    rep = GoldenLoopController(eng).run_one_cycle(synthesize=False, distill=False)
    assert rep["propose_only"] is True
    assert rep["topics_intake"] == 2
    assert rep["topics_resolved"] == 0
    assert rep["errors"] == []


def test_run_one_cycle_intake_papers_runs_research_pipeline(monkeypatch):
    """Caller-supplied papers trigger the unified intake stage (research-pipeline
    runner) before assimilate (CONCEPT:KG-2.77)."""
    from types import SimpleNamespace

    import agent_utilities.automation.research_pipeline as rp

    seen: dict = {}

    class _FakeRunner:
        def __init__(self, engine=None, **kw):
            seen["engine"] = engine

        async def run_daily_pipeline(self, papers=None):
            seen["papers"] = papers
            return SimpleNamespace(
                papers_discovered=len(papers or []),
                papers_relevant=1,
                papers_marginal=0,
                papers_already_known=0,
                owl_inferences=0,
                errors=[],
            )

    monkeypatch.setattr(rp, "ResearchPipelineRunner", _FakeRunner)

    eng = _StubEngine([], [])
    rep = GoldenLoopController(eng).run_one_cycle(
        papers=[{"id": "2606.09498", "title": "Self-Harness"}],
        assimilate=False,
        synthesize=False,
        breadth=False,
    )
    assert rep["intake_papers"]["papers_discovered"] == 1
    assert seen["papers"][0]["id"] == "2606.09498"
    assert rep["errors"] == []
