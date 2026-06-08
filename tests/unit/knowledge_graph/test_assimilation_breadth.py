#!/usr/bin/python
"""Breadth-ingest orchestration + pilot harness (VU-10).

CONCEPT:KG-2.7
"""

import pytest

from agent_utilities.knowledge_graph.assimilation import (
    classify_project,
    discover_projects,
    organize_libraries,
    run_breadth_ingest,
    run_pilot,
)

pytestmark = pytest.mark.concept("KG-2.7")


# --- classification / discovery (real temp tree) ----------------------------
def _mk_project(root, rel, marker, files=()):
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / marker).write_text("x", encoding="utf-8")
    for f in files:
        (d / f).write_text("x", encoding="utf-8")
    return d


def test_classify_project_language_and_pillars(tmp_path):
    d = _mk_project(tmp_path, "memory-rag-kg/reasoning-bank", "pyproject.toml")
    m = classify_project(d)
    assert m.language == "python"
    assert "KG" in m.pillars  # memory/rag/kg keywords → KG pillar


def test_discover_projects_finds_roots_and_skips_nested(tmp_path):
    _mk_project(tmp_path, "agent-frameworks/llm-council", "package.json")
    _mk_project(tmp_path, "quant-trading/crypto-trader", "Cargo.toml")
    _mk_project(tmp_path, "agent-frameworks/llm-council/vendor/dep", "package.json")
    projects = discover_projects(tmp_path)
    names = {m.name for m in projects}
    # the two real roots are found; the nested vendored dep under a project is not descended
    assert "llm-council" in names and "crypto-trader" in names
    assert "dep" not in names


def test_organize_writes_manifests(tmp_path):
    _mk_project(tmp_path, "memory-os", "pyproject.toml")
    manifests = organize_libraries(tmp_path)
    assert len(manifests) == 1
    assert (tmp_path / "memory-os" / "manifest.json").exists()


# --- orchestration with injected ingest -------------------------------------
def test_run_breadth_ingest_with_fakes(tmp_path):
    _mk_project(tmp_path, "libs/memory-os", "pyproject.toml")
    _mk_project(tmp_path, "libs/quant-trading", "Cargo.toml")
    ingested: list[str] = []

    def fake_cb(engine, manifest):
        ingested.append(manifest.name)
        return True

    def fake_doc(engine, docs):
        return len(docs)

    report = run_breadth_ingest(
        object(),
        library_roots=[str(tmp_path / "libs")],
        docs=[{"uri": "/prd/x.md", "text": "req"}],
        codebase_ingest=fake_cb,
        doc_ingest=fake_doc,
    )
    assert report.projects == 2
    assert report.codebases_ingested == 2
    assert set(ingested) == {"memory-os", "quant-trading"}
    assert report.docs_ingested == 1


# --- pilot harness ----------------------------------------------------------
class _Graph:
    def __init__(self, nodes):
        self._n = dict(nodes)
        self._out: dict = {}
        self._in: dict = {}

    def nodes(self, data=False):
        return list(self._n.items()) if data else list(self._n)

    def add_node(self, nid, attrs):
        self._n[nid] = attrs

    def add_edge(self, s, d, p):
        self._out.setdefault(s, []).append((s, d, p))
        self._in.setdefault(d, []).append((s, d, p))

    def out_edges(self, nid, data=False):
        e = self._out.get(nid, [])
        return e if data else [(s, t) for s, t, _ in e]

    def in_edges(self, nid, data=False):
        e = self._in.get(nid, [])
        return e if data else [(s, t) for s, t, _ in e]


class _Engine:
    def __init__(self, nodes):
        self.graph = _Graph(nodes)

    def add_node(self, nid, nt, properties=None, ephemeral=False):
        self.graph.add_node(nid, {**(properties or {}), "type": nt})

    def link_nodes(self, s, d, rel, properties=None, ephemeral=False):
        self.graph.add_edge(s, d, properties or {})


def test_pilot_passes_when_built_features_not_reproposed():
    engine = _Engine(
        {
            # an OPEN gap the engine should propose
            "open1": {
                "type": "capability",
                "name": "new cap",
                "concept_ids": ["KG-2.1"],
                "research_sources": ["p1"],
                "status": "open",
            },
            # an ALREADY-IMPLEMENTED feature that must NOT be re-proposed
            "built1": {
                "type": "sdd_feature",
                "name": "done cap",
                "status": "implemented",
            },
        }
    )
    rep = run_pilot(engine, top_n=10)
    assert rep.already_built == 1
    assert "open1" in {g["feature_id"] for g in rep.ranked_gaps}
    assert rep.reproposed_built == []  # the invariant
    assert rep.passed is True
