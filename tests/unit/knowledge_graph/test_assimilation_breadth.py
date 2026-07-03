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


class _ConceptEngine:
    """Minimal engine: records add_node calls + supports content-hash idempotency."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}

    def add_node(self, node_id, node_type, properties=None):
        self.nodes[node_id] = {"type": node_type, **(properties or {})}

    class _G:
        def __init__(self, outer):
            self._o = outer

        def nodes(self, data=False):
            if data:
                return list(self._o.nodes.items())
            return list(self._o.nodes)

    @property
    def graph(self):
        return _ConceptEngine._G(self)


def test_ingest_concepts_creates_canonical_concept_nodes():
    from agent_utilities.knowledge_graph.assimilation import ingest_concepts

    eng = _ConceptEngine()
    rep = ingest_concepts(
        eng,
        [
            {"id": "KG-2.7", "name": "Research Assimilation", "pillar": "KG-2"},
            {"id": "orch-1.0", "name": "Routing"},  # lowercase normalizes
            {"id": "not-a-concept"},  # gated out (no letters-then-digit)
        ],
    )
    assert rep.ingested == 2
    node = eng.nodes["concept:KG-2.7"]
    assert node["concept_id"] == "KG-2.7" and node["concept_ids"] == ["KG-2.7"]
    assert "concept:ORCH-1.0" in eng.nodes  # upper-normalized
    assert not any("NOT-A-CONCEPT" in k for k in eng.nodes)  # junk skipped
    # idempotent re-run → no new nodes
    rep2 = ingest_concepts(
        eng, [{"id": "KG-2.7", "name": "Research Assimilation", "pillar": "KG-2"}]
    )
    assert rep2.ingested == 0 and rep2.skipped == 1


def test_discover_concepts_reads_registry_then_falls_back_to_markers(tmp_path):
    from agent_utilities.knowledge_graph.assimilation import discover_concepts

    # repo A: ships a concepts.yaml registry (authoritative)
    a = tmp_path / "repo-a" / "docs"
    a.mkdir(parents=True)
    (a / "concepts.yaml").write_text(
        "concepts:\n  - id: KG-2.7\n    name: Research Assimilation\n    pillar: KG-2\n",
        encoding="utf-8",
    )
    # repo B: no registry → CONCEPT: markers in source are scanned
    b = tmp_path / "repo-b"
    b.mkdir()
    (b / "engine.rs").write_text(
        "// CONCEPT:EG-009 native reasoner\n", encoding="utf-8"
    )

    out = {c["id"].upper(): c for c in discover_concepts([str(a.parent), str(b)])}
    assert out["KG-2.7"]["name"] == "Research Assimilation"  # from registry
    assert "EG-009" in out  # from the raw marker fallback


def test_run_breadth_ingest_bridges_concepts(tmp_path):
    from agent_utilities.knowledge_graph.assimilation import run_breadth_ingest

    docs = tmp_path / "repo" / "docs"
    docs.mkdir(parents=True)
    (docs / "concepts.yaml").write_text(
        "concepts:\n  - id: KG-2.7\n    name: Research Assimilation\n", encoding="utf-8"
    )
    captured: list[dict] = []
    report = run_breadth_ingest(
        object(),
        repo_roots=[str(tmp_path / "repo")],
        codebase_ingest=lambda e, m: True,
        concept_ingest=lambda e, cs: (captured.extend(cs), len(cs))[1],
    )
    assert report.concepts >= 1
    assert any(
        c["id"] == "KG-2.7" for c in captured
    )  # breadth drives the concept bridge


def test_workspace_project_roots_returns_existing_local_paths(tmp_path):
    """workspace.yml-defined repos that exist on disk are returned; missing skipped."""
    from agent_utilities.core.workspace_config import workspace_project_roots

    (tmp_path / "agent-utilities").mkdir()  # present
    yml = tmp_path / "workspace.yml"
    yml.write_text(
        f"path: {tmp_path}\n"
        "repositories:\n"
        "  - url: https://x/agent-utilities.git\n"
        "  - url: https://x/not-cloned.git\n",
        encoding="utf-8",
    )
    roots = workspace_project_roots(str(yml))
    assert str(tmp_path / "agent-utilities") in roots
    assert str(tmp_path / "not-cloned") not in roots  # missing dir excluded


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
