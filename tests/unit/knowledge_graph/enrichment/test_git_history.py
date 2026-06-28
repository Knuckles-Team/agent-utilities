"""Git commit-history → KG ingestion (CONCEPT:KG-2.282 / KG-2.283)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.enrichment.git_history import (
    aggregate_churn,
    existing_commit_shas,
    extract_commits,
    ingest_commit_history,
    query_evolution,
)


class _FakeBackend:
    """In-memory backend that records add_node/add_edge (no engine needed)."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, dict]] = []
        self._graph = None  # forces _BatchedBackend per-item fallback

    def add_node(self, node_id: str, label: str = "", **props) -> None:
        self.nodes[node_id] = {"label": label, **props}

    def add_edge(self, source: str, target: str, rel_type: str = "", **props) -> None:
        self.edges.append((source, target, {"rel_type": rel_type, **props}))

    def nodes_by_label(self, label: str, limit: int = 0):
        return [
            (nid, p)
            for nid, p in self.nodes.items()
            if p.get("type") == label or p.get("label") == label
        ]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "demo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "dev@example.com")
    _git(repo, "config", "user.name", "Dev One")
    # a.py & b.py co-change 3x (coupling), c.py changes alone.
    for i in range(3):
        (repo / "a.py").write_text(f"a = {i}\n")
        (repo / "b.py").write_text(f"b = {i}\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", f"touch a+b {i}")
    (repo / "c.py").write_text("c = 0\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add c")
    return repo


def test_extract_commits_one_pass(tmp_path):
    repo = _make_repo(tmp_path)
    res = extract_commits(str(repo))
    assert len(res.commits) == 4
    assert not res.capped
    newest = res.commits[0]  # git log is newest-first
    assert newest.subject == "add c"
    assert newest.author_email == "dev@example.com"
    assert [f.path for f in newest.files] == ["c.py"]
    assert newest.files[0].insertions == 1
    # The 4 commits form a linear DAG: every non-root has exactly one parent.
    assert sum(1 for c in res.commits if c.parents) == 3


def test_extract_handles_renames(tmp_path):
    repo = _make_repo(tmp_path)
    _git(repo, "mv", "c.py", "d.py")
    _git(repo, "commit", "-q", "-m", "rename c->d")
    res = extract_commits(str(repo))
    head = res.commits[0]
    fc = head.files[0]
    assert fc.path == "d.py"
    assert fc.renamed_from == "c.py"


def test_aggregate_churn(tmp_path):
    repo = _make_repo(tmp_path)
    churn = aggregate_churn(extract_commits(str(repo)).commits)
    assert churn["a.py"]["commits"] == 3
    assert churn["c.py"]["commits"] == 1
    assert churn["a.py"]["authors"] == {"dev@example.com"}


def test_ingest_builds_graph_and_coupling(tmp_path):
    repo = _make_repo(tmp_path)
    be = _FakeBackend()
    out = ingest_commit_history(be, str(repo), min_support=3)

    assert out["commits"] == 4 and out["no_op"] is False
    assert out["commits_per_sec"] > 0

    # :Commit / :Author / :File nodes exist.
    commits = [n for n in be.nodes.values() if n.get("type") == "Commit"]
    authors = [n for n in be.nodes.values() if n.get("type") == "Author"]
    files = {n["path"] for n in be.nodes.values() if n.get("type") == "File"}
    assert len(commits) == 4
    assert len(authors) == 1 and authors[0]["email"] == "dev@example.com"
    assert {"a.py", "b.py", "c.py"} <= files

    # Hotspot churn props on the File node.
    a_node = next(n for n in be.nodes.values() if n.get("path") == "a.py")
    assert a_node["commit_count"] == 3 and a_node["churn"] >= 3

    # Edge types: AUTHORED, PARENT (DAG), TOUCHED, and derived FILE_CHANGES_WITH.
    rels = {e[2]["rel_type"] for e in be.edges}
    assert {"AUTHORED", "PARENT", "TOUCHED", "FILE_CHANGES_WITH"} <= rels
    coupling = [e for e in be.edges if e[2]["rel_type"] == "FILE_CHANGES_WITH"]
    coupled_pairs = {(e[0], e[1]) for e in coupling}
    assert ("file:a.py", "file:b.py") in coupled_pairs  # a & b co-change 3x
    parents = [e for e in be.edges if e[2]["rel_type"] == "PARENT"]
    assert len(parents) == 3  # linear history


def test_reingest_is_a_noop(tmp_path):
    repo = _make_repo(tmp_path)
    be = _FakeBackend()
    ingest_commit_history(be, str(repo), min_support=3)
    n_nodes, n_edges = len(be.nodes), len(be.edges)

    # Re-ingest with the shas already present → delta says nothing new.
    shas = existing_commit_shas(be, "demo")
    assert len(shas) == 4
    out2 = ingest_commit_history(be, str(repo), existing_shas=shas, min_support=3)
    assert out2["no_op"] is True and out2["commits"] == 0
    assert out2["skipped"] == 4
    # No new writes.
    assert len(be.nodes) == n_nodes and len(be.edges) == n_edges


def test_incremental_ingest_adds_only_new(tmp_path):
    repo = _make_repo(tmp_path)
    be = _FakeBackend()
    ingest_commit_history(be, str(repo), min_support=3)
    shas = existing_commit_shas(be, "demo")

    # New commit after the watermark.
    (repo / "e.py").write_text("e = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add e")

    out = ingest_commit_history(be, str(repo), existing_shas=shas, min_support=3)
    assert out["commits"] == 1 and out["no_op"] is False
    assert "file:e.py" in be.nodes


def test_query_evolution_hotspots_and_owners(tmp_path):
    repo = _make_repo(tmp_path)
    be = _FakeBackend()
    ingest_commit_history(be, str(repo), min_support=3)

    hot = query_evolution(be, "hotspots", limit=5)
    assert hot["mode"] == "hotspots"
    top = hot["hotspots"][0]
    # a.py / b.py churn 3x → ahead of c.py (1x).
    assert top["file"] in {"a.py", "b.py"} and top["commits"] == 3


def test_extract_non_git_path_is_empty(tmp_path):
    res = extract_commits(str(tmp_path / "not-a-repo"))
    assert res.commits == [] and res.capped is False


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
