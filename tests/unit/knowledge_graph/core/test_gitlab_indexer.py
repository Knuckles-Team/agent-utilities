"""Tests for whole-instance GitLab code indexing into the KG (CONCEPT:KG-2.9g).

Validates the dependency-injected orchestration (project/file filtering, delta
watermark skipping, id scoping, namespacing, IndexResult→entities/relationships
mapping, and source-sync routing) without a live GitLab or engine.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.gitlab_indexer import (
    GitLabProject,
    index_instance,
    map_index_result,
)
from agent_utilities.knowledge_graph.core.source_sync import sync_source

# A resolved IndexResult as the engine's `index_repository` returns it: util.shared
# defined, app.run calls it (resolved), app imports util (resolved depends_on).
INDEX_RESULT = {
    "symbols_extracted": 2,
    "files_parsed": 2,
    "calls_resolved": 1,
    "imports_resolved": 1,
    "nodes": [
        {
            "node_id": "symbol:aaa",
            "node_type": "SYMBOL",
            "properties": {
                "name": "shared",
                "symbol_type": "Function",
                "file_path": "pkg/util.py",
                "language": "python",
            },
        },
        {
            "node_id": "symbol:bbb",
            "node_type": "SYMBOL",
            "properties": {
                "name": "run",
                "symbol_type": "Function",
                "file_path": "pkg/app.py",
                "language": "python",
                "calls": "shared",
            },
        },
    ],
    "edges": [
        {"source": "file:pkg/util.py", "target": "symbol:aaa", "edge_type": "IMPLEMENTS", "properties": {}},
        {"source": "file:pkg/app.py", "target": "symbol:bbb", "edge_type": "IMPLEMENTS", "properties": {}},
        {"source": "symbol:bbb", "target": "symbol:aaa", "edge_type": "calls", "properties": {"name": "shared"}},
        {"source": "file:pkg/app.py", "target": "file:pkg/util.py", "edge_type": "depends_on", "properties": {"module": "pkg.util"}},
    ],
}


class FakeSource:
    """In-memory GitLabSource: projects + their files + raw contents."""

    def __init__(self, projects, files, contents):
        self._projects = projects
        self._files = files
        self._contents = contents

    def list_projects(self):
        return self._projects

    def list_files(self, project):
        return self._files.get(project.id, [])

    def get_file(self, project, path):
        return self._contents.get((project.id, path))


def _proj(pid="42", **kw):
    return GitLabProject(id=pid, path_with_namespace=f"grp/{pid}", default_branch="main", **kw)


def _source(pid="42"):
    files = {pid: ["pkg/util.py", "pkg/app.py", "README.md", "logo.png"]}
    contents = {
        (pid, "pkg/util.py"): b"def shared():\n    return 1\n",
        (pid, "pkg/app.py"): b"from pkg.util import shared\n\ndef run():\n    return shared()\n",
        (pid, "README.md"): b"# readme",
        (pid, "logo.png"): b"\x89PNG...",
    }
    return FakeSource([_proj(pid, last_activity_at="2026-06-01")], files, contents)


def test_index_instance_filters_code_and_maps_resolved_graph():
    sent_to_index: list = []

    def index_fn(files):
        sent_to_index.append(files)
        return INDEX_RESULT

    ingested: list = []

    def ingest(domain, entities, relationships):
        ingested.append((domain, entities, relationships))

    summary = index_instance(
        instance="test", source=_source(), index_fn=index_fn, ingest=ingest
    )

    # Only the two .py files were shipped to the resolver — README.md/logo.png skipped.
    assert len(sent_to_index) == 1
    paths = [p for p, _ in sent_to_index[0]]
    assert paths == ["pkg/util.py", "pkg/app.py"]

    assert summary.projects_indexed == 1
    assert summary.files_indexed == 2
    assert summary.symbols == 2
    assert summary.calls_resolved == 1
    assert summary.imports_resolved == 1
    assert summary.watermark == "2026-06-01"

    # Written under the per-instance source_system.
    domain, entities, rels = ingested[0]
    assert domain == "gitlab:test"
    types = {e["type"] for e in entities}
    assert {"Repository", "File", "Code"} <= types
    rel_types = sorted(r["type"] for r in rels)
    assert rel_types == ["CONTAINS", "CONTAINS", "IMPLEMENTS", "IMPLEMENTS", "calls", "depends_on"]


def test_no_dangling_edge_endpoints_and_namespacing():
    entities, rels = map_index_result(INDEX_RESULT, project=_proj("7"), instance="acme")
    ids = {e["id"] for e in entities}
    for r in rels:
        assert r["source"] in ids, f"dangling source {r['source']}"
        assert r["target"] in ids, f"dangling target {r['target']}"
    # Engine ids are namespaced by instance+project so they never collide globally.
    assert all(e["id"].startswith("gitlab:acme:7:") for e in entities)
    # The resolved call carries the callee name through.
    call = next(r for r in rels if r["type"] == "calls")
    assert call["name"] == "shared"


def test_project_ids_scoping_skips_others():
    summary = index_instance(
        instance="test",
        source=_source("42"),
        index_fn=lambda f: INDEX_RESULT,
        ingest=lambda *a: None,
        project_ids={"99"},
    )
    assert summary.projects_indexed == 0


def test_delta_watermark_skips_untouched_projects():
    summary = index_instance(
        instance="test",
        source=_source("42"),  # project last_activity_at = 2026-06-01
        index_fn=lambda f: INDEX_RESULT,
        ingest=lambda *a: None,
        since="2026-07-01",  # newer than the project → skipped
    )
    assert summary.projects_indexed == 0
    assert summary.projects_skipped == 1


class _FakeCompute:
    supports_index_repository = True

    def index_repository(self, files):
        return INDEX_RESULT


class _FakeEngine:
    def __init__(self):
        self.graph_compute = _FakeCompute()
        self.backend = None
        self.batches: list = []

    def ingest_external_batch(self, domain, entities, relationships=None):
        self.batches.append((domain, entities, relationships))
        return {"status": "success"}


def test_sync_source_routes_to_gitlab_handler():
    engine = _FakeEngine()
    res = sync_source(engine, "gitlab", mode="full", client=_source())
    assert res["status"] == "ok"
    assert res["source"] == "gitlab"
    assert res["calls_resolved"] == 1
    assert res["projects_indexed"] == 1
    assert engine.batches and engine.batches[0][0] == "gitlab:gitlab"


def test_sync_source_skips_when_engine_lacks_index_repository():
    engine = _FakeEngine()
    engine.graph_compute.supports_index_repository = False
    res = sync_source(engine, "gitlab", mode="full", client=_source())
    assert res["status"] == "skipped"
