"""Git-SHA pre-skip for the always-on breadth codebase loop (CONCEPT:KG-2.8).

An always-on LoopController breadth tick must not re-enqueue (or stat-walk) a repo
that is still at the HEAD we last ingested. ``_default_codebase_ingest`` mirrors the
engine's ``codebase_git`` watermark so it skips only when the engine would have.
"""

from __future__ import annotations

import types

import agent_utilities.knowledge_graph.ingestion.engine as eng
import agent_utilities.knowledge_graph.ingestion.manifest as man
from agent_utilities.knowledge_graph.assimilation.breadth_ingest import (
    ProjectManifest,
    _default_codebase_ingest,
)


class _Engine:
    backend = None
    graph_compute = types.SimpleNamespace(graph_name="__commons__")

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def submit_task(self, **kw):
        self.calls.append(kw)
        return "job-x"


def _patch(monkeypatch, *, head, clean, stored):
    monkeypatch.setattr(eng, "_git_head_sha", lambda p: head)
    monkeypatch.setattr(eng, "_git_worktree_clean", lambda p: clean)

    class _DM:
        def __init__(self, **kw):
            pass

        def get(self, g, c, u):
            return stored

    monkeypatch.setattr(man, "DeltaManifest", _DM)


M = ProjectManifest(name="repo", path="/repo", language="python")


def test_skips_unchanged_clean_repo(monkeypatch):
    _patch(monkeypatch, head="abc", clean=True, stored="abc")
    e = _Engine()
    assert _default_codebase_ingest(e, M) is False
    assert e.calls == []  # no task enqueued


def test_submits_when_head_advanced(monkeypatch):
    _patch(monkeypatch, head="def", clean=True, stored="abc")
    e = _Engine()
    assert _default_codebase_ingest(e, M) is True
    assert len(e.calls) == 1 and e.calls[0]["is_codebase"] is True


def test_submits_dirty_worktree(monkeypatch):
    # uncommitted changes at the same HEAD must still ingest
    _patch(monkeypatch, head="abc", clean=False, stored="abc")
    e = _Engine()
    assert _default_codebase_ingest(e, M) is True
    assert len(e.calls) == 1


def test_submits_non_git(monkeypatch):
    _patch(monkeypatch, head=None, clean=True, stored=None)
    e = _Engine()
    assert _default_codebase_ingest(e, M) is True
    assert len(e.calls) == 1
