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


# ── CONCEPT:KG-2.150 — dirty self-repo scoped to git-status-modified files ──

import agent_utilities.knowledge_graph.assimilation.breadth_ingest as bi


def test_dirty_self_repo_scopes_to_modified_files(monkeypatch, tmp_path):
    """A DIRTY agent-utilities self-checkout submits a task scoped to just its
    git-status-modified source files (only_files)."""
    # Make tmp_path look like the self-repo checkout.
    pkg = tmp_path / "agent_utilities"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    self_m = ProjectManifest(
        name="agent-utilities", path=str(tmp_path), language="python"
    )

    _patch(monkeypatch, head="abc", clean=False, stored="abc")
    monkeypatch.setattr(
        bi, "_git_modified_source_files", lambda p: ["/x/a.py", "/x/b.py"]
    )
    e = _Engine()
    assert _default_codebase_ingest(e, self_m) is True
    assert len(e.calls) == 1
    assert e.calls[0]["extra_meta"] == {"only_files": ["/x/a.py", "/x/b.py"]}


def test_dirty_self_repo_no_source_changes_skips(monkeypatch, tmp_path):
    """A dirty self-repo whose modified files are all non-source enqueues nothing."""
    pkg = tmp_path / "agent_utilities"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    self_m = ProjectManifest(
        name="agent-utilities", path=str(tmp_path), language="python"
    )

    _patch(monkeypatch, head="abc", clean=False, stored="abc")
    monkeypatch.setattr(bi, "_git_modified_source_files", lambda p: [])
    e = _Engine()
    assert _default_codebase_ingest(e, self_m) is False
    assert e.calls == []


def test_dirty_non_self_repo_unscoped_submit(monkeypatch):
    """A dirty NON-self repo still submits a whole-repo task (no only_files)."""
    _patch(monkeypatch, head="abc", clean=False, stored="abc")
    e = _Engine()
    assert _default_codebase_ingest(e, M) is True  # M.path == "/repo" (not self)
    assert len(e.calls) == 1
    assert "extra_meta" not in e.calls[0]


def test_git_modified_source_files_filters(monkeypatch):
    """_git_modified_source_files keeps source files, drops deletions/non-source/vendored."""
    import subprocess as _sp

    porcelain = (
        " M agent_utilities/core/foo.py\n"
        "?? newmodule.py\n"
        " D removed.py\n"
        " M README.md\n"
        " M .venv/lib/site.py\n"
        "R  old.py -> agent_utilities/renamed.py\n"
    )

    class _CP:
        returncode = 0
        stdout = porcelain

    monkeypatch.setattr(_sp, "run", lambda *a, **k: _CP())
    # Make every candidate appear to exist as a file.
    monkeypatch.setattr(bi.Path, "is_file", lambda self: True)
    out = bi._git_modified_source_files("/anyrepo")
    names = {p.rsplit("/", 1)[-1] for p in out}
    assert "foo.py" in names and "newmodule.py" in names and "renamed.py" in names
    assert "removed.py" not in names  # deletion dropped
    assert "README.md" not in names  # non-source dropped
    assert "site.py" not in names  # vendored (.venv) dropped
