#!/usr/bin/python
"""Tests for the unattended overnight loop-driver core.

CONCEPT:ECO-4.47 — Unattended overnight loop-driver + morning-summary write-back
CONCEPT:SAFE-1.8 — Unattended-session stop-on-ask containment
"""

from __future__ import annotations

from pathlib import Path

from agent_utilities.claude_harness import overnight_runner as ovr


class _FakeController:
    """A LoopController stand-in returning a scripted sequence of cycle reports."""

    def __init__(self, reports):
        self._reports = list(reports)
        self.calls = 0

    def run_one_cycle(self, **kwargs):
        self.calls += 1
        idx = min(self.calls - 1, len(self._reports) - 1)
        return self._reports[idx]


def _productive(resolved=1):
    return {"topics_resolved": resolved, "sources_linked": 0, "errors": []}


def _empty():
    return {
        "topics_resolved": 0,
        "sources_linked": 0,
        "skill_proposals": {"proposed": 0},
        "errors": [],
    }


def test_cycle_productive_detection():
    assert ovr.cycle_productive(_productive())
    assert ovr.cycle_productive({"skill_proposals": {"proposed": 2}})
    assert not ovr.cycle_productive(_empty())
    assert not ovr.cycle_productive({})


def test_run_session_stops_on_convergence(tmp_path: Path):
    # Two empty cycles in a row -> converged before the cap of 10.
    ctrl = _FakeController([_empty()])
    report = ovr.run_session(
        max_cycles=10,
        workspace=tmp_path,
        controller=ctrl,
        commit=False,
        convergence_patience=2,
    )
    assert report["stop_reason"] == "converged"
    assert ctrl.calls == 2  # stopped at the patience threshold
    assert report["productive_cycles"] == 0


def test_run_session_hits_cap(tmp_path: Path):
    ctrl = _FakeController([_productive()])  # always productive -> never converges
    report = ovr.run_session(
        max_cycles=3, workspace=tmp_path, controller=ctrl, commit=False
    )
    assert report["stop_reason"] == "max_cycles"
    assert report["cycles_run"] == 3
    assert report["productive_cycles"] == 3


def test_commit_called_per_productive_cycle(tmp_path: Path, monkeypatch):
    calls = []

    def _fake_commit(workspace, message):
        calls.append(message)
        return {"committed": True, "message": message}

    monkeypatch.setattr(ovr, "_git_commit", _fake_commit)
    # productive, productive, empty, empty -> 2 commits, then converge.
    ctrl = _FakeController([_productive(), _productive(), _empty(), _empty()])
    ovr.run_session(max_cycles=10, workspace=tmp_path, controller=ctrl, commit=True)
    assert len(calls) == 2
    assert all("unattended cycle" in m for m in calls)


def test_morning_summary_written_to_memory_md(tmp_path: Path):
    ctrl = _FakeController([_productive(), _empty(), _empty()])
    report = ovr.run_session(
        max_cycles=10, workspace=tmp_path, controller=ctrl, commit=False
    )
    memory = tmp_path / "MEMORY.md"
    assert memory.exists()
    text = memory.read_text()
    assert "Unattended loop session" in text
    assert ovr._MEMORY_START in text and ovr._MEMORY_END in text
    assert report["summary_path"] == str(memory)


def test_morning_summary_replaces_in_place(tmp_path: Path):
    # Pre-existing operator content is preserved; our block is replaced, not duplicated.
    memory = tmp_path / "MEMORY.md"
    memory.write_text("# Operator notes\n\nkeep me\n", encoding="utf-8")
    ctrl = _FakeController([_empty(), _empty()])
    ovr.run_session(max_cycles=10, workspace=tmp_path, controller=ctrl, commit=False)
    ovr.run_session(max_cycles=10, workspace=tmp_path, controller=ctrl, commit=False)
    text = memory.read_text()
    assert "keep me" in text
    assert text.count(ovr._MEMORY_START) == 1  # not duplicated across runs


def test_surfaced_by_memory_bridge(tmp_path: Path):
    # The summary lands where inject_project_context (KG-2.1) reads it.
    ctrl = _FakeController([_productive(), _empty(), _empty()])
    ovr.run_session(max_cycles=10, workspace=tmp_path, controller=ctrl, commit=False)
    from agent_utilities.knowledge_graph.core.agents_md import inject_project_context

    out = inject_project_context("BASE PROMPT", str(tmp_path), include_memory=True)
    assert "Unattended loop session" in out
