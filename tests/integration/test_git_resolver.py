"""CONCEPT:ECO-4.43 — git issue/PR resolver: parse -> classify -> ingest KG object -> enqueue."""

from __future__ import annotations

from agent_utilities.integrations.git_resolver import (
    FAILING_CHECKS,
    MERGE_CONFLICTS,
    OPEN_ISSUE,
    OPEN_PR,
    classify,
    ingest_task,
    parse_webhook,
    resolve_and_dispatch,
    suggested_tasks,
)


def test_classify_taxonomy():
    assert classify(False) == OPEN_ISSUE
    assert classify(True) == OPEN_PR
    assert classify(True, checks_failing=True) == FAILING_CHECKS
    assert classify(True, mergeable=False) == MERGE_CONFLICTS


def test_parse_github_issue():
    payload = {
        "repository": {"full_name": "o/r", "clone_url": "https://x/o/r.git"},
        "issue": {"number": 7, "title": "Bug in add", "body": "returns a-b"},
    }
    ref = parse_webhook(payload)
    assert ref.source == "github" and ref.repo == "o/r" and ref.number == 7
    assert ref.kind == OPEN_ISSUE
    assert ref.task_id == "gittask:github:o/r#7"
    assert "Bug in add" in ref.problem_statement


def test_parse_gitlab_merge_request_conflict():
    payload = {
        "object_kind": "merge_request",
        "project": {"path_with_namespace": "g/p", "git_http_url": "https://x/g/p.git"},
        "object_attributes": {
            "iid": 3,
            "title": "MR",
            "description": "d",
            "merge_status": "cannot_be_merged",
            "target_branch": "main",
        },
    }
    ref = parse_webhook(payload)
    assert ref.source == "gitlab" and ref.repo == "g/p" and ref.kind == MERGE_CONFLICTS
    assert ref.base_ref == "main"


def test_parse_unsupported_returns_none():
    assert parse_webhook({"random": "noise"}) is None


class _Engine:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self._rows = []

    def add_node(self, nid, ntype, properties=None):
        self.nodes.append((nid, ntype, properties or {}))

    def add_edge(self, s, t, rel="", **p):
        self.edges.append((s, t, rel))

    class _B:
        def __init__(self, outer):
            self.outer = outer

        def execute(self, q, params=None):
            return self.outer._rows

    @property
    def backend(self):
        return _Engine._B(self)


def test_ingest_creates_gittask_object_and_repo_edge():
    eng = _Engine()
    payload = {
        "repository": {"full_name": "o/r", "clone_url": "u"},
        "issue": {"number": 1, "title": "t", "body": "b"},
    }
    ref = parse_webhook(payload)
    node_id = ingest_task(eng, ref)
    assert node_id == ref.task_id
    assert eng.nodes[0][1] == "GitTask"
    assert ("gittask:github:o/r#1", "repo:o/r", "TARGETS_REPO") in eng.edges


def test_resolve_and_dispatch_enqueues(monkeypatch):
    captured = {}

    def fake_enqueue(envelope, queue=None):
        captured["envelope"] = envelope
        return {"job_id": "j1", "status": "pending", "dispatch": "queued"}

    import agent_utilities.orchestration.agent_dispatch as ad

    monkeypatch.setattr(ad, "enqueue_agent_turn", fake_enqueue)

    eng = _Engine()
    payload = {
        "repository": {"full_name": "o/r", "clone_url": "u"},
        "issue": {"number": 9, "title": "Fix it", "body": "please"},
    }
    result = resolve_and_dispatch(payload, eng)
    assert result["task_id"] == "gittask:github:o/r#9"
    assert result["kind"] == OPEN_ISSUE
    assert result["job_id"] == "j1"
    assert captured["envelope"].agent_name == "swe_engineer"
    assert captured["envelope"].payload_ref == "gittask:github:o/r#9"


def test_suggested_tasks_query_shape():
    eng = _Engine()
    eng._rows = [
        {"id": "gittask:github:o/r#1", "repo": "o/r", "kind": "open_issue", "name": "t"}
    ]
    rows = suggested_tasks(eng, repo="o/r", kind="open_issue")
    assert rows and rows[0]["kind"] == "open_issue"
