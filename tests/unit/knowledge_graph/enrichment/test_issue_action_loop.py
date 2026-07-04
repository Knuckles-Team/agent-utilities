"""Issue-tracking action loop: KG risk findings → tickets (CONCEPT:AU-KG.ingest.enterprise-source-extractor)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.writeback import (
    collect_risk_findings,
    core,
    run_writeback,
)


class FakeGitLab:
    def __init__(self):
        self.issues = []

    def create_issue(self, project_id, title, description):
        self.issues.append((project_id, title, description))


class FakeBackend:
    def execute(self, query, params=None):
        if "TechnologyRisk" in query:
            return [
                {
                    "id": "snrisk:m1",
                    "name": "PowerEdge EOL",
                    "rating": "high",
                    "eol": "2027-01-01",
                },
            ]
        return []


def test_collect_risk_findings():
    out = collect_risk_findings(FakeBackend())
    assert out[0]["title"] == "Technology risk: PowerEdge EOL"
    assert "high" in out[0]["body"] and "2027-01-01" in out[0]["body"]
    assert out[0]["node"] == "snrisk:m1"


def test_gitlab_issue_sink_dry_run():
    out = run_writeback(
        "gitlab",
        client=FakeGitLab(),
        creations=[{"title": "Tech debt", "project_id": "42"}],
        dry_run=True,
    )
    assert out["proposals"][0]["op"] == "create_issue"


def test_findings_to_issues_live(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    client = FakeGitLab()
    # The risk findings collected from the KG become filed GitLab issues.
    findings = [c | {"project_id": "42"} for c in collect_risk_findings(FakeBackend())]
    out = run_writeback(
        "gitlab",
        backend=FakeBackend(),
        client=client,
        creations=findings,
        dry_run=False,
    )
    assert out["created"] == 1
    assert client.issues[0][0] == "42"
    assert client.issues[0][1] == "Technology risk: PowerEdge EOL"


def test_issue_sink_refused_without_flag(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: False)
    out = run_writeback(
        "gitlab",
        client=FakeGitLab(),
        creations=[{"title": "x", "project_id": "1"}],
        dry_run=False,
    )
    assert out["status"] == "refused"
