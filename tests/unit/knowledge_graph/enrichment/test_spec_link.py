"""Spec ↔ ticket ↔ agent linking (CONCEPT:AU-KG.ingest.enterprise-source-extractor)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.writeback import spec_link
from agent_utilities.knowledge_graph.enrichment.writeback.spec_link import (
    link_spec,
    pull_assigned,
)

SPEC = {
    "feature_id": "feat-auth-v2",
    "title": "Auth v2",
    "user_stories": [
        {"title": "SSO login", "description": "as a user I log in via SSO"}
    ],
    "non_functional_requirements": ["p95 < 200ms"],
}


class FakePlane:
    def __init__(self):
        self.updated = []
        self.links = []
        self.comments = []

    def update_work_item(self, project_id, item_id, data):
        self.updated.append((project_id, item_id, data))

    def create_work_item_link(self, project_id, item_id, data):
        self.links.append((item_id, data))

    def create_work_item_comment(self, project_id, item_id, data):
        self.comments.append((item_id, data))

    def list_work_items(self, project_id):
        return type(
            "R",
            (),
            {
                "data": {
                    "results": [
                        {
                            "id": "w1",
                            "name": "Do SSO",
                            "assignees": ["u-alice"],
                            "state": "todo",
                        },
                        {"id": "w2", "name": "Other", "assignees": ["u-bob"]},
                    ]
                }
            },
        )


def test_link_dry_run_records_external_link():
    spec = dict(SPEC)
    out = link_spec(
        spec, target="plane", issue_id="w1", project_id="proj", dry_run=True
    )
    assert out["status"] == "completed" and out["dry_run"] is True
    ops = {a["op"] for a in out["proposals"]}
    assert {"update_item", "link_spec"} <= ops
    assert spec["external_links"]["plane"] == "w1"


def test_link_live_assigns_and_links(monkeypatch):
    monkeypatch.setattr(
        spec_link,
        "setting",
        lambda k, d=None, cast=None: True if "ENABLE_WRITE" in k else (d or ""),
    )
    client = FakePlane()
    out = link_spec(
        SPEC,
        target="plane",
        issue_id="w1",
        project_id="proj",
        assignee="u-alice",
        comment="linked from KG",
        client=client,
        dry_run=False,
    )
    assert out["status"] == "completed"
    assert out["assignee"] == "u-alice"
    assert client.updated[0][2]["assignees"] == ["u-alice"]
    assert client.links and client.comments


def test_agent_maps_to_bot_user(monkeypatch):
    def fake_setting(k, d=None, cast=None):
        if "ENABLE_WRITE" in k:
            return True
        if k == "AGENT_USER_MAP":
            return '{"a2a:planner": {"plane": "u-bot"}}'
        return d or ""

    monkeypatch.setattr(spec_link, "setting", fake_setting)
    client = FakePlane()
    out = link_spec(
        SPEC,
        target="plane",
        issue_id="w1",
        project_id="proj",
        agent="a2a:planner",
        client=client,
        dry_run=False,
    )
    assert out["assignee"] == "u-bot"


def test_link_refused_without_flag(monkeypatch):
    monkeypatch.setattr(spec_link, "setting", lambda k, d=None, cast=None: False)
    out = link_spec(SPEC, target="plane", issue_id="w1", dry_run=False)
    assert out["status"] == "refused"


def test_pull_assigned_items():
    out = pull_assigned("plane", user="u-alice", project_id="proj", client=FakePlane())
    assert out["status"] == "completed"
    assert [i["id"] for i in out["items"]] == ["w1"]  # only u-alice's item
