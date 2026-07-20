"""Ticket status/comment writeback sinks (KG-2.126) — dry-run + live apply."""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.enrichment.writeback.core import WritebackContext
from agent_utilities.knowledge_graph.enrichment.writeback.sinks.issue_tracker import (
    JiraTransitionSink,
    PlaneStateSink,
)


class _FakeJira:
    def __init__(self) -> None:
        self.transitions: list[tuple] = []
        self.comments: list[tuple] = []

    def jira_cloud_get_transitions(self, issue_id_or_key: str):
        return type(
            "R",
            (),
            {
                "data": {
                    "transitions": [
                        {
                            "id": "31",
                            "name": "Ready for QA",
                            "to": {"name": "Ready for QA"},
                        }
                    ]
                }
            },
        )()

    def jira_cloud_do_transition(self, issue_id_or_key: str, payload: dict[str, Any]):
        self.transitions.append((issue_id_or_key, payload["transition"]["id"]))

    def jira_cloud_add_comment(self, issue_id_or_key: str, payload: dict[str, Any]):
        self.comments.append((issue_id_or_key, payload))


def test_jira_transition_dry_run_proposes_without_calling():
    sink = JiraTransitionSink()
    ops = {
        "transitions": [{"key": "DB-1", "status": "Ready for QA", "comment": "done"}]
    }
    res = sink.run(WritebackContext(), ops, dry_run=True)
    assert res.proposals == [
        {"op": "transition", "ticket": "DB-1", "to": "Ready for QA"}
    ]
    assert res.enriched == 0


def test_jira_transition_live_applies_transition_and_comment():
    client = _FakeJira()
    sink = JiraTransitionSink()
    ops = {
        "client": client,
        "transitions": [{"key": "DB-1", "status": "Ready for QA", "comment": "done"}],
    }
    res = sink.run(WritebackContext(), ops, dry_run=False)
    assert res.enriched == 1
    assert client.transitions == [("DB-1", "31")]
    assert client.comments and client.comments[0][0] == "DB-1"


def test_jira_transition_high_stakes():
    assert getattr(JiraTransitionSink(), "risk_tier", None) == "high_stakes"


class _FakePlane:
    def __init__(self) -> None:
        self.updates: list[tuple] = []
        self.comments: list[tuple] = []

    def update_work_item(self, project_id, work_item_id, data):
        self.updates.append((project_id, work_item_id, data))

    def create_work_item_comment(self, project_id, work_item_id, data):
        self.comments.append((project_id, work_item_id, data))


def test_plane_state_live_updates_state_and_comment():
    client = _FakePlane()
    sink = PlaneStateSink()
    ops = {
        "client": client,
        "transitions": [
            {
                "project_id": "p1",
                "work_item_id": "wi-1",
                "state": "s-done",
                "comment": "ok",
            }
        ],
    }
    res = sink.run(WritebackContext(), ops, dry_run=False)
    assert res.enriched == 1
    assert client.updates == [("p1", "wi-1", {"state": "s-done"})]
    assert client.comments and client.comments[0][:2] == ("p1", "wi-1")
