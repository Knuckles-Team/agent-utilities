"""Nextcloud: calendar/contacts extract + calendar/file write-back (CONCEPT:AU-KG.ingest.enterprise-source-extractor)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.extractors import nextcloud as nc_ext
from agent_utilities.knowledge_graph.enrichment.writeback import core, run_writeback


class FakeNextcloud:
    def list_calendars(self):
        return [{"url": "https://nc/cal/work/"}]

    def list_events(self, calendar_url):
        return [
            {
                "uid": "ev1",
                "summary": "Arch review",
                "start": "20260701T100000Z",
                "end": "20260701T110000Z",
                "location": "Room 1",
            }
        ]

    def list_address_books(self):
        return [{"url": "https://nc/ab/personal/"}]

    def list_contacts(self, ab_url):
        return [{"uid": "c1", "fn": "Ada Lovelace", "email": "ada@x.io"}]

    def __init__(self):
        self.events = []
        self.files = []

    def create_calendar_event(self, calendar_url, event_data, filename=None):
        self.events.append((calendar_url, event_data))
        return True

    def write_file(self, path, content):
        self.files.append((path, content))


def test_extract_calendar_and_contacts():
    batch = nc_ext.extract({"client": FakeNextcloud()})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["event:ev1"].type == "CalendarEvent"
    assert by_id["event:ev1"].props["scheduledStart"] == "20260701T100000Z"
    assert by_id["event:ev1"].props["externalToolId"] == "ev1"
    assert by_id["event:ev1"].props["domain"] == "nextcloud"
    assert by_id["contact:c1"].type == "Person"
    assert by_id["contact:c1"].props["email"] == "ada@x.io"


def test_writeback_create_event_dry_run_and_live(monkeypatch):
    out = run_writeback(
        "nextcloud",
        client=FakeNextcloud(),
        creations=[{"type": "CalendarEvent", "name": "EOL review"}],
        dry_run=True,
    )
    assert out["proposals"][0]["op"] == "create_calendar_event"

    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: True)
    client = FakeNextcloud()
    out = run_writeback(
        "nextcloud",
        client=client,
        creations=[{"type": "CalendarEvent", "name": "EOL review", "node": "ev9"}],
        dry_run=False,
    )
    assert out["created"] == 1
    assert client.events and "SUMMARY:EOL review" in client.events[0][1]


def test_writeback_refused_without_flag(monkeypatch):
    monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: False)
    out = run_writeback(
        "nextcloud",
        client=FakeNextcloud(),
        creations=[{"type": "CalendarEvent", "name": "x"}],
        dry_run=False,
    )
    assert out["status"] == "refused"


def test_doc_source_preset_registered():
    from agent_utilities.protocols.source_connectors.connectors.mcp_tool import (
        MCP_TOOL_PRESETS,
    )

    p = MCP_TOOL_PRESETS["nextcloud-files"]
    assert p["server"] == "nextcloud-agent"
    assert p["action"] == "list_files"
    assert p["detail"]["action"] == "read_file"
