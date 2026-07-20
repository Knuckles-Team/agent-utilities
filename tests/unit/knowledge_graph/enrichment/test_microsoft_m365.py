"""Phase-fixup — Microsoft 365 async ingestion path (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Proves the sync-over-async bridge that lets the fully-async Microsoft Graph
client flow through the synchronous materialize/extractor path, plus the M365
extractor mapping to canonical :CalendarEvent / :Person.
"""

from __future__ import annotations

import asyncio

from agent_utilities.knowledge_graph.enrichment.extractors import microsoft as ms_ext
from agent_utilities.knowledge_graph.enrichment.materialize import MATERIALIZE_SOURCES
from agent_utilities.knowledge_graph.enrichment.source_adapters import (
    MicrosoftGraphSourceClient,
    run_sync,
)


class FakeAsyncGraph:
    """Mimics the async MicrosoftGraphApi surface (coroutine methods, Graph envelope)."""

    async def list_calendar_events(self, params=None):
        return {
            "value": [
                {
                    "id": "e1",
                    "subject": "Standup",
                    "start": {"dateTime": "2026-06-15T09:00:00"},
                    "end": {"dateTime": "2026-06-15T09:15:00"},
                    "location": {"displayName": "Room A"},
                }
            ]
        }

    async def list_users(self, params=None):
        return {"value": [{"id": "u1", "displayName": "Alice", "mail": "alice@x.com"}]}


def test_run_sync_no_running_loop():
    async def coro():
        return 42

    assert run_sync(coro) == 42


def test_run_sync_inside_running_loop():
    """The thread-bridge path: call the sync adapter from within a live loop."""

    async def driver():
        client = MicrosoftGraphSourceClient(FakeAsyncGraph())
        # adapter._call must complete even though a loop is already running here
        return client.calendar_events()

    rows = asyncio.run(driver())
    assert rows[0]["id"] == "e1"


def test_adapter_unwraps_graph_value_envelope():
    client = MicrosoftGraphSourceClient(FakeAsyncGraph())
    events = client.calendar_events()
    users = client.users()
    assert events[0]["subject"] == "Standup"
    assert users[0]["displayName"] == "Alice"


def test_adapter_accepts_sync_fake_too():
    class SyncGraph:
        def list_users(self):
            return [{"id": "u9", "displayName": "Bob"}]

    assert MicrosoftGraphSourceClient(SyncGraph()).users()[0]["id"] == "u9"


def test_microsoft_extract_maps_canonical_types():
    client = MicrosoftGraphSourceClient(FakeAsyncGraph())
    batch = ms_ext.extract({"client": client})
    by_id = {n.id: n for n in batch.nodes}
    ev = by_id["msevent:e1"]
    assert ev.type == "CalendarEvent"
    assert ev.props["name"] == "Standup"
    assert ev.props["scheduledStart"] == "2026-06-15T09:00:00"
    assert ev.props["eventLocation"] == "Room A"
    assert ev.props["domain"] == "microsoft"
    assert ev.props["externalToolId"] == "e1"
    user = by_id["msuser:u1"]
    assert user.type == "Person"
    assert user.props["email"] == "alice@x.com"


def test_microsoft_is_a_materialize_source():
    assert "microsoft" in MATERIALIZE_SOURCES
