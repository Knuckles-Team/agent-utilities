"""Non-blocking ``ingest_sessions`` upload path (CONCEPT:AU-KG.ingest.drain-session-bundle).

A large remote session-bundle upload must ENQUEUE a durable ``session_upload``
background task and return a ``job_id`` immediately (mirroring ``source_sync`` /
``graph_ingest``) rather than synchronously running the per-bundle
``record_bundle`` loop, which blew past the 60s MCP client window under load.
A small batch still runs inline. The host worker's drain step then lands the
bundles in the usage store.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin
from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools.write_ingest_tools import register_write_ingest_tools
from agent_utilities.usage.backends.sqlite_fts import SqliteUsageBackend
from agent_utilities.usage.models import (
    ParsedSessionBundle,
    UsageEvent,
    UsageSession,
)
from agent_utilities.usage.recorder import UsageRecorder


# ── fixtures ────────────────────────────────────────────────────────────────
class _FakeMCP:
    """Capture ``@mcp.tool(...)``-decorated functions without FastMCP."""

    def tool(self, *args, **kwargs):
        def _wrap(fn):
            return fn

        return _wrap


def _bundle_dict(sid: str) -> dict:
    return ParsedSessionBundle(
        session=UsageSession(
            id=sid,
            project="p",
            agent="claude",
            message_count=1,
            total_output_tokens=10,
            outcome="success",
        ),
        usage_events=[
            UsageEvent(
                session_id=sid,
                source="agent",
                model="claude-opus-4-8",
                input_tokens=100,
                output_tokens=10,
                dedup_key=f"{sid}-e1",
            ),
        ],
    ).model_dump(mode="json")


@pytest.fixture()
def upload_tool():
    register_write_ingest_tools(_FakeMCP())
    return kg_server.REGISTERED_TOOLS["ingest_sessions"]


class _CapturingEngine:
    def __init__(self):
        self.calls: list[dict] = []

    def submit_task(self, **kwargs):
        self.calls.append(kwargs)
        return "job-test-123"


class _CountingRecorder:
    def __init__(self):
        self.count = 0

    def record_bundle(self, bundle) -> bool:  # noqa: ANN001
        self.count += 1
        return True


# ── enqueue half ────────────────────────────────────────────────────────────
def test_large_upload_enqueues_and_returns_fast(upload_tool, monkeypatch):
    engine = _CapturingEngine()
    recorder = _CountingRecorder()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    monkeypatch.setattr(
        "agent_utilities.usage.recorder.get_usage_recorder", lambda: recorder
    )

    bundles = [_bundle_dict(f"s{i}") for i in range(10)]
    out = json.loads(
        asyncio.run(
            upload_tool(action="upload", bundles_json=json.dumps(bundles), tenant_id="")
        )
    )

    # Returns enqueued immediately — NOT synchronously ingested.
    assert out["status"] == "enqueued"
    assert out["job_id"] == "job-test-123"
    assert out["received"] == 10
    assert recorder.count == 0  # the heavy record_bundle loop did NOT run inline

    # Enqueued as the right task type, carrying the bundles on the payload.
    assert len(engine.calls) == 1
    call = engine.calls[0]
    assert call["task_type"] == "session_upload"
    assert call["skip_dedupe"] is True
    # The tool privacy-normalizes each bundle (normalize_bundle) before it
    # ever reaches the durable queue -- session/tenant ids are pseudonymized,
    # so the enqueued payload is never byte-identical to the raw input.
    # Reproduce the same transform to assert what actually should be there.
    from agent_utilities.usage.authorization import resolve_usage_tenant
    from agent_utilities.usage.privacy import normalize_bundle

    authoritative_tenant = resolve_usage_tenant(None)
    expected_bundles = []
    for raw_bundle in bundles:
        b = ParsedSessionBundle.model_validate(raw_bundle)
        if authoritative_tenant:
            b.session.tenant_id = authoritative_tenant
        expected_bundles.append(normalize_bundle(b).model_dump())
    assert call["extra_meta"]["payload"]["bundles"] == expected_bundles


def test_small_upload_runs_inline(upload_tool, monkeypatch):
    engine = _CapturingEngine()
    recorder = _CountingRecorder()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    monkeypatch.setattr(
        "agent_utilities.usage.recorder.get_usage_recorder", lambda: recorder
    )

    bundles = [_bundle_dict("s0")]
    out = json.loads(
        asyncio.run(
            upload_tool(action="upload", bundles_json=json.dumps(bundles), tenant_id="")
        )
    )

    assert out["status"] == "ingested"
    assert out["received"] == 1 and out["ingested"] == 1
    assert recorder.count == 1  # tiny batch → inline, never enqueued
    assert engine.calls == []


# ── drain half ──────────────────────────────────────────────────────────────
class _FakeWorkItemAuthority:
    """Minimal ``query_cypher`` double satisfying ``work_item.get_work_item``.

    ``_ingest_task_metadata`` now reads the ingestion definition off the sole
    native WorkItem (``work_item.get_work_item``) instead of a raw
    ``_control_cypher`` row -- metadata lives there as a plain dict (WorkItem
    write paths like ``submit_work_item`` store it natively, never
    base64-encoded; ``_encode_metadata``/``_decode_metadata`` are a distinct,
    unrelated legacy Cypher-storage helper).
    """

    def __init__(self, metadata: dict):
        self._metadata = metadata

    def query_cypher(self, query, params=None):  # noqa: ANN001
        return [{"id": (params or {}).get("id", ""), "metadata": self._metadata}]


class _DrainEngine:
    """Minimal carrier for the extracted ``_drain_session_upload`` helper."""

    _drain_session_upload = TaskManagerMixin._drain_session_upload
    _ingest_task_metadata = TaskManagerMixin._ingest_task_metadata

    def __init__(self, metadata: dict):
        self._work_item_engine = _FakeWorkItemAuthority(metadata)
        self.final = None

    def _update_task_status(self, job_id, status, info):  # noqa: ANN001
        self.final = (status, info)


def test_drain_lands_bundles_in_usage_store(tmp_path, monkeypatch):
    bundles = [_bundle_dict(f"s{i}") for i in range(10)]
    metadata = {"payload": {"bundles": bundles, "tenant_id": ""}}

    backend = SqliteUsageBackend(tmp_path / "usage.db")
    backend.ensure_schema()
    monkeypatch.setattr(
        "agent_utilities.usage.recorder.get_usage_recorder",
        lambda: UsageRecorder(backend),
    )

    eng = _DrainEngine(metadata)
    result = eng._drain_session_upload("job-test-123")

    assert result == {"received": 10, "ingested": 10}
    assert eng.final[0] == "completed"
    assert eng.final[1]["type"] == "session_upload"
    # The 10 enqueued sessions are now durably in the usage store.
    assert backend.summary().session_count == 10
