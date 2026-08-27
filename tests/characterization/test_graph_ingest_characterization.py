"""CX-AU-03 characterization for ``register_write_ingest_tools.graph_ingest``
(CCN 215 pre-refactor) in ``agent_utilities/mcp/tools/write_ingest_tools.py``.

Measured fact motivating this file: ``graph_ingest`` had ZERO characterization
coverage of its action-dispatch chain before this lane (existing tests --
``tests/unit/mcp/test_graph_ingest_explicit_selection.py`` and
``tests/unit/mcp/test_external_graph_ingest_action.py`` -- cover one narrow
slice each: the U-06 explicit graph/connection selector on action='ingest',
and the external-graph ingest_connection path. Neither exercises the
dispatch chain itself). This file pins the OBSERVED behaviour of the giant
if/elif chain BEFORE it is decomposed into a dict dispatch of module-level
handlers, per the two-commit characterize-then-refactor discipline
(AGENTS.md). Written and made green against the UNMODIFIED function --
that green run is the only thing making this refactor reversible.
"""

from __future__ import annotations

import json

from agent_utilities.mcp import kg_server


class _FakeIngestEngine:
    """Just enough of ``IntelligenceGraphEngine`` for the cheap,
    validation-only action paths characterized here."""

    def list_tasks(self):
        return {}

    def get_task_status(self, job_id):
        return None

    def cancel_task(self, job_id):
        return {"cancelled": job_id}

    def clear_tasks(self, status_filter):
        return {"cleared": 0, "filter": status_filter}

    def prioritize_task(self, job_id, priority_bucket):
        return {"job_id": job_id, "priority_bucket": priority_bucket}


async def _call(monkeypatch, action: str, **kwargs) -> str:
    monkeypatch.setattr(kg_server, "_get_engine", lambda: _FakeIngestEngine())
    kg_server.ensure_tools_registered()
    return await kg_server._execute_tool("graph_ingest", action=action, **kwargs)


async def test_unknown_action_falls_back_with_the_action_name(monkeypatch):
    out = await _call(monkeypatch, "definitely-not-a-real-action")
    assert out == "Error: Unknown ingest action 'definitely-not-a-real-action'"


async def test_ingest_requires_target_path(monkeypatch):
    out = await _call(monkeypatch, "ingest", target_path="")
    assert out == "Error: target_path required for ingest action"


async def test_job_status_requires_job_id(monkeypatch):
    out = await _call(monkeypatch, "job_status", job_id="")
    assert out == "Error: job_id required"


async def test_status_alias_requires_job_id(monkeypatch):
    # 'status' is a documented alias for 'job_status' -- same handler.
    out = await _call(monkeypatch, "status", job_id="")
    assert out == "Error: job_id required"


async def test_cancel_requires_job_id(monkeypatch):
    out = await _call(monkeypatch, "cancel", job_id="")
    assert out == "Error: job_id required for cancel"


async def test_prioritize_requires_job_id(monkeypatch):
    out = await _call(monkeypatch, "prioritize", job_id="")
    assert out == "Error: job_id required for prioritize"


async def test_jobs_lists_with_no_active_jobs(monkeypatch):
    out = await _call(monkeypatch, "jobs")
    assert out == "No active or recent ingestion jobs."


async def test_clear_defaults_to_completed_filter(monkeypatch):
    out = await _call(monkeypatch, "clear", target_path="")
    assert json.loads(out) == {"cleared": 0, "filter": "completed"}


async def test_cancel_and_prioritize_are_independently_routed(monkeypatch):
    # A dict-dispatch bug class this specifically guards against: two action
    # names accidentally mapped to the SAME handler.
    cancel = await _call(monkeypatch, "cancel", job_id="")
    prioritize = await _call(monkeypatch, "prioritize", job_id="")
    assert cancel != prioritize
    assert "cancel" in cancel
    assert "prioritize" in prioritize
