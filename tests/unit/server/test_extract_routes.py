"""The fact-extraction gateway contract is registered (CONCEPT:AU-ECO.connector.git-task-resolver).

The shared SSE/jobs/JSONL surface all three frontends consume. With a cold engine
the endpoints must degrade gracefully (``unavailable``), never 500.
"""

from __future__ import annotations

import pytest

from agent_utilities.server.routers import enhanced


def test_extract_routes_registered() -> None:
    paths = {r.path for r in enhanced.router.routes}
    for p in (
        "/api/enhanced/extract/submit",
        "/api/enhanced/extract/stream/{job_id}",
        "/api/enhanced/extract/jobs",
        "/api/enhanced/extract/status/{job_id}",
        "/api/enhanced/extract/jsonl/{job_id}",
        "/api/enhanced/extract/pause/{job_id}",
        "/api/enhanced/extract/resume/{job_id}",
    ):
        assert p in paths


@pytest.mark.asyncio
async def test_extract_jobs_graceful_when_cold(monkeypatch) -> None:
    # Force the "engine cold" path; the endpoint must answer, not raise.
    monkeypatch.setattr(enhanced, "_active_engine", lambda: None)
    enhanced._EXTRACTION_MANAGER = None
    res = await enhanced.extract_jobs()
    assert res["status"] == "unavailable"
    assert res["jobs"] == []
