"""Proves ``gateway_health``'s two lazy imports of
``agent_utilities.observability.health`` (moved to
``graph_os.observability.health``, GRAPHOS-HOST-R008) degrade gracefully now
that the module is permanently gone from AU.

Replaces the live-wiring half of the deleted
``test_gateway_health_evidence_live_path.py`` (which hard-imported
``observability.health`` at module level to prove the end-to-end anomaly
path): that path now lives with the kernel in graph-os
(``graph_os.observability.health``'s own test suite). What AU still owns —
``gateway_health.py`` itself, pending its own removal (AU #100) — must keep
tolerating the kernel's absence exactly as its "best-effort, never affects
the response path" docstring promises, the SAME contract already proven for
its ``health_ingest``/``native_ingest`` imports.

Simulates the permanent ``ModuleNotFoundError`` the real deletion causes by
setting ``sys.modules["agent_utilities.observability.health"] = None`` —
the standard import-system idiom that forces the next
``from ... import ...`` of that name to raise, without needing the file to
actually be absent from this checkout.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_utilities.observability import gateway_health as gh


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    monkeypatch.setattr(gh, "_buffer", None)
    monkeypatch.setattr(gh, "_history", [])
    monkeypatch.setitem(
        __import__("sys").modules, "agent_utilities.observability.health", None
    )
    yield


def test_record_request_duration_skips_without_raising(caplog):
    """``_get_buffer()``'s lazy import of the moved kernel fails closed: the
    request path never sees an exception."""
    gh.record_request_duration(0.05)  # must not raise
    assert gh._buffer is None  # buffer was never constructed


@pytest.mark.asyncio
async def test_check_and_record_logs_and_returns_without_raising(caplog):
    """``_check_and_record``'s lazy import of the moved kernel fails closed
    before it ever reaches ``health_ingest``."""
    trend = {
        "min": 0.04,
        "max": 0.06,
        "avg": 0.05,
        "avg_control": None,
        "samples": 10,
        "window_s": gh._WINDOW_S,
        "start_at": 1_000_000.0,
        "end_at": 1_000_000.0 + gh._WINDOW_S,
    }
    caplog.set_level("DEBUG", logger=gh.logger.name)
    await gh._check_and_record(trend)  # must not raise
    assert "gateway health anomaly check failed" in caplog.text
    # The history append happens AFTER the health import in the try block, so
    # a kernel import failure here means the window is dropped, not queued.
    assert gh._history == []


def test_record_request_duration_flush_also_skips_the_check(monkeypatch):
    """The real entry point: enough samples to cross the window boundary
    still schedules `_check_and_record`, which itself degrades (proven
    above) -- the buffer construction is what's actually gone."""
    asyncio.run(_drive(monkeypatch))


async def _drive(monkeypatch) -> None:
    gh.record_request_duration(0.05)
    assert gh._buffer is None
    gh.record_request_duration(0.06)
    assert gh._buffer is None
