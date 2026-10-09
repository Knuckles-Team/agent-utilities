"""AU-INTEGRATION-R022 — the connector-sync sweep dispatches every configured
connector `sync_source` knows (not just the three hardcoded research-feed
sources), skips unconfigured ones, and isolates one source's failure from the
rest."""

from __future__ import annotations

import pytest

from agent_utilities.core import schedule_engine as se
from agent_utilities.knowledge_graph.core import source_sync


class _FakeEngine:
    """A minimal stand-in; the dispatcher never touches engine internals
    directly -- it only threads it through to ``sync_source``."""


@pytest.mark.spec("AU-INTEGRATION-R022")
def test_connector_sync_sweep_dispatches_configured_skips_unconfigured_isolates_errors(
    monkeypatch,
):
    calls: list[tuple[str, str]] = []

    def _fake_sync_source(engine, source, *, mode="delta", **_kw):
        calls.append((source, mode))
        if source == "unconfigured_source":
            return {"status": "skipped", "reason": "not configured"}
        if source == "broken_source":
            raise RuntimeError("boom")
        return {"status": "ok", "source": source}

    # Monkeypatch the REAL registry/dispatch table the dispatcher iterates --
    # never a hardcoded list in the dispatcher itself.
    monkeypatch.setattr(
        source_sync,
        "_DELTA_HANDLERS",
        {
            "configured_source": lambda *a, **k: None,
            "unconfigured_source": lambda *a, **k: None,
            "broken_source": lambda *a, **k: None,
        },
    )
    monkeypatch.setattr(source_sync, "sync_source", _fake_sync_source)

    result = se._dispatch_connector_sync_sweep(_FakeEngine(), {"kind": "connector_sync_sweep"})

    assert result["status"] == "ok"
    # Every registered source is dispatched through sync_source with mode=delta.
    assert sorted(calls) == [
        ("broken_source", "delta"),
        ("configured_source", "delta"),
        ("unconfigured_source", "delta"),
    ]
    # Configured source ran, unconfigured source's own skip is surfaced as-is.
    assert result["sources"]["configured_source"]["status"] == "ok"
    assert result["sources"]["unconfigured_source"]["status"] == "skipped"
    # The raising source is isolated as an error -- it never stops the sweep.
    assert result["sources"]["broken_source"]["status"] == "error"
    assert "boom" in result["sources"]["broken_source"]["reason"]


@pytest.mark.spec("AU-INTEGRATION-R022")
def test_connector_sync_sweep_is_routed_by_kind(monkeypatch):
    seen = {}

    def _fake_dispatch(engine, payload):
        seen["called"] = True
        return {"status": "ok", "sources": {}}

    monkeypatch.setattr(se, "_dispatch_connector_sync_sweep", _fake_dispatch)
    result = se._dispatch_scheduled_job(
        _FakeEngine(), {"kind": "connector_sync_sweep"}
    )
    assert seen.get("called") is True
    assert result["status"] == "ok"
