"""Bitemporal ``as_of`` on writeback loops (CONCEPT:AU-KG.temporal.bi-temporal-memory-layers).

Read paths (``engine_query.py``, ``hybrid_retriever.py``, ``context_compiler.py``) have long
accepted ``as_of`` and filtered via ``bitemporal.filter_as_of``; these tests cover the write
side closing the loop: :class:`WritebackContext.stamp_valid_time`/``stamp_external_id`` stamp
the SAME ``storage_time``/``event_time``/``valid_from``/``valid_to`` quadruple, and
``run_writeback``/``push_inventory``/``push_findings`` thread an explicit ``as_of`` through to
that stamp — defaulting to ``None`` ("now") so every pre-existing caller is unaffected.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.writeback.core import (
    WritebackContext,
    WritebackResult,
    register_sink,
    run_writeback,
)
from agent_utilities.knowledge_graph.enrichment.writeback.findings import push_findings
from agent_utilities.knowledge_graph.enrichment.writeback.inventory import (
    push_inventory,
)


class _FakeEngine:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def add_node(self, node_id, node_type, properties=None):
        self.calls.append((node_id, node_type, dict(properties or {})))


# ── WritebackContext.stamp_valid_time / stamp_external_id ──────────────────


def test_stamp_valid_time_defaults_to_now_when_as_of_is_none():
    ctx = WritebackContext()
    props = ctx.stamp_valid_time({})
    assert props["storage_time"]
    assert props["event_time"] == props["storage_time"]
    assert props["valid_from"] == props["event_time"]
    assert props["valid_to"] is None


def test_stamp_valid_time_uses_explicit_as_of():
    ctx = WritebackContext(as_of="2026-01-01T00:00:00+00:00")
    props = ctx.stamp_valid_time({})
    assert props["event_time"] == "2026-01-01T00:00:00+00:00"
    assert props["valid_from"] == "2026-01-01T00:00:00+00:00"


def test_stamp_external_id_carries_bitemporal_stamp_onto_the_node():
    engine = _FakeEngine()
    ctx = WritebackContext(engine=engine, as_of="2026-02-02T00:00:00+00:00")
    ok = ctx.stamp_external_id("host:r510", "servicenow", "SYS-1")
    assert ok is True
    _node_id, _label, props = engine.calls[0]
    assert props["servicenow_ci_id"] == "SYS-1"
    assert props["externalToolId"] == "SYS-1"
    assert props["valid_from"] == "2026-02-02T00:00:00+00:00"
    assert props["event_time"] == "2026-02-02T00:00:00+00:00"
    assert props["valid_to"] is None
    assert props["storage_time"]


def test_stamp_external_id_defaults_as_of_to_now_when_context_has_none():
    engine = _FakeEngine()
    ctx = WritebackContext(engine=engine)  # as_of=None -> "now"
    ctx.stamp_external_id("host:r710", "servicenow", "SYS-2")
    _node_id, _label, props = engine.calls[0]
    assert props["valid_from"] == props["storage_time"]
    assert props["valid_to"] is None


# ── run_writeback threading ─────────────────────────────────────────────────


class _CapturingSink:
    domain = "faketarget"
    enable_flag = "FAKETARGET_ENABLE_WRITE"

    def __init__(self) -> None:
        self.seen_as_of: list[str | None] = []

    def run(self, ctx: WritebackContext, ops, *, dry_run: bool) -> WritebackResult:
        self.seen_as_of.append(ctx.as_of)
        return WritebackResult(
            target=self.domain, created=len(ops.get("creations", []))
        )


def test_run_writeback_threads_as_of_onto_the_sink_context():
    sink = _CapturingSink()
    register_sink(sink)
    out = run_writeback(
        "faketarget", dry_run=True, as_of="2026-03-03T00:00:00+00:00", creations=[]
    )
    assert sink.seen_as_of == ["2026-03-03T00:00:00+00:00"]
    assert out["as_of"] == "2026-03-03T00:00:00+00:00"


def test_run_writeback_as_of_defaults_to_none_unaffecting_existing_callers():
    sink = _CapturingSink()
    register_sink(sink)
    out = run_writeback("faketarget", dry_run=True, creations=[])
    assert sink.seen_as_of == [None]
    assert out["as_of"] is None


# ── push_inventory / push_findings threading ────────────────────────────────


def test_push_inventory_threads_as_of_through_to_run_writeback(monkeypatch):
    import agent_utilities.knowledge_graph.enrichment.writeback.inventory as inv_mod

    seen: dict[str, object] = {}

    def fake_run_writeback(target, **kwargs):
        seen.update(kwargs)
        return {"created": 0}

    monkeypatch.setattr(inv_mod, "run_writeback", fake_run_writeback)
    monkeypatch.setattr(inv_mod, "collect_inventory_creations", lambda *a, **k: [])

    push_inventory("servicenow", dry_run=True, as_of="2026-04-04T00:00:00+00:00")
    assert seen["as_of"] == "2026-04-04T00:00:00+00:00"


def test_push_findings_threads_as_of_through_to_run_writeback(monkeypatch):
    import agent_utilities.knowledge_graph.enrichment.writeback.findings as find_mod

    seen: dict[str, object] = {}

    def fake_run_writeback(target, **kwargs):
        seen.update(kwargs)
        return {"created": 0}

    monkeypatch.setattr(find_mod, "run_writeback", fake_run_writeback)
    monkeypatch.setattr(find_mod, "collect_risk_findings", lambda *a, **k: [])

    push_findings("gitlab", dry_run=True, as_of="2026-05-05T00:00:00+00:00")
    assert seen["as_of"] == "2026-05-05T00:00:00+00:00"
