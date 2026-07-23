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
    ok = ctx.stamp_external_id("host:storage-node-a", "servicenow", "SYS-1")
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
    ctx.stamp_external_id("host:compute-node-b", "servicenow", "SYS-2")
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


# ── X5 write-path closure: as_of on the outbound proposal/payload (W3.4) ────


class _ProposingSink:
    """A minimal sink whose ``run`` always returns one proposal — isolates
    :func:`~agent_utilities.knowledge_graph.enrichment.writeback.core._stamp_proposals_as_of`
    from any real target system's payload shape."""

    domain = "fakeproposing"
    enable_flag = "FAKEPROPOSING_ENABLE_WRITE"

    def run(self, ctx: WritebackContext, ops, *, dry_run: bool) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        result.proposals.append({"op": "create", "name": "widget"})
        return result


def test_run_writeback_stamps_as_of_onto_every_proposal():
    register_sink(_ProposingSink())
    out = run_writeback(
        "fakeproposing", dry_run=True, as_of="2026-06-03T00:00:00+00:00", creations=[]
    )
    assert out["proposals"][0]["as_of"] == "2026-06-03T00:00:00+00:00"


def test_run_writeback_no_as_of_leaves_proposals_unstamped_byte_identical():
    register_sink(_ProposingSink())
    out = run_writeback("fakeproposing", dry_run=True, creations=[])
    assert "as_of" not in out["proposals"][0]


class _ServiceNowClient:
    def __init__(self) -> None:
        self.patched: list[dict] = []

    def patch_table_record(self, *, table, table_record_sys_id, data):
        self.patched.append({"table": table, "sys_id": table_record_sys_id, **data})


def test_servicenow_work_notes_dry_run_proposal_carries_as_of():
    out = run_writeback(
        "servicenow",
        dry_run=True,
        as_of="2026-06-01T00:00:00+00:00",
        work_notes=[{"table": "u_trm_request", "sys_id": "sys-1", "note": "hello"}],
    )
    assert out["status"] == "completed"
    proposal = out["proposals"][0]
    # Core-level audit-trail stamp (every sink, for free).
    assert proposal["as_of"] == "2026-06-01T00:00:00+00:00"
    # Sink-level: embedded directly into the ServiceNow-native audit field text.
    assert "2026-06-01T00:00:00+00:00" in proposal["note"]


def test_servicenow_work_notes_live_payload_carries_as_of(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.writeback.core.setting",
        lambda key, default=None, cast=None: True,
    )
    client = _ServiceNowClient()
    out = run_writeback(
        "servicenow",
        dry_run=False,
        as_of="2026-06-01T00:00:00+00:00",
        client=client,
        work_notes=[{"table": "u_trm_request", "sys_id": "sys-1", "note": "hello"}],
    )
    assert out["status"] == "completed"
    assert client.patched[0]["work_notes"] == (
        "hello\n\n(KG state as of 2026-06-01T00:00:00+00:00)"
    )


def test_servicenow_work_notes_no_as_of_is_byte_identical():
    client = _ServiceNowClient()
    out = run_writeback(
        "servicenow",
        dry_run=True,
        client=client,
        work_notes=[{"table": "u_trm_request", "sys_id": "sys-1", "note": "hello"}],
    )
    assert out["proposals"][0]["note"] == "hello"
    assert "as_of" not in out["proposals"][0]


class _EgeriaClient:
    def __init__(self) -> None:
        self.created: list[dict] = []

    def create_asset(self, asset_type, qn, name, *, description, additional_properties):
        self.created.append(
            {
                "asset_type": asset_type,
                "qn": qn,
                "name": name,
                "description": description,
                "additional_properties": additional_properties,
            }
        )
        return {"guid": "egeria-guid-1"}


def test_egeria_create_asset_live_payload_carries_as_of(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.writeback.core.setting",
        lambda key, default=None, cast=None: True,
    )
    client = _EgeriaClient()
    out = run_writeback(
        "egeria",
        dry_run=False,
        as_of="2026-06-02T00:00:00+00:00",
        client=client,
        creations=[{"type": "Application", "name": "Ledger"}],
    )
    assert out["status"] == "completed"
    assert (
        client.created[0]["additional_properties"]["as_of"]
        == "2026-06-02T00:00:00+00:00"
    )
    assert client.created[0]["additional_properties"]["source"] == "graph-os"


def test_egeria_create_asset_no_as_of_is_byte_identical(monkeypatch):
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.writeback.core.setting",
        lambda key, default=None, cast=None: True,
    )
    client = _EgeriaClient()
    run_writeback(
        "egeria",
        dry_run=False,
        client=client,
        creations=[{"type": "Application", "name": "Ledger"}],
    )
    assert client.created[0]["additional_properties"] == {"source": "graph-os"}
