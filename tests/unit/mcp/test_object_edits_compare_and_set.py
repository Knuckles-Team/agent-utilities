"""Optimistic-concurrency on the object-mutation layer (CONCEPT:AU-KG.ontology.optimistic-concurrency-object-property).

``object_edits action=record`` for a ``property_set`` accepts an optional
``expect`` precondition. When non-empty, the set is applied through the engine's
atomic ``backend.compare_and_set_node_fields(object_id, conditions=expect,
updates=properties)`` and the ledger edit is recorded ONLY if it wins; a lost
race surfaces ``applied=False`` and records nothing. When ``expect`` is empty the
path is unchanged — the normal unconditional ``ledger.record`` runs and the
backend CAS is never touched.

These call the tool via ``kg_server._execute_tool`` (the same dispatch the MCP
and REST surfaces use, which resolves ``Field`` defaults) with the engine backend
and ontology ledger mocked — no live engine, no daemon.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.mcp import kg_server


class _CASBackend:
    """Records the compare_and_set call and returns a configurable result."""

    def __init__(self, result: bool):
        self.result = result
        self.calls: list[tuple[str, dict, dict]] = []

    def compare_and_set_node_fields(self, node_id, conditions, updates):
        self.calls.append((node_id, conditions, updates))
        return self.result


class _CASEngine:
    def __init__(self, result: bool):
        self.backend = _CASBackend(result)


class _RecordingLedger:
    """A ledger whose ``record`` captures the edit and never touches a backend."""

    def __init__(self):
        self.recorded: list[object] = []

    def record(self, edit):
        edit.persisted = True
        self.recorded.append(edit)
        return edit


class _OntologyStub:
    def __init__(self, ledger):
        self.edits = ledger


@pytest.fixture
def _ensure_registered():
    kg_server.ensure_tools_registered()


def _patch(monkeypatch, *, cas_result: bool, ledger: _RecordingLedger):
    engine = _CASEngine(cas_result)
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    monkeypatch.setattr(kg_server, "_ontology_system", lambda: _OntologyStub(ledger))
    return engine


async def test_conditional_set_calls_backend_and_records_on_win(
    monkeypatch, _ensure_registered
):
    # A non-empty ``expect`` routes the property_set through the engine CAS with
    # the EXACT object_id/conditions/updates; on a win it records the edit and
    # surfaces applied=True (CONCEPT:AU-KG.ontology.optimistic-concurrency-object-property).
    ledger = _RecordingLedger()
    engine = _patch(monkeypatch, cas_result=True, ledger=ledger)

    out = await kg_server._execute_tool(
        "object_edits",
        action="record",
        object_id="paper:001",
        edit_type="property_set",
        properties_json='{"status": "claimed", "owner": "agent-7"}',
        expect={"status": "pending"},
    )
    payload = json.loads(out)

    assert payload["applied"] is True
    # The CAS hit the backend with object_id as the node id and the right maps.
    assert engine.backend.calls == [
        (
            "paper:001",
            {"status": "pending"},
            {"status": "claimed", "owner": "agent-7"},
        )
    ]
    # The winning edit was recorded in the ledger for audit/history.
    assert len(ledger.recorded) == 1
    assert ledger.recorded[0].object_id == "paper:001"
    assert ledger.recorded[0].after == {"status": "claimed", "owner": "agent-7"}


async def test_conditional_set_lost_race_is_surfaced_not_swallowed(
    monkeypatch, _ensure_registered
):
    # A failed precondition (backend returns False) must surface applied=False
    # AND record nothing — no misleading audit edit for a write that did not land.
    ledger = _RecordingLedger()
    engine = _patch(monkeypatch, cas_result=False, ledger=ledger)

    out = await kg_server._execute_tool(
        "object_edits",
        action="record",
        object_id="paper:001",
        edit_type="property_set",
        properties_json='{"status": "claimed"}',
        expect={"status": "pending"},
    )
    payload = json.loads(out)

    assert payload["applied"] is False
    assert payload["object_id"] == "paper:001"
    assert engine.backend.calls == [
        ("paper:001", {"status": "pending"}, {"status": "claimed"})
    ]
    # Nothing recorded: the conditional set lost the race.
    assert ledger.recorded == []


async def test_conditional_set_accepts_json_string_expect(
    monkeypatch, _ensure_registered
):
    # Some MCP clients send dict params as JSON strings; expect is coerced.
    ledger = _RecordingLedger()
    engine = _patch(monkeypatch, cas_result=True, ledger=ledger)

    out = await kg_server._execute_tool(
        "object_edits",
        action="record",
        object_id="paper:001",
        edit_type="property_set",
        properties_json='{"score": 9}',
        expect='{"score": 1}',
    )
    payload = json.loads(out)

    assert payload["applied"] is True
    assert engine.backend.calls == [("paper:001", {"score": 1}, {"score": 9})]


async def test_no_precondition_is_unchanged_normal_record(
    monkeypatch, _ensure_registered
):
    # Empty expect (the default) must NOT touch the backend CAS — it records the
    # edit through the normal ledger path exactly as before.
    ledger = _RecordingLedger()
    engine = _patch(monkeypatch, cas_result=True, ledger=ledger)

    out = await kg_server._execute_tool(
        "object_edits",
        action="record",
        object_id="paper:001",
        edit_type="property_set",
        properties_json='{"status": "claimed"}',
    )
    payload = json.loads(out)

    # The normal record path returns the edit model_dump (no 'applied' key) ...
    assert "applied" not in payload
    assert payload["object_id"] == "paper:001"
    assert payload["after"] == {"status": "claimed"}
    # ... and crucially the backend CAS was never invoked.
    assert engine.backend.calls == []
    assert len(ledger.recorded) == 1


async def test_expect_only_applies_to_property_set(monkeypatch, _ensure_registered):
    # An ``expect`` on a non-property_set edit (e.g. link_add) is ignored — the
    # CAS path is property-set-specific; the normal record runs.
    ledger = _RecordingLedger()
    engine = _patch(monkeypatch, cas_result=True, ledger=ledger)

    out = await kg_server._execute_tool(
        "object_edits",
        action="record",
        object_id="paper:001",
        edit_type="link_add",
        link_target="concept:KG-2.142",
        link_label="enhances",
        expect={"status": "pending"},
    )
    payload = json.loads(out)

    assert "applied" not in payload
    assert engine.backend.calls == []
    assert len(ledger.recorded) == 1
    assert ledger.recorded[0].link_target == "concept:KG-2.142"
