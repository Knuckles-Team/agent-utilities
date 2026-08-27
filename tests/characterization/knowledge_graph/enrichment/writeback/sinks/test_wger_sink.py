"""Characterization tests for WgerSink.run (CX-AU-06, pre-refactor CCN 15).

Pins OBSERVED behaviour: no-client live skip, the two top-level ``creations``
type groups (bodymeasurement/weightentry/weight -- itself split by ``kind`` --
and workoutsession/session), the unrecognised-type unconditional skip, and
live success/exception counting. No behaviour changed.

"No client" scenarios monkeypatch ``WgerSink._client`` directly for
environment independence -- OBSERVED: ``wger_agent`` IS installed in this
--all-extras venv and ``get_client()`` returns a live (unconfigured) object.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.enrichment.writeback.core import WritebackContext
from agent_utilities.knowledge_graph.enrichment.writeback.sinks.wger import WgerSink


class _FakeClient:
    def __init__(self, *, raise_on: set[str] | None = None) -> None:
        self.weight_entries: list[tuple] = []
        self.measurements: list[tuple] = []
        self.sessions: list[tuple] = []
        self._raise_on = raise_on or set()

    def create_weight_entry(self, date, weight) -> None:
        if "weight" in self._raise_on:
            raise RuntimeError("boom")
        self.weight_entries.append((date, weight))

    def create_measurement(self, category, date, value) -> None:
        if "measurement" in self._raise_on:
            raise RuntimeError("boom")
        self.measurements.append((category, date, value))

    def create_workout_session(self, routine, date, *, impression, notes) -> None:
        if "session" in self._raise_on:
            raise RuntimeError("boom")
        self.sessions.append((routine, date, impression, notes))


def _fields(result: Any) -> tuple:
    return (result.created, result.errors, result.skipped, result.proposals)


def _sink_with_no_client(monkeypatch: pytest.MonkeyPatch) -> WgerSink:
    sink = WgerSink()
    monkeypatch.setattr(sink, "_client", lambda ops: None)
    return sink


def test_no_client_live_mode_marks_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = _sink_with_no_client(monkeypatch)
    ops: dict[str, Any] = {"creations": [{"type": "weight", "date": "d", "value": 80}]}
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert _fields(result) == (0, 0, 1, [])


def test_dry_run_weight_defaults_kind_to_weight() -> None:
    sink = WgerSink()
    ops: dict[str, Any] = {
        "creations": [{"type": "weightentry", "date": "2026-01-01", "value": 82.5}]
    }
    result = sink.run(WritebackContext(), ops, dry_run=True)
    assert result.proposals == [
        {"op": "create_weight_entry", "date": "2026-01-01", "weight": 82.5}
    ]


def test_dry_run_weight_prefers_value_over_weight_key() -> None:
    sink = WgerSink()
    ops: dict[str, Any] = {
        "creations": [{"type": "weight", "date": "d", "value": 1, "weight": 2}]
    }
    result = sink.run(WritebackContext(), ops, dry_run=True)
    assert result.proposals[0]["weight"] == 1


def test_dry_run_bodymeasurement_kind_routes_to_measurement() -> None:
    sink = WgerSink()
    ops: dict[str, Any] = {
        "creations": [
            {
                "type": "bodymeasurement",
                "kind": "waist",
                "category": "waist",
                "date": "2026-01-01",
                "value": 90,
            }
        ]
    }
    result = sink.run(WritebackContext(), ops, dry_run=True)
    assert result.proposals == [
        {
            "op": "create_measurement",
            "category": "waist",
            "date": "2026-01-01",
            "value": 90,
        }
    ]


def test_dry_run_workout_session_defaults() -> None:
    sink = WgerSink()
    ops: dict[str, Any] = {
        "creations": [{"type": "workoutsession", "routine": "R1", "date": "2026-01-01"}]
    }
    result = sink.run(WritebackContext(), ops, dry_run=True)
    assert result.proposals == [
        {
            "op": "create_workout_session",
            "routine": "R1",
            "date": "2026-01-01",
            "impression": "3",
            "notes": "",
        }
    ]


def test_unrecognised_type_skipped_even_in_dry_run() -> None:
    sink = WgerSink()
    ops: dict[str, Any] = {"creations": [{"type": "exercise"}]}
    result = sink.run(WritebackContext(), ops, dry_run=True)
    assert _fields(result) == (0, 0, 1, [])


def test_live_weight_entry_success() -> None:
    client = _FakeClient()
    sink = WgerSink()
    ops: dict[str, Any] = {
        "client": client,
        "creations": [{"type": "weight", "date": "2026-01-01", "value": 80}],
    }
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.created, result.errors, client.weight_entries) == (
        1,
        0,
        [("2026-01-01", 80)],
    )


def test_live_measurement_exception_increments_errors() -> None:
    client = _FakeClient(raise_on={"measurement"})
    sink = WgerSink()
    ops: dict[str, Any] = {
        "client": client,
        "creations": [{"type": "bodymeasurement", "kind": "waist", "category": "waist"}],
    }
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.errors, result.created) == (1, 0)


def test_live_session_success() -> None:
    client = _FakeClient()
    sink = WgerSink()
    ops: dict[str, Any] = {
        "client": client,
        "creations": [
            {
                "type": "session",
                "routine": "R2",
                "date": "2026-01-02",
                "impression": "5",
                "notes": "great",
            }
        ],
    }
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.created, result.errors, client.sessions) == (
        1,
        0,
        [("R2", "2026-01-02", "5", "great")],
    )


def test_multiple_creations_mixed_outcomes() -> None:
    client = _FakeClient(raise_on={"session"})
    sink = WgerSink()
    ops: dict[str, Any] = {
        "client": client,
        "creations": [
            {"type": "weight", "date": "d1", "value": 1},
            {"type": "session", "routine": "R", "date": "d"},
            {"type": "unknown"},
            {"type": "bodymeasurement", "kind": "waist", "category": "c", "date": "d", "value": 1},
        ],
    }
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.created, result.errors, result.skipped) == (2, 1, 1)
    assert client.weight_entries == [("d1", 1)]
    assert client.measurements == [("c", "d", 1)]


def test_no_creations_key_returns_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = _sink_with_no_client(monkeypatch)
    result = sink.run(WritebackContext(), {}, dry_run=True)
    assert _fields(result) == (0, 0, 0, [])
