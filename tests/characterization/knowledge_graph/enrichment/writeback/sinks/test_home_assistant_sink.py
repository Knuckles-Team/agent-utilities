"""Characterization tests for HomeAssistantSink.run (CX-AU-06, pre-refactor CCN 11).

Pins OBSERVED behaviour exactly as it stands today, including the difference
from AnsibleSink: a creation missing ``domain``/``service`` increments
``skipped`` here (Ansible drops it silently). No behaviour is changed.

The "no client" scenarios monkeypatch ``HomeAssistantSink._client`` directly
rather than relying on ``home_assistant_agent`` being absent from the
environment -- OBSERVED: in this repo's --all-extras venv the package IS
installed and ``get_client()`` returns a live (unconfigured) ``Api`` object,
so ``ops={}`` alone does not exercise the ``client is None`` branch here.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.enrichment.writeback.core import WritebackContext
from agent_utilities.knowledge_graph.enrichment.writeback.sinks.home_assistant import (
    HomeAssistantSink,
)


class _FakeClient:
    def __init__(self, *, raise_on: set[tuple[str, str]] | None = None) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self._raise_on = raise_on or set()

    def call_service(self, domain: str, service: str, data: dict) -> None:
        if (domain, service) in self._raise_on:
            raise RuntimeError("boom")
        self.calls.append((domain, service, data))


class _NoCallClient:
    """A client with no callable ``call_service`` attribute."""


def _fields(result: Any) -> tuple:
    return (result.created, result.errors, result.skipped, result.proposals)


def _sink_with_no_client(monkeypatch: pytest.MonkeyPatch) -> HomeAssistantSink:
    sink = HomeAssistantSink()
    monkeypatch.setattr(sink, "_client", lambda ops: None)
    return sink


def test_no_client_live_mode_marks_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = _sink_with_no_client(monkeypatch)
    ops: dict[str, Any] = {"creations": [{"domain": "light", "service": "turn_on"}]}
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert _fields(result) == (0, 0, 1, [])


def test_no_client_dry_run_still_proposes(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = _sink_with_no_client(monkeypatch)
    ops: dict[str, Any] = {"creations": [{"domain": "light", "service": "turn_on"}]}
    result = sink.run(WritebackContext(), ops, dry_run=True)
    assert _fields(result) == (0, 0, 0, [{"op": "call_service", "service": "light.turn_on"}])


def test_creation_missing_domain_or_service_increments_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OBSERVED: unlike AnsibleSink, an incomplete creation here counts as skipped."""
    sink = _sink_with_no_client(monkeypatch)
    ops: dict[str, Any] = {
        "creations": [{"domain": "light"}, {"service": "turn_on"}, {}]
    }
    result = sink.run(WritebackContext(), ops, dry_run=True)
    assert _fields(result) == (0, 0, 3, [])


def test_live_call_success_increments_created() -> None:
    client = _FakeClient()
    sink = HomeAssistantSink()
    ops: dict[str, Any] = {
        "client": client,
        "creations": [{"domain": "light", "service": "turn_on", "data": {"brightness": 5}}],
    }
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.created, result.errors, client.calls) == (
        1,
        0,
        [("light", "turn_on", {"brightness": 5})],
    )


def test_live_call_exception_increments_errors() -> None:
    client = _FakeClient(raise_on={("light", "turn_on")})
    sink = HomeAssistantSink()
    ops: dict[str, Any] = {
        "client": client,
        "creations": [{"domain": "light", "service": "turn_on"}],
    }
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.errors, result.created) == (1, 0)


def test_live_client_without_call_service_increments_errors() -> None:
    sink = HomeAssistantSink()
    ops: dict[str, Any] = {
        "client": _NoCallClient(),
        "creations": [{"domain": "light", "service": "turn_on"}],
    }
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.errors, result.created) == (1, 0)


def test_multiple_creations_mixed_outcomes() -> None:
    client = _FakeClient(raise_on={("switch", "turn_off")})
    sink = HomeAssistantSink()
    ops: dict[str, Any] = {
        "client": client,
        "creations": [
            {"domain": "light", "service": "turn_on"},
            {"domain": "switch", "service": "turn_off"},
            {"domain": "light"},  # missing service -> skipped
            {"domain": "fan", "service": "set_speed", "data": {"speed": 2}},
        ],
    }
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.created, result.errors, result.skipped, client.calls) == (
        2,
        1,
        1,
        [("light", "turn_on", {}), ("fan", "set_speed", {"speed": 2})],
    )


def test_no_creations_key_returns_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = _sink_with_no_client(monkeypatch)
    result = sink.run(WritebackContext(), {}, dry_run=True)
    assert _fields(result) == (0, 0, 0, [])
