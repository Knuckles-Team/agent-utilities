"""Characterization tests for KeycloakSink.run (CX-AU-06, pre-refactor CCN 11).

Pins OBSERVED behaviour: no-client live skip, dry-run proposal shape (type
defaults to "user" when absent), the silent drop of a creation missing
``name``, the ``application`` vs default (user) client-method dispatch, realm
default/override, and live success/exception counting. No behaviour changed.

"No client" scenarios monkeypatch the module-level ``_resolve_client`` --
OBSERVED: ``keycloak_agent`` IS installed in this --all-extras venv and
``get_client()`` returns a live (unconfigured) object, so an empty ``ops``
alone does not exercise the ``client is None`` branch here.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.enrichment.writeback.core import WritebackContext
from agent_utilities.knowledge_graph.enrichment.writeback.sinks import identity as identity_mod
from agent_utilities.knowledge_graph.enrichment.writeback.sinks.identity import KeycloakSink


class _FakeClient:
    def __init__(self, *, raise_on: set[str] | None = None) -> None:
        self.users: list[tuple[str, dict]] = []
        self.clients: list[tuple[str, dict]] = []
        self._raise_on = raise_on or set()

    def create_user(self, realm: str, payload: dict) -> None:
        if payload.get("username") in self._raise_on:
            raise RuntimeError("boom")
        self.users.append((realm, payload))

    def create_client(self, realm: str, payload: dict) -> None:
        if payload.get("clientId") in self._raise_on:
            raise RuntimeError("boom")
        self.clients.append((realm, payload))


def _fields(result: Any) -> tuple:
    return (result.created, result.errors, result.skipped, result.proposals)


def _no_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(identity_mod, "_resolve_client", lambda ops, module: None)


def test_no_client_live_mode_marks_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_client(monkeypatch)
    sink = KeycloakSink()
    ops: dict[str, Any] = {"creations": [{"name": "alice"}]}
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert _fields(result) == (0, 0, 1, [])


def test_no_client_dry_run_defaults_type_to_user(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_client(monkeypatch)
    sink = KeycloakSink()
    ops: dict[str, Any] = {"creations": [{"name": "alice"}]}
    result = sink.run(WritebackContext(), ops, dry_run=True)
    assert result.proposals == [
        {"op": "create", "type": "user", "name": "alice", "realm": "master"}
    ]


def test_dry_run_application_type_and_custom_realm(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_client(monkeypatch)
    sink = KeycloakSink()
    ops: dict[str, Any] = {
        "realm": "prod",
        "creations": [{"name": "svc-a", "type": "Application"}],
    }
    result = sink.run(WritebackContext(), ops, dry_run=True)
    assert result.proposals == [
        {"op": "create", "type": "application", "name": "svc-a", "realm": "prod"}
    ]


def test_creation_missing_name_is_silently_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_client(monkeypatch)
    sink = KeycloakSink()
    ops: dict[str, Any] = {"creations": [{"type": "user"}]}
    result = sink.run(WritebackContext(), ops, dry_run=True)
    assert _fields(result) == (0, 0, 0, [])


def test_live_default_type_creates_user() -> None:
    client = _FakeClient()
    sink = KeycloakSink()
    ops: dict[str, Any] = {"client": client, "creations": [{"name": "alice"}]}
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.created, result.errors, client.users, client.clients) == (
        1,
        0,
        [("master", {"username": "alice", "enabled": True})],
        [],
    )


def test_live_application_type_creates_client() -> None:
    client = _FakeClient()
    sink = KeycloakSink()
    ops: dict[str, Any] = {
        "client": client,
        "creations": [{"name": "svc-a", "type": "application"}],
    }
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.created, result.errors, client.clients, client.users) == (
        1,
        0,
        [("master", {"clientId": "svc-a"})],
        [],
    )


def test_live_exception_increments_errors() -> None:
    client = _FakeClient(raise_on={"alice"})
    sink = KeycloakSink()
    ops: dict[str, Any] = {"client": client, "creations": [{"name": "alice"}]}
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.errors, result.created) == (1, 0)


def test_multiple_creations_mixed_outcomes() -> None:
    client = _FakeClient(raise_on={"bob"})
    sink = KeycloakSink()
    ops: dict[str, Any] = {
        "client": client,
        "creations": [
            {"name": "alice"},
            {"name": "bob"},
            {"type": "user"},  # missing name -> dropped silently
            {"name": "svc-a", "type": "application"},
        ],
    }
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.created, result.errors, result.skipped) == (2, 1, 0)
    assert client.users == [("master", {"username": "alice", "enabled": True})]
    assert client.clients == [("master", {"clientId": "svc-a"})]


def test_no_creations_key_returns_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_client(monkeypatch)
    sink = KeycloakSink()
    result = sink.run(WritebackContext(), {}, dry_run=True)
    assert _fields(result) == (0, 0, 0, [])
