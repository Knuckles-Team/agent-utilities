"""Characterization tests for OktaSink.run (CX-AU-06, pre-refactor CCN 19).

Pins OBSERVED behaviour across the three independent ``creations`` /
``inferences`` / ``retirements`` loops: no-client live skip, a creation
missing ``name`` silently dropped (no skip counter, unlike the other two
loops), dry-run proposal shapes, live success/exception counting per loop,
and that all three loops still run even when an earlier one produced errors
(no short-circuit). No behaviour changed.

``ctx.resolver`` is monkeypatched directly rather than exercising the real KG
query path (:func:`resolve_external_id`) -- that function itself is not part
of this lane's target surface; only ``OktaSink.run``'s own branching is.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.enrichment.writeback.core import WritebackContext
from agent_utilities.knowledge_graph.enrichment.writeback.sinks.identity import OktaSink


class _FakeClient:
    def __init__(self, *, raise_on: set[str] | None = None) -> None:
        self.created_users: list[dict] = []
        self.assigned: list[tuple[str, str]] = []
        self.deactivated: list[str] = []
        self._raise_on = raise_on or set()

    def create_user(self, payload: dict) -> None:
        if payload["profile"]["login"] in self._raise_on:
            raise RuntimeError("boom")
        self.created_users.append(payload)

    def assign_user_to_app(self, app: str, user: str) -> None:
        if f"{user}->{app}" in self._raise_on:
            raise RuntimeError("boom")
        self.assigned.append((app, user))

    def deactivate_user(self, uid: str) -> None:
        if uid in self._raise_on:
            raise RuntimeError("boom")
        self.deactivated.append(uid)


def _fields(result: Any) -> tuple:
    return (
        result.created,
        result.relations_written,
        result.retired,
        result.errors,
        result.skipped,
        result.proposals,
    )


def _ctx_with_resolver(resolve_map: dict[str, str]) -> WritebackContext:
    ctx = WritebackContext()
    ctx.resolver = lambda domain: (lambda node_id: resolve_map.get(node_id))  # type: ignore[method-assign]
    return ctx


def _no_client(monkeypatch: pytest.MonkeyPatch) -> OktaSink:
    sink = OktaSink()
    import agent_utilities.knowledge_graph.enrichment.writeback.sinks.identity as identity_mod

    monkeypatch.setattr(identity_mod, "_resolve_client", lambda ops, module: None)
    return sink


def test_no_client_live_mode_marks_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = _no_client(monkeypatch)
    ops: dict[str, Any] = {"creations": [{"name": "alice"}]}
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert _fields(result) == (0, 0, 0, 0, 1, [])


def test_creation_missing_name_is_silently_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = _no_client(monkeypatch)
    ops: dict[str, Any] = {"creations": [{}]}
    result = sink.run(WritebackContext(), ops, dry_run=True)
    assert _fields(result) == (0, 0, 0, 0, 0, [])


def test_dry_run_creation_proposal() -> None:
    sink = OktaSink()
    ops: dict[str, Any] = {"creations": [{"name": "alice"}]}
    result = sink.run(WritebackContext(), ops, dry_run=True)
    assert result.proposals == [{"op": "create_user", "name": "alice"}]


def test_live_creation_success() -> None:
    client = _FakeClient()
    sink = OktaSink()
    ops: dict[str, Any] = {"client": client, "creations": [{"name": "alice"}]}
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.created, result.errors, client.created_users) == (
        1,
        0,
        [{"profile": {"login": "alice", "email": "alice"}}],
    )


def test_live_creation_exception_increments_errors() -> None:
    client = _FakeClient(raise_on={"alice"})
    sink = OktaSink()
    ops: dict[str, Any] = {"client": client, "creations": [{"name": "alice"}]}
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.errors, result.created) == (1, 0)


def test_inference_unresolved_source_or_target_increments_skipped() -> None:
    sink = OktaSink()
    ctx = _ctx_with_resolver({"u1": "alice"})  # "app1" deliberately unresolved
    ops: dict[str, Any] = {"inferences": [{"source": "u1", "target": "app1"}]}
    result = sink.run(ctx, ops, dry_run=True)
    assert _fields(result) == (0, 0, 0, 0, 1, [])


def test_inference_dry_run_proposal() -> None:
    sink = OktaSink()
    ctx = _ctx_with_resolver({"u1": "alice", "a1": "MyApp"})
    ops: dict[str, Any] = {"inferences": [{"source": "u1", "target": "a1"}]}
    result = sink.run(ctx, ops, dry_run=True)
    assert result.proposals == [
        {"op": "assign_user_to_app", "user": "alice", "app": "MyApp"}
    ]


def test_inference_live_success() -> None:
    client = _FakeClient()
    sink = OktaSink()
    ctx = _ctx_with_resolver({"u1": "alice", "a1": "MyApp"})
    ops: dict[str, Any] = {"client": client, "inferences": [{"source": "u1", "target": "a1"}]}
    result = sink.run(ctx, ops, dry_run=False)
    assert (result.relations_written, result.errors, client.assigned) == (
        1,
        0,
        [("MyApp", "alice")],
    )


def test_inference_live_exception_increments_errors() -> None:
    client = _FakeClient(raise_on={"alice->MyApp"})
    sink = OktaSink()
    ctx = _ctx_with_resolver({"u1": "alice", "a1": "MyApp"})
    ops: dict[str, Any] = {"client": client, "inferences": [{"source": "u1", "target": "a1"}]}
    result = sink.run(ctx, ops, dry_run=False)
    assert (result.errors, result.relations_written) == (1, 0)


def test_retirement_unresolved_node_increments_skipped() -> None:
    sink = OktaSink()
    ctx = _ctx_with_resolver({})
    ops: dict[str, Any] = {"retirements": [{"node": "u1"}]}
    result = sink.run(ctx, ops, dry_run=True)
    assert _fields(result) == (0, 0, 0, 0, 1, [])


def test_retirement_dry_run_proposal() -> None:
    sink = OktaSink()
    ctx = _ctx_with_resolver({"u1": "alice"})
    ops: dict[str, Any] = {"retirements": [{"node": "u1"}]}
    result = sink.run(ctx, ops, dry_run=True)
    assert result.proposals == [{"op": "deactivate_user", "user": "alice"}]


def test_retirement_live_success_and_exception() -> None:
    client = _FakeClient(raise_on={"bob"})
    sink = OktaSink()
    ctx = _ctx_with_resolver({"u1": "alice", "u2": "bob"})
    ops: dict[str, Any] = {
        "client": client,
        "retirements": [{"node": "u1"}, {"node": "u2"}],
    }
    result = sink.run(ctx, ops, dry_run=False)
    assert (result.retired, result.errors, client.deactivated) == (1, 1, ["alice"])


def test_all_three_loops_run_independently_in_one_call() -> None:
    """OBSERVED: no short-circuit -- an error in one loop does not block the next."""
    client = _FakeClient(raise_on={"alice"})
    sink = OktaSink()
    ctx = _ctx_with_resolver({"u1": "carol"})
    ops: dict[str, Any] = {
        "client": client,
        "creations": [{"name": "alice"}],  # raises
        "inferences": [{"source": "u1", "target": "missing"}],  # skipped (target unresolved)
        "retirements": [{"node": "u1"}],  # succeeds
    }
    result = sink.run(ctx, ops, dry_run=False)
    assert (result.created, result.errors, result.skipped, result.retired) == (0, 1, 1, 1)
    assert client.deactivated == ["carol"]


def test_no_ops_keys_returns_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = _no_client(monkeypatch)
    result = sink.run(WritebackContext(), {}, dry_run=True)
    assert _fields(result) == (0, 0, 0, 0, 0, [])
