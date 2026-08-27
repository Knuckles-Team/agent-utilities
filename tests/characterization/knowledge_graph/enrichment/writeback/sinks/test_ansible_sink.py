"""Characterization tests for AnsibleSink.run (CX-AU-06, pre-refactor CCN 11).

Pins OBSERVED behaviour of
``agent_utilities.knowledge_graph.enrichment.writeback.sinks.ansible.AnsibleSink.run``
exactly as it stands today, including quirks (a creation missing both
``template_id`` and ``name`` is silently dropped -- no skip/error counter moves).
No behaviour is changed by this commit.

Each test collapses its checks into a single tuple-equality assertion so the
known-bad discipline (every assertion flipped and observed to fail, see the
lane report) stays tractable per-assertion rather than per-field, without
pinning any less behaviour: every field checked below is still individually
compared, just packed into one ``assert (...) == (...)``.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.enrichment.writeback.core import WritebackContext
from agent_utilities.knowledge_graph.enrichment.writeback.sinks.ansible import AnsibleSink


class _FakeClient:
    def __init__(self, *, raise_on: set[str] | None = None) -> None:
        self.launched: list[tuple[str, dict]] = []
        self._raise_on = raise_on or set()

    def launch_job(self, template: str, extra_vars: dict) -> None:
        if template in self._raise_on:
            raise RuntimeError("boom")
        self.launched.append((template, extra_vars))


class _NoLaunchClient:
    """A client object with no callable ``launch_job`` attribute."""


def _fields(result: Any) -> tuple:
    return (result.created, result.errors, result.skipped, result.proposals)


def test_no_client_live_mode_marks_skipped() -> None:
    sink = AnsibleSink()
    ops: dict[str, Any] = {"creations": [{"template_id": "T1"}]}
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert _fields(result) == (0, 0, 1, [])


def test_no_client_dry_run_still_proposes() -> None:
    sink = AnsibleSink()
    ops: dict[str, Any] = {"creations": [{"template_id": "T1", "extra_vars": {"a": 1}}]}
    result = sink.run(WritebackContext(), ops, dry_run=True)
    assert _fields(result) == (0, 0, 0, [{"op": "launch_job", "template": "T1"}])


def test_dry_run_uses_name_fallback_when_no_template_id() -> None:
    sink = AnsibleSink()
    ops: dict[str, Any] = {"creations": [{"name": "by-name"}]}
    result = sink.run(WritebackContext(), ops, dry_run=True)
    assert result.proposals == [{"op": "launch_job", "template": "by-name"}]


def test_creation_missing_template_and_name_is_silently_skipped() -> None:
    """OBSERVED quirk: no counter moves for a creation with neither key."""
    sink = AnsibleSink()
    ops: dict[str, Any] = {"creations": [{"extra_vars": {}}]}
    result = sink.run(WritebackContext(), ops, dry_run=True)
    assert _fields(result) == (0, 0, 0, [])


def test_live_launch_success_increments_created() -> None:
    client = _FakeClient()
    sink = AnsibleSink()
    ops: dict[str, Any] = {
        "client": client,
        "creations": [{"template_id": "T1", "extra_vars": {"x": 1}}],
    }
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.created, result.errors, client.launched) == (1, 0, [("T1", {"x": 1})])


def test_live_launch_exception_increments_errors() -> None:
    client = _FakeClient(raise_on={"T1"})
    sink = AnsibleSink()
    ops: dict[str, Any] = {"client": client, "creations": [{"template_id": "T1"}]}
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.errors, result.created) == (1, 0)


def test_live_client_without_launch_job_increments_errors() -> None:
    sink = AnsibleSink()
    ops: dict[str, Any] = {
        "client": _NoLaunchClient(),
        "creations": [{"template_id": "T1"}],
    }
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.errors, result.created) == (1, 0)


def test_multiple_creations_mixed_outcomes() -> None:
    client = _FakeClient(raise_on={"BAD"})
    sink = AnsibleSink()
    ops: dict[str, Any] = {
        "client": client,
        "creations": [
            {"template_id": "GOOD1"},
            {"template_id": "BAD"},
            {},  # dropped silently
            {"template_id": "GOOD2"},
        ],
    }
    result = sink.run(WritebackContext(), ops, dry_run=False)
    assert (result.created, result.errors, result.skipped, client.launched) == (
        2,
        1,
        0,
        [("GOOD1", {}), ("GOOD2", {})],
    )


def test_no_creations_key_returns_empty_result() -> None:
    sink = AnsibleSink()
    result = sink.run(WritebackContext(), {}, dry_run=True)
    assert _fields(result) == (0, 0, 0, [])
