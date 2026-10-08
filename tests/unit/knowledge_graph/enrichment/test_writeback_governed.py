"""``run_writeback`` runs sinks through the SDK contract (AU boundary spec).

Every sink call becomes one canonical SDK ``SourceChangeSet`` and runs through
``DurableWritableConnector``. These tests use fake sinks and a temporary ledger
root; no live system-of-record is contacted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from agent_connector_sdk.writeback.errors import (
    AuthorizationDeniedError,
    SourceVersionConflictError,
)

from agent_utilities.knowledge_graph.enrichment.writeback import core
from agent_utilities.knowledge_graph.enrichment.writeback.governed import (
    PATCH_FIELD,
    SinkGrant,
    build_change_set,
    execute_governed,
    ledger_root,
)

_DOMAIN = "governedfake"


class _FakeSink:
    domain = _DOMAIN
    enable_flag = "GOVERNEDFAKE_ENABLE_WRITE"

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[bool] = []
        self.fail = fail

    def run(self, ctx, ops, *, dry_run: bool) -> core.WritebackResult:
        self.calls.append(dry_run)
        if self.fail and not dry_run:
            raise RuntimeError("sink exploded mid-batch")
        result = core.WritebackResult(target=self.domain)
        if dry_run:
            result.proposals.append({"op": "create", "items": ops.get("creations")})
        else:
            result.created = len(ops.get("creations") or [])
        return result


@pytest.fixture
def ledger(monkeypatch, tmp_path) -> Path:
    root = tmp_path / "ledger"
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.writeback.governed.ledger_root",
        lambda: root,
    )
    return root


@pytest.fixture
def register(monkeypatch):
    registered: list[_FakeSink] = []

    def _register(sink: _FakeSink, *, enabled: bool) -> _FakeSink:
        core.register_sink(sink)
        registered.append(sink)
        monkeypatch.setattr(core, "setting", lambda k, d=None, cast=None: enabled)
        return sink

    yield _register
    core._SINKS.pop(_DOMAIN, None)


def _grant(**overrides) -> SinkGrant:
    values = {
        "target": _DOMAIN,
        "enable_flag": "GOVERNEDFAKE_ENABLE_WRITE",
        "write_enabled": True,
    }
    values.update(overrides)
    return SinkGrant(**values)


def _only_change_set_dir(root: Path) -> Path:
    dirs = [p for p in root.iterdir() if p.is_dir()]
    assert len(dirs) == 1
    return dirs[0]


def test_change_set_is_canonical_and_drops_private_ops():
    ops = {"creations": [{"type": "CI", "name": "a"}], "_approved": True}
    ops["client"] = object()
    change_set = build_change_set(_grant(), ops)
    assert change_set.change_set_digest == change_set.canonical_digest()
    assert change_set.field_scope == [PATCH_FIELD]
    assert change_set.desired_patch == {
        PATCH_FIELD: {"creations": [{"name": "a", "type": "CI"}]}
    }
    assert change_set.connector_id == f"agent-utilities.writeback.{_DOMAIN}"
    assert change_set.base_source_version == "unversioned"


def test_dry_run_routes_through_sdk_without_a_durable_ledger(register, ledger):
    sink = register(_FakeSink(), enabled=False)
    out = core.run_writeback(_DOMAIN, dry_run=True, creations=[{"name": "a"}])
    assert out["status"] == "completed"
    assert sink.calls == [True]
    assert out["proposals"] == [{"op": "create", "items": [{"name": "a"}]}]
    evidence = out["governed_writeback"]
    assert evidence["connector_id"] == f"agent-utilities.writeback.{_DOMAIN}"
    assert evidence["authorization_mode"] == "standing_policy"
    assert evidence["effect_status"] is None
    assert not ledger.exists()


def test_live_write_records_change_set_audit_and_applied_receipt(register, ledger):
    sink = register(_FakeSink(), enabled=True)
    out = core.run_writeback(_DOMAIN, dry_run=False, creations=[{"name": "a"}])
    assert out["status"] == "completed"
    assert out["created"] == 1
    assert sink.calls == [False]
    assert out["governed_writeback"]["effect_status"] == "applied"
    directory = _only_change_set_dir(ledger)
    change_set = json.loads((directory / "change_set.json").read_text())
    assert change_set["change_set_id"] == out["governed_writeback"]["change_set_id"]
    assert (directory / "audit_reservation.json").exists()
    receipts = json.loads((directory / "receipts.json").read_text())
    assert [r["receipt"]["effect_status"] for r in receipts] == ["applied"]


def test_live_write_stays_refused_without_the_enable_flag(register, ledger):
    sink = register(_FakeSink(), enabled=False)
    out = core.run_writeback(_DOMAIN, dry_run=False, creations=[{"name": "a"}])
    assert out["status"] == "refused"
    assert sink.calls == []
    assert not ledger.exists()


def test_sink_failure_is_recorded_as_uncertain_and_reported(register, ledger):
    register(_FakeSink(fail=True), enabled=True)
    out = core.run_writeback(_DOMAIN, dry_run=False, creations=[{"name": "a"}])
    assert out == {
        "status": "error",
        "target": _DOMAIN,
        "error": "sink exploded mid-batch",
    }
    receipts = json.loads((_only_change_set_dir(ledger) / "receipts.json").read_text())
    assert [r["receipt"]["effect_status"] for r in receipts] == ["outcome_uncertain"]


def test_sdk_refuses_an_unauthorized_live_call_before_the_sink(tmp_path):
    sink = _FakeSink()
    with pytest.raises(AuthorizationDeniedError):
        execute_governed(
            sink,
            core.WritebackContext(),
            {"creations": []},
            grant=_grant(write_enabled=False),
            dry_run=False,
            root=tmp_path,
        )
    assert sink.calls == []


def test_sdk_refuses_a_stale_source_version_before_the_sink(tmp_path):
    sink = _FakeSink()
    ops = {"expected_source_version": "v1", "current_source_version": "v2"}
    with pytest.raises(SourceVersionConflictError):
        execute_governed(
            sink,
            core.WritebackContext(),
            ops,
            grant=_grant(),
            dry_run=False,
            root=tmp_path,
        )
    assert sink.calls == []


def test_approved_replay_uses_proposal_approval_mode(tmp_path):
    outcome = execute_governed(
        _FakeSink(),
        core.WritebackContext(),
        {"creations": [{"name": "a"}], "_approved": True},
        grant=_grant(approved=True),
        dry_run=False,
        root=tmp_path,
    )
    evidence = outcome.evidence()
    assert evidence["authorization_mode"] == "proposal_approval"
    assert evidence["effect_status"] == "applied"
    assert outcome.result.created == 1


def test_default_ledger_root_is_under_the_runtime_directory():
    from agent_utilities.core.paths import runtime_dir

    assert ledger_root() == runtime_dir() / "writeback-ledger"
