"""Focused NE-177 fixtures for durable fleet action fencing.

These fixtures model the narrow engine-native outbox protocol exposed by
``fleet_actuation``.  They deliberately do not replace the production engine:
the tests prove that an actuator is never called without a committed fence,
that completion loss leaves an observable recovery record, that a same-key
payload conflict is rejected, and that duplicate approval delivery cannot
re-run a side effect.
"""

from __future__ import annotations

import json
from typing import Any

from agent_utilities.orchestration.action_policy import ActionRequest
from agent_utilities.orchestration.fleet_actuation import (
    _action_request_digest,
    execute_action,
)
from agent_utilities.orchestration.fleet_reconciler import FleetReconciler

from .fleet_autonomy_fakes import FakeEngine, FakeObserver, obs


class DurableActionOutbox:
    """Transactional fixture for the ActionOutboxStore contract."""

    def __init__(self, engine: FakeEngine | None = None) -> None:
        self.engine = engine
        self.records: dict[str, dict[str, Any]] = {}
        self.fail_prepare = False
        self.fail_complete = False
        self.fail_complete_after_commit = False

    def prepare(self, request: dict[str, Any]) -> dict[str, Any]:
        key = str(request["idempotency_key"])
        existing = self.records.get(key)
        if existing is not None:
            if (
                existing.get("kind") != request.get("kind")
                or existing.get("target") != request.get("target")
                or existing.get("params") != request.get("params")
                or existing.get("request_digest") != request.get("request_digest")
                or existing.get("approval_id") != request.get("approval_id", "")
            ):
                return {
                    "accepted": False,
                    "durability_available": True,
                    "outcome_unknown": False,
                    "conflict": True,
                    "reason": "outbox identity conflict",
                }
            return {"accepted": True, "replayed": True, **dict(existing)}
        if self.fail_prepare:
            return {
                "accepted": False,
                "durability_available": True,
                "outcome_unknown": False,
                "rejected": True,
                "reason": "fixture prepare failure",
            }
        self.records[key] = {
            "idempotency_key": key,
            "execution_id": request["execution_id"],
            "status": "prepared",
            "kind": request["kind"],
            "target": request["target"],
            "params": dict(request.get("params") or {}),
            "approval_id": request.get("approval_id", ""),
            "request_digest": request.get("request_digest", ""),
        }
        return {"accepted": True, "status": "prepared"}

    def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.fail_complete:
            return {"accepted": False, "reason": "fixture completion loss"}
        key = str(request["idempotency_key"])
        record = self.records[key]
        if (
            record.get("request_digest") != request.get("request_digest")
            or record.get("approval_id") != request.get("approval_id", "")
        ):
            return {
                "accepted": False,
                "durability_available": True,
                "conflict": True,
                "reason": "outbox completion identity conflict",
            }
        record.update(
            {
                "status": request["state"],
                "ok": bool(request.get("ok")),
                "dry_run": bool(request.get("dry_run")),
            }
        )
        approval_id = str(request.get("approval_id") or "")
        if approval_id and self.engine is not None:
            approval = self.engine.nodes.get(approval_id)
            if approval is not None:
                approval["status"] = str(
                    request.get("approval_status") or request["state"]
                )
        if self.fail_complete_after_commit:
            raise RuntimeError("fixture completion acknowledgement lost")
        return {
            "accepted": True,
            "status": record["status"],
            "approval_committed": bool(approval_id),
        }


class CountingActuator:
    name = "counting"

    def __init__(self, *, raise_after_apply: bool = False) -> None:
        self.calls: list[ActionRequest] = []
        self.raise_after_apply = raise_after_apply

    def apply(self, request: ActionRequest) -> dict[str, Any]:
        self.calls.append(request)
        if self.raise_after_apply:
            raise RuntimeError("fixture crash after side effect")
        return {"ok": True, "dry_run": False, "detail": "fixture applied"}


class ProjectionFailEngine(FakeEngine):
    """Compatibility projection failure; the durable outbox remains authority."""

    def add_node(
        self, node_id: str, node_type: str, properties: dict | None = None
    ):
        if node_type == "ActionExecution":
            raise RuntimeError("fixture projection write failure")
        return super().add_node(node_id, node_type, properties)


def _request() -> ActionRequest:
    return ActionRequest(
        kind="restart_service",
        target="graph-os",
        source="fixture",
        reason="durable outbox fixture",
    )


def test_real_action_fails_closed_without_durable_outbox():
    engine = FakeEngine()
    actuator = CountingActuator()

    result = execute_action(engine, _request(), actuator)

    assert result["ok"] is False
    assert result["durability_unavailable"] is True
    assert actuator.calls == []
    assert engine.by_type("ActionExecution") == []


def test_prepare_failure_blocks_side_effect():
    engine = FakeEngine()
    outbox = DurableActionOutbox(engine)
    outbox.fail_prepare = True
    actuator = CountingActuator()

    result = execute_action(engine, _request(), actuator, outbox_store=outbox)

    assert result["ok"] is False
    assert result["state"] == "failed"
    assert actuator.calls == []
    assert outbox.records == {}


def test_compatibility_evidence_projection_failure_does_not_erase_outbox():
    engine = ProjectionFailEngine()
    outbox = DurableActionOutbox(engine)
    actuator = CountingActuator()

    result = execute_action(engine, _request(), actuator, outbox_store=outbox)

    assert result["state"] == "executed"
    assert result["ok"] is True
    assert outbox.records[result["idempotency_key"]]["status"] == "executed"
    assert engine.by_type("ActionExecution") == []


def test_completion_loss_marks_recovery_and_replay_never_reapplies():
    engine = FakeEngine()
    outbox = DurableActionOutbox(engine)
    outbox.fail_complete = True
    actuator = CountingActuator()
    request = _request()

    first = execute_action(
        engine, request, actuator, outbox_store=outbox, idempotency_key="approval:a"
    )
    second = execute_action(
        engine, request, actuator, outbox_store=outbox, idempotency_key="approval:a"
    )

    assert first["state"] == "recovery_pending"
    assert second["state"] == "recovery_pending"
    assert second["replayed"] is True
    assert len(actuator.calls) == 1


def test_success_lost_ack_replays_durable_terminal_without_reapply():
    engine = FakeEngine()
    outbox = DurableActionOutbox(engine)
    outbox.fail_complete_after_commit = True
    actuator = CountingActuator()
    request = _request()

    first = execute_action(
        engine,
        request,
        actuator,
        outbox_store=outbox,
        idempotency_key="ack-lost",
    )
    assert first["state"] == "recovery_pending"
    assert outbox.records["ack-lost"]["status"] == "executed"

    outbox.fail_complete_after_commit = False
    replay = execute_action(
        engine,
        request,
        actuator,
        outbox_store=outbox,
        idempotency_key="ack-lost",
    )

    assert replay["state"] == "executed"
    assert replay["replayed"] is True
    assert len(actuator.calls) == 1


def test_crash_after_prepare_requires_observation_before_retry():
    engine = FakeEngine()
    outbox = DurableActionOutbox(engine)
    actuator = CountingActuator()
    request = _request()
    outbox.prepare(
        {
            "operation": "prepare",
            "idempotency_key": "crash:before",
            "execution_id": "execution:crash-before",
            "kind": request.kind,
            "target": request.target,
            "params": {},
            "source": request.source,
            "reason": request.reason,
            "request_digest": _action_request_digest(request),
        }
    )

    result = execute_action(
        engine,
        request,
        actuator,
        outbox_store=outbox,
        idempotency_key="crash:before",
    )

    assert result["state"] == "recovery_pending"
    assert result["replayed"] is True
    assert actuator.calls == []


def test_action_success_and_duplicate_delivery_are_idempotent():
    engine = FakeEngine()
    outbox = DurableActionOutbox(engine)
    actuator = CountingActuator()
    request = _request()

    first = execute_action(
        engine,
        request,
        actuator,
        outbox_store=outbox,
        idempotency_key="approval:duplicate",
        approval_id="approval-1",
    )
    second = execute_action(
        engine,
        request,
        actuator,
        outbox_store=outbox,
        idempotency_key="approval:duplicate",
        approval_id="approval-1",
    )

    assert first["state"] == "executed"
    assert second["state"] == "executed"
    assert second["replayed"] is True
    assert second["approval_committed"] is True
    assert len(actuator.calls) == 1


def test_same_outbox_key_with_different_action_is_rejected():
    engine = FakeEngine()
    outbox = DurableActionOutbox(engine)
    actuator = CountingActuator()
    first_request = _request()
    second_request = ActionRequest(
        kind="restart_service", target="different-service", source="fixture"
    )

    first = execute_action(
        engine,
        first_request,
        actuator,
        outbox_store=outbox,
        idempotency_key="identity:conflict",
    )
    second = execute_action(
        engine,
        second_request,
        actuator,
        outbox_store=outbox,
        idempotency_key="identity:conflict",
    )

    assert first["state"] == "executed"
    assert second["ok"] is False
    assert second["state"] == "failed"
    assert "identity conflict" in second["detail"]
    assert len(actuator.calls) == 1


def test_crash_after_apply_is_not_replayed():
    engine = FakeEngine()
    outbox = DurableActionOutbox(engine)
    actuator = CountingActuator(raise_after_apply=True)
    request = _request()

    first = execute_action(
        engine, request, actuator, outbox_store=outbox, idempotency_key="crash:after"
    )
    second = execute_action(
        engine, request, actuator, outbox_store=outbox, idempotency_key="crash:after"
    )

    assert first["state"] == "recovery_pending"
    assert second["state"] == "recovery_pending"
    assert second["replayed"] is True
    assert len(actuator.calls) == 1


def test_duplicate_approval_drain_closes_once():
    engine = FakeEngine()
    outbox = DurableActionOutbox(engine)
    outbox.engine = engine
    engine.add_node(
        "approval-1",
        "ActionApproval",
        properties={
            "kind": "restart_service",
            "target": "graph-os",
            "params_json": json.dumps({}),
            "source": "fixture",
            "reason": "fixture approval",
            "status": "approved",
        },
    )
    actuator = CountingActuator()
    reconciler = FleetReconciler(
        engine,
        observer=FakeObserver({"graph-os": obs("graph-os", "up", replicas=1)}),
        actuator=actuator,
        action_outbox_store=outbox,
    )

    first = reconciler._drain_approved(1)
    second = reconciler._drain_approved(1)

    assert first[0]["status"] == "executed"
    assert second == []
    assert engine.nodes["approval-1"]["status"] == "executed"
    assert len(actuator.calls) == 1


def test_approval_completion_loss_reconciles_observed_without_reapply():
    engine = FakeEngine()
    outbox = DurableActionOutbox(engine)
    outbox.fail_complete = True
    engine.add_node(
        "approval-recovery",
        "ActionApproval",
        properties={
            "kind": "restart_service",
            "target": "graph-os",
            "params_json": json.dumps({}),
            "source": "fixture",
            "reason": "fixture approval recovery",
            "status": "approved",
        },
    )
    actuator = CountingActuator()
    reconciler = FleetReconciler(
        engine,
        observer=FakeObserver({"graph-os": obs("graph-os", "up", replicas=1)}),
        actuator=actuator,
        action_outbox_store=outbox,
    )

    first = reconciler._drain_approved(1)
    assert first[0]["status"] == "recovery_pending"
    assert engine.nodes["approval-recovery"]["status"] == "approved"
    assert len(actuator.calls) == 1

    outbox.fail_complete = False
    recovered = reconciler._drain_approved(1)
    assert recovered[0]["status"] == "observed"
    assert engine.nodes["approval-recovery"]["status"] == "observed"
    assert len(actuator.calls) == 1
    assert reconciler._drain_approved(1) == []
