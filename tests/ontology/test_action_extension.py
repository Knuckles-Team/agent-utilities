#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:KG-2.42 — Action-Type extension tests (extends KG-2.25).

Provenance (Palantir AIP doc: *action-types/overview*): typed multi-effect
actions journaled as durable edits, submission-criteria gating, undo/revert,
notification + webhook dispatch, and batch. These tests exercise the *live*
governed path through :class:`ActionExecutor` against the C1 Edit Ledger and the
in-process recording dispatch sinks — fully offline (no KG backend required).
"""

import pytest

from agent_utilities.knowledge_graph.actions import (
    ActionEffect,
    ActionEffectSpec,
    ActionExecutor,
    ActionParameter,
    ActionRegistry,
    ActionStatus,
    CriterionOp,
    EffectKind,
    NotificationSpec,
    OntologyAction,
    RecordingNotifier,
    SubmissionCriterion,
    WebhookSpec,
)
from agent_utilities.knowledge_graph.ontology.edits import EditLedger
from agent_utilities.observability.escalation_matrix import make_decision_provider
from agent_utilities.security.permissions_kernel import AgentRole, PermissionsKernel

# These mutating actions are HIGH-risk (mutation + non-idempotent); the HITL
# escalation gate (CONCEPT:OS-5.12) requires an operator/admin approval, so the
# tests supply an approving decision_provider — exercising the full governed path
# (authorize → escalate-approve → validate → submission-criteria → side-effects).
APPROVE = make_decision_provider(
    {
        name: {"approved": True, "approver": "ops", "approver_role": "operator"}
        for name in ("demo.onboard", "demo.tag_batch")
    }
)


@pytest.fixture
def kernel() -> PermissionsKernel:
    return PermissionsKernel()


@pytest.fixture
def ledger() -> EditLedger:
    # Hermetic, in-memory ledger (no backend probed): exercises the real apply/
    # journal/revert path against an in-memory graph_state.
    return EditLedger()


def _writer(kernel: PermissionsKernel):
    return kernel.issue_identity(
        "agent:writer", role=AgentRole.SPECIALIST, capabilities=["kg_write"]
    )


def _onboard_action() -> OntologyAction:
    """An action with TWO typed side-effects: create an object, then link it."""
    return OntologyAction(
        name="demo.onboard",
        verb="onboard",
        description="Create a record object and link it to its owner.",
        parameters=[
            ActionParameter(name="record_id", required=True),
            ActionParameter(name="owner_id", required=True),
            ActionParameter(name="title", required=True),
        ],
        acts_on=["record"],
        required_capability="kg_write",
        produces_effect=ActionEffect.MUTATION,
        idempotent=False,
        submission_criteria=[
            SubmissionCriterion(
                field="params.title",
                op=CriterionOp.NON_EMPTY,
                message="title must be non-empty",
            ),
        ],
        side_effects=[
            ActionEffectSpec(
                kind=EffectKind.CREATE_OBJECT,
                target="$record_id",
                params={"title": "$title", "type": "record"},
            ),
            ActionEffectSpec(
                kind=EffectKind.ADD_LINK,
                target="$record_id",
                params={"link_target": "$owner_id", "link_label": "owned_by"},
            ),
        ],
        notifications=[
            NotificationSpec(
                channel="ops", recipient="$owner_id", template="record ${record_id} created"
            )
        ],
        webhooks=[WebhookSpec(url="https://example.invalid/hook")],
    )


def test_two_side_effects_apply_and_write_two_edits(
    kernel: PermissionsKernel, ledger: EditLedger
) -> None:
    reg = ActionRegistry()
    reg.register(_onboard_action(), handler=lambda p: {"ok": p["record_id"]})
    ex = ActionExecutor(reg, kernel=kernel, persist=False, ledger=ledger)

    inv = ex.execute(
        "demo.onboard",
        _writer(kernel),
        {"record_id": "record:1", "owner_id": "user:a", "title": "Q3 filing"},
        decision_provider=APPROVE,
    )

    assert inv.status == ActionStatus.SUCCESS
    # Two side-effects → two durable edit-ledger records on the invocation.
    assert len(inv.edit_ids) == 2
    assert len(ledger.all_edits()) == 2
    # The CREATE side-effect actually mutated the versioned graph_state.
    node = ledger.graph_state["nodes"].get("record:1")
    assert node is not None and node["title"] == "Q3 filing"
    # The ADD_LINK side-effect added the edge.
    assert ("record:1", "user:a", "owned_by") in ledger.graph_state["edges"]
    # Each edit references the producing invocation (full action provenance).
    for eid in inv.edit_ids:
        assert ledger.get(eid).invocation_ref == inv.id


def test_submission_criteria_denies_invalid_invocation(
    kernel: PermissionsKernel, ledger: EditLedger
) -> None:
    reg = ActionRegistry()
    reg.register(_onboard_action(), handler=lambda p: None)
    ex = ActionExecutor(reg, kernel=kernel, persist=False, ledger=ledger)

    # Empty title fails the NON_EMPTY submission criterion → denied, NO edits.
    inv = ex.execute(
        "demo.onboard",
        _writer(kernel),
        {"record_id": "record:2", "owner_id": "user:b", "title": ""},
        decision_provider=APPROVE,
    )
    assert inv.status == ActionStatus.DENIED
    assert "title must be non-empty" in inv.error
    assert ledger.all_edits() == []  # criteria gate before any mutation


def test_undo_reverts_invocation_edits(
    kernel: PermissionsKernel, ledger: EditLedger
) -> None:
    reg = ActionRegistry()
    reg.register(_onboard_action(), handler=lambda p: None)
    ex = ActionExecutor(reg, kernel=kernel, persist=False, ledger=ledger)

    inv = ex.execute(
        "demo.onboard",
        _writer(kernel),
        {"record_id": "record:3", "owner_id": "user:c", "title": "T"},
        decision_provider=APPROVE,
    )
    assert inv.status == ActionStatus.SUCCESS
    assert "record:3" in ledger.graph_state["nodes"]
    assert ("record:3", "user:c", "owned_by") in ledger.graph_state["edges"]

    compensating = ex.undo(inv, actor="agent:writer")
    # Two compensating edits recorded (append-only revert trail).
    assert len(compensating) == 2
    # State restored: object deleted, link removed.
    assert "record:3" not in ledger.graph_state["nodes"]
    assert ("record:3", "user:c", "owned_by") not in ledger.graph_state["edges"]


def test_notification_and_webhook_dispatch_recorded(
    kernel: PermissionsKernel, ledger: EditLedger
) -> None:
    reg = ActionRegistry()
    reg.register(_onboard_action(), handler=lambda p: None)
    notifier = RecordingNotifier()
    ex = ActionExecutor(
        reg, kernel=kernel, persist=False, ledger=ledger, notifier=notifier
    )

    inv = ex.execute(
        "demo.onboard",
        _writer(kernel),
        {"record_id": "record:4", "owner_id": "user:d", "title": "X"},
        decision_provider=APPROVE,
    )
    assert inv.status == ActionStatus.SUCCESS
    kinds = [d["kind"] for d in inv.dispatches]
    assert "notification" in kinds
    assert "webhook" in kinds
    # Notification was really produced (recorded sink), with template resolved.
    assert notifier.records
    assert notifier.records[0]["message"] == "record record:4 created"
    # Webhook attempt is journaled (httpx delivers or it is recorded) — never lost.
    webhook = next(d for d in inv.dispatches if d["kind"] == "webhook")
    assert webhook["transport"] in {"httpx", "recorded"}


def test_batch_over_three_targets(
    kernel: PermissionsKernel, ledger: EditLedger
) -> None:
    action = OntologyAction(
        name="demo.tag_batch",
        verb="tag",
        parameters=[ActionParameter(name="targets", required=False)],
        acts_on=["record"],
        required_capability="kg_write",
        produces_effect=ActionEffect.MUTATION,
        idempotent=False,
        batch=True,
        side_effects=[
            ActionEffectSpec(
                kind=EffectKind.CREATE_OBJECT,
                target="$target",
                params={"tagged": "yes"},
            ),
        ],
    )
    reg = ActionRegistry()
    reg.register(action, handler=lambda p: p.get("target"))
    ex = ActionExecutor(reg, kernel=kernel, persist=False, ledger=ledger)

    inv = ex.execute(
        "demo.tag_batch",
        _writer(kernel),
        {"targets": ["r:1", "r:2", "r:3"]},
        decision_provider=APPROVE,
    )
    assert inv.status == ActionStatus.SUCCESS
    # One sub-invocation per target, one edit each → three edits total.
    assert len(inv.batch_results) == 3
    assert len(inv.edit_ids) == 3
    assert len(ledger.all_edits()) == 3
    for rid in ("r:1", "r:2", "r:3"):
        assert ledger.graph_state["nodes"][rid]["tagged"] == "yes"
