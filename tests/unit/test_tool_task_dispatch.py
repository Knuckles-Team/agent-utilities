"""Worker wiring with controlled owner/backend ports, not served acceptance."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from agent_utilities.api.tool_task import (
    ToolTaskRefused,
    ToolTaskRequest,
    ToolTaskResult,
)
from agent_utilities.knowledge_graph.core import work_durability as work
from agent_utilities.knowledge_graph.core.queue_backend import MemoryQueueBackend
from agent_utilities.orchestration import agent_dispatch_worker as worker
from agent_utilities.orchestration.agent_dispatch import (
    KIND_TOOL_TASK,
    AgentTurnEnvelope,
    DispatchCarrierError,
)

from .test_agent_dispatch import _FakeOrchEngine
from .test_agent_dispatch_delivery_safety import _drain_loop

pytestmark = pytest.mark.spec("AU-CONTROL-R035.8")


def request(**changes: Any) -> ToolTaskRequest:
    return ToolTaskRequest.model_validate(
        {
            "operation_id": "fixture.read",
            "parameters_json": '{"key":"value"}',
            "principal": "person:one",
            "tenant": "tenant:one",
            "session_id": "session:one",
            "authority_ref": "authority:original",
            "policy_revision": "policy:one",
            "registry_digest": "a" * 64,
            **changes,
        }
    )


class Owner:
    """Controlled authority/effect port; does not qualify production identity."""

    def __init__(self, value: ToolTaskRequest) -> None:
        self.value = value
        self.effects = 0
        self.revoked = False
        self.wrong_result = False

    def resolve(self, ref: str, *, tenant: str, session_id: str) -> ToolTaskRequest:
        return self.value

    def execute(
        self,
        value: ToolTaskRequest,
        *,
        job_id: str,
        require_current: Callable[[], None],
    ) -> ToolTaskResult:
        if self.revoked:
            raise ToolTaskRefused("authority_revoked")
        require_current()
        self.effects += 1
        return ToolTaskResult(
            job_id=job_id,
            request_ref=value.request_ref,
            principal="person:other" if self.wrong_result else value.principal,
            outcome="completed",
            receipt_ref="receipt:controlled-effect",
        )


def admitted(value: ToolTaskRequest) -> tuple[AgentTurnEnvelope, Any]:
    envelope = AgentTurnEnvelope(
        kind=KIND_TOOL_TASK,
        job_id="tool-job:one",
        session_id=value.session_id,
        tenant=value.tenant,
        payload_ref=value.request_ref,
        allowed_tool_subset=(value.operation_id,),
    )
    envelope.ensure_authenticated_carrier()
    engine = _FakeOrchEngine()
    work.submit_work_item(
        engine,
        kind="agent_turn",
        tenant=value.tenant,
        payload_ref=value.request_ref,
        work_item_id=f"workitem:dispatch:{envelope.job_id}",
        idempotency_key=envelope.job_id,
    )
    return envelope, engine


def test_worker_executes_once_and_commits_bound_receipt() -> None:
    value = request()
    envelope, engine = admitted(value)
    owner = Owner(value)
    assert (
        worker.execute_agent_turn(envelope, engine, tool_task_dispatcher=owner)
        == "completed"
    )
    assert (
        worker.execute_agent_turn(envelope, engine, tool_task_dispatcher=owner)
        == "skipped"
    )
    assert owner.effects == 1
    row = work.get_work_item(engine, f"workitem:dispatch:{envelope.job_id}")
    assert row["status"] == "succeeded"
    assert row["result_ref"] == "receipt:controlled-effect"


@pytest.mark.parametrize(
    "failure", ["missing", "params", "authority", "tenant", "subset", "revoked"]
)
def test_worker_refusals_have_no_effect(failure: str) -> None:
    value = request()
    envelope, engine = admitted(value)
    owner = Owner(value)
    if failure == "params":
        owner.value = request(parameters_json='{"key":"changed"}')
    elif failure == "authority":
        owner.value = request(authority_ref="authority:other")
    elif failure == "tenant":
        owner.value = request(tenant="tenant:other")
    elif failure == "subset":
        envelope.allowed_tool_subset = ("fixture.other",)
        envelope.carrier = None
        envelope.ensure_authenticated_carrier()
    elif failure == "revoked":
        owner.revoked = True
    assert worker.execute_agent_turn(
        envelope,
        engine,
        tool_task_dispatcher=None if failure == "missing" else owner,
    ) == ("indeterminate" if failure == "revoked" else "failed")
    assert owner.effects == 0
    row = work.get_work_item(engine, f"workitem:dispatch:{envelope.job_id}")
    if failure == "revoked":
        assert row["status"] not in work.TERMINAL_WORK_ITEM_STATUSES
    else:
        assert row["status"] == "failed"


def test_wrong_result_cannot_report_success_or_repeat_effect() -> None:
    value = request()
    envelope, engine = admitted(value)
    owner = Owner(value)
    owner.wrong_result = True
    assert (
        worker.execute_agent_turn(envelope, engine, tool_task_dispatcher=owner)
        == "indeterminate"
    )
    assert (
        worker.execute_agent_turn(envelope, engine, tool_task_dispatcher=owner)
        == "fenced"
    )
    assert owner.effects == 1


def test_lost_lease_prevents_owner_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    value = request()
    envelope, engine = admitted(value)
    owner = Owner(value)
    monkeypatch.setattr(
        worker, "_work_item_fence_still_valid", lambda *args, **kwargs: False
    )
    assert (
        worker.execute_agent_turn(envelope, engine, tool_task_dispatcher=owner)
        == "fenced"
    )
    assert owner.effects == 0


@pytest.mark.parametrize(
    "text", ["[]", '{"a":1,"a":2}', '{"a":NaN}', '{ "a":1}', '{"a":Infinity}']
)
def test_noncanonical_or_nonobject_parameters_refused(text: str) -> None:
    with pytest.raises((ValueError, ValidationError)):
        request(parameters_json=text)


def test_registered_consumer_loop_passes_owner_and_acks_terminal_result() -> None:
    value = request()
    envelope, engine = admitted(value)
    owner = Owner(value)
    queue = MemoryQueueBackend()
    queue.put(envelope.to_item())
    _drain_loop(queue, engine, tool_task_dispatcher=owner)
    assert owner.effects == 1
    assert queue.get_queue_size() == 0
    row = work.get_work_item(engine, f"workitem:dispatch:{envelope.job_id}")
    assert row["status"] == "succeeded"


def test_tampered_delivery_refused_before_work_item_claim() -> None:
    value = request()
    envelope, engine = admitted(value)
    envelope.allowed_tool_subset = ("fixture.other",)
    owner = Owner(value)
    with pytest.raises(DispatchCarrierError):
        worker.execute_agent_turn(envelope, engine, tool_task_dispatcher=owner)
    row = work.get_work_item(engine, f"workitem:dispatch:{envelope.job_id}")
    assert row["status"] == "ready"
    assert owner.effects == 0


def test_signed_different_request_cannot_replace_existing_work_item() -> None:
    value = request()
    envelope, engine = admitted(value)
    owner = Owner(request(parameters_json='{"key":"replacement"}'))
    envelope.payload_ref = owner.value.request_ref
    envelope.carrier = None
    envelope.ensure_authenticated_carrier()
    with pytest.raises(ToolTaskRefused, match="work_item_mismatch"):
        worker.execute_agent_turn(envelope, engine, tool_task_dispatcher=owner)
    row = work.get_work_item(engine, f"workitem:dispatch:{envelope.job_id}")
    assert row["status"] == "ready"
    assert owner.effects == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"version": 2},
        {"registry_digest": "registry:one"},
        {"registry_digest": "A" * 64},
    ],
)
def test_request_version_and_registry_binding_are_closed(
    changes: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        request(**changes)


def test_terminal_fence_loss_redelivery_uses_controlled_owner_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only worker/port wiring: this controlled owner is not production durability."""
    value = request()
    envelope, engine = admitted(value)
    receipts: dict[str, ToolTaskResult] = {}

    class ReceiptOwner(Owner):
        def execute(
            self,
            value: ToolTaskRequest,
            *,
            job_id: str,
            require_current: Callable[[], None],
        ) -> ToolTaskResult:
            if job_id in receipts:
                return receipts[job_id]
            result = super().execute(
                value, job_id=job_id, require_current=require_current
            )
            receipts[job_id] = result
            return result

    owner = ReceiptOwner(value)
    original_fence = worker._work_item_fence_still_valid
    monkeypatch.setattr(
        worker,
        "_work_item_fence_still_valid",
        lambda *args, **kwargs: not receipts and original_fence(*args, **kwargs),
    )
    assert (
        worker.execute_agent_turn(envelope, engine, tool_task_dispatcher=owner)
        == "fenced"
    )
    row = work.get_work_item(engine, f"workitem:dispatch:{envelope.job_id}")
    assert row["status"] in {"leased", "running"}
    assert not row.get("result_ref")
    assert owner.effects == 1
    # Controlled backend simulates expiry, permitting a new native lease epoch.
    engine.nodes[f"workitem:dispatch:{envelope.job_id}"]["lease_expires_at"] = 0
    engine.nodes[f"workitem:dispatch:{envelope.job_id}"]["max_attempts"] = 2
    monkeypatch.setattr(worker, "_work_item_fence_still_valid", original_fence)
    assert (
        worker.execute_agent_turn(envelope, engine, tool_task_dispatcher=owner)
        == "completed"
    )
    assert owner.effects == 1
    row = work.get_work_item(engine, f"workitem:dispatch:{envelope.job_id}")
    assert row["result_ref"] == receipts[envelope.job_id].receipt_ref


def test_indeterminate_and_live_claim_never_ack_before_owner_reconciliation() -> None:
    value = request()
    envelope, engine = admitted(value)

    class ReconcilingOwner(Owner):
        receipt: ToolTaskResult | None = None

        def execute(
            self,
            value: ToolTaskRequest,
            *,
            job_id: str,
            require_current: Callable[[], None],
        ) -> ToolTaskResult:
            if self.receipt is None:
                self.receipt = (
                    super()
                    .execute(value, job_id=job_id, require_current=require_current)
                    .model_copy(update={"outcome": "indeterminate"})
                )
            return self.receipt

    class Queue:
        acknowledgments = 0

        def ack(self, item_id: Any) -> None:
            self.acknowledgments += 1

    owner = ReconcilingOwner(value)
    queue = Queue()
    item_id = f"workitem:dispatch:{envelope.job_id}"
    first = worker.execute_agent_turn(envelope, engine, tool_task_dispatcher=owner)
    assert first == "indeterminate"
    worker._handle_dispatch_outcome(queue, "delivery", engine, item_id, first, 0)
    second = worker.execute_agent_turn(envelope, engine, tool_task_dispatcher=owner)
    assert second == "fenced"
    worker._handle_dispatch_outcome(queue, "delivery", engine, item_id, second, 0)
    assert queue.acknowledgments == 0
    assert owner.effects == 1
    assert owner.receipt is not None
    assert (
        work.get_work_item(engine, item_id)["status"]
        not in work.TERMINAL_WORK_ITEM_STATUSES
    )
    # Controlled authority reconciles its recorded effect; worker cannot do so.
    owner.receipt = owner.receipt.model_copy(
        update={"outcome": "completed", "receipt_ref": "receipt:reconciled"}
    )
    engine.nodes[item_id]["lease_expires_at"] = 0
    engine.nodes[item_id]["max_attempts"] = 2
    third = worker.execute_agent_turn(envelope, engine, tool_task_dispatcher=owner)
    assert third == "completed"
    worker._handle_dispatch_outcome(queue, "delivery", engine, item_id, third, 0)
    assert queue.acknowledgments == 1
    assert owner.effects == 1
    assert work.get_work_item(engine, item_id)["result_ref"] == "receipt:reconciled"


@pytest.mark.parametrize(
    "failure",
    [
        "mark_running",
        "effect_then_timeout",
        "effect_then_refused",
        "effect_then_bad_result",
    ],
)
def test_tool_task_uncertainty_never_terminal_commits_or_acks(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = request()
    envelope, engine = admitted(value)

    class TimeoutOwner(Owner):
        def execute(
            self,
            value: ToolTaskRequest,
            *,
            job_id: str,
            require_current: Callable[[], None],
        ) -> ToolTaskResult:
            super().execute(value, job_id=job_id, require_current=require_current)
            if failure == "effect_then_refused":
                raise ToolTaskRefused("owner_cannot_prove_pre_effect")
            if failure == "effect_then_bad_result":
                return ToolTaskResult(
                    job_id=job_id,
                    request_ref=value.request_ref,
                    principal="wrong:principal",
                    outcome="completed",
                    receipt_ref="receipt:wrong",
                )
            raise TimeoutError("reply lost after controlled effect")

    class Queue:
        acknowledgments = 0

        def ack(self, item_id: Any) -> None:
            self.acknowledgments += 1

    owner = TimeoutOwner(value)
    if failure == "mark_running":
        monkeypatch.setattr(work, "mark_running", lambda *args, **kwargs: False)
    outcome = worker.execute_agent_turn(envelope, engine, tool_task_dispatcher=owner)
    assert outcome == ("fenced" if failure == "mark_running" else "indeterminate")
    item_id = f"workitem:dispatch:{envelope.job_id}"
    queue = Queue()
    worker._handle_dispatch_outcome(queue, "delivery", engine, item_id, outcome, 0)
    row = work.get_work_item(engine, item_id)
    assert row["status"] not in work.TERMINAL_WORK_ITEM_STATUSES
    assert not row.get("error_ref")
    assert not row.get("result_ref")
    assert queue.acknowledgments == 0
    assert owner.effects == (0 if failure == "mark_running" else 1)


def test_expired_tool_deadline_cannot_cancel_uncertain_effect_or_ack() -> None:
    value = request()
    envelope, engine = admitted(value)
    now = time.time()
    envelope.deadline_unix = now + 30
    envelope.carrier = None
    envelope.ensure_authenticated_carrier()

    class TimeoutOwner(Owner):
        def execute(
            self,
            value: ToolTaskRequest,
            *,
            job_id: str,
            require_current: Callable[[], None],
        ) -> ToolTaskResult:
            super().execute(value, job_id=job_id, require_current=require_current)
            raise TimeoutError("lost controlled receipt")

    class Queue:
        acknowledgments = 0

        def ack(self, item_id: Any) -> None:
            self.acknowledgments += 1

    owner = TimeoutOwner(value)
    assert (
        worker.execute_agent_turn(envelope, engine, tool_task_dispatcher=owner, now=now)
        == "indeterminate"
    )
    outcome = worker.execute_agent_turn(
        envelope, engine, tool_task_dispatcher=owner, now=now + 31
    )
    assert outcome == "indeterminate"
    item_id = f"workitem:dispatch:{envelope.job_id}"
    queue = Queue()
    worker._handle_dispatch_outcome(queue, "delivery", engine, item_id, outcome, 0)
    assert queue.acknowledgments == 0
    assert owner.effects == 1
    assert (
        work.get_work_item(engine, item_id)["status"]
        not in work.TERMINAL_WORK_ITEM_STATUSES
    )


@pytest.mark.parametrize("rejection", ["forged", "expired"])
def test_rejected_tool_carrier_cannot_ack_unresolved_original_task(
    rejection: str,
) -> None:
    value = request()
    envelope, engine = admitted(value)

    class Queue:
        acknowledgments = 0

        def ack(self, item_id: Any) -> None:
            self.acknowledgments += 1

    assert envelope.carrier is not None
    changes: dict[str, Any] = {"signature": "0" * 64}
    if rejection == "expired":
        changes.update(issued_at=time.time() - 60, expires_at=time.time() - 30)
    envelope.carrier = envelope.carrier.model_copy(update=changes)
    queue = Queue()
    assert not worker._authenticate_or_poison(
        engine, queue, "delivery", envelope.to_item(), envelope, 0
    )
    assert queue.acknowledgments == 0
    assert (
        work.get_work_item(engine, f"workitem:dispatch:{envelope.job_id}")["status"]
        == "ready"
    )
