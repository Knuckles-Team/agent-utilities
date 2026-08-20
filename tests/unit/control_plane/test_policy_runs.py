"""Focused NE-086 policy/resolved-run acceptance fixtures."""

from __future__ import annotations

import hashlib

from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from agent_utilities.control_plane.policy import (
    ApprovalDecision,
    CapabilityBinding,
    ExecutionBudget,
    ExactSignatureVerifier,
    InMemoryPolicyRepository,
    PolicyAuthority,
    PolicyDomainError,
    PolicyException,
    PolicyRule,
    PolicyVersion,
)
from agent_utilities.control_plane.runs import (
    ArtifactRef,
    AuditChain,
    InMemoryNativeAdmission,
    InMemoryObservationLedger,
    InvocationRef,
    JobRef,
    NativeAdmissionRequest,
    NativeWorkItemAdmission,
    ObservationDiscontinuityError,
    RelationalObservation,
    ReplayDriftError,
    RunResolution,
    TaskRef,
    ToolBindingRef,
    TraceRef,
    native_work_item_id,
)


def _digest(letter: str) -> str:
    """A distinct, deterministic, CONTRACT-VALID digest per label.

    ``Digest`` requires ``^sha256:[0-9a-f]{64}$``. The previous ``letter * 64``
    form produced a non-hex string for any non-hex label -- i, j, o, p, q, r, s
    and t are all used here -- which the model correctly rejected. Deriving the
    body from the label keeps every call site and its distinctness while
    actually satisfying the contract.
    """
    return "sha256:" + hashlib.sha256(letter.encode("utf-8")).hexdigest()


def _budget(**overrides: int) -> ExecutionBudget:
    values = {
        "max_tasks": 8,
        "max_depth": 8,
        "max_fanout": 4,
        "max_tool_calls": 4,
        "max_tokens": 10_000,
        "max_seconds": 120,
        "max_cost_micros": 10_000,
    }
    values.update(overrides)
    return ExecutionBudget(**values)


def _binding(letter: str = "a") -> CapabilityBinding:
    return CapabilityBinding(
        kind="tool",
        binding_id="tool:search",
        version="1.0.0",
        digest=_digest(letter),
        privilege="read",
    )


def _policy(*, effect: str = "allow") -> PolicyVersion:
    return PolicyVersion(
        policy_id="policy:research",
        version="1.0.0",
        issuer_ref="authority:policy",
        budget=_budget(),
        rules=(
            PolicyRule(
                operation="run:research",
                effect=effect,
                approval_required=True,
                budget=_budget(),
            ),
        ),
        bindings=(_binding(),),
        max_privilege="read",
    )


def _authority(policy: PolicyVersion, approval: ApprovalDecision) -> PolicyAuthority:
    verifier = ExactSignatureVerifier(
        ((approval.signer_ref, approval.decision_digest, approval.signature),)
    )
    authority = PolicyAuthority(
        InMemoryPolicyRepository(),
        verifier=verifier,
        clock=lambda: 100,
    )
    authority.publish(policy)
    authority.record_approval(approval)
    return authority


def _approval(policy: PolicyVersion, *, request_digest: str) -> ApprovalDecision:
    return ApprovalDecision(
        approval_id="approval:research",
        policy_ref=policy.ref,
        operation="run:research",
        request_digest=request_digest,
        decision="approved",
        approver_refs=("principal:operator",),
        issued_at=90,
        expires_at=120,
        signer_ref="principal:operator",
        signature="sig:approval",
    )


def _resolution(authority: PolicyAuthority, *, request_digest: str) -> RunResolution:
    policy = _policy()
    approval = _approval(policy, request_digest=request_digest)
    # The helper's authority already contains the exact approval used by the
    # caller; constructing the resolution from that authorization pins it.
    authorization = authority.authorize(
        policy_ref=policy.ref,
        operation="run:research",
        request_digest=request_digest,
        bindings=(_binding(),),
        requested_budget=_budget(),
        approval=approval,
        now=100,
    )
    job = JobRef(job_id="job:research", version="1.0.0", digest=_digest("j"))
    task = TaskRef(
        task_id="task:research",
        job_id=job.job_id,
        version="1.0.0",
        digest=_digest("t"),
    )
    tool = ToolBindingRef(binding=_binding())
    invocation = InvocationRef(
        invocation_id="invocation:research",
        task_id=task.task_id,
        tool_binding_id=tool.binding_id,
        tool_binding_digest=tool.digest,
        input_digest=request_digest,
        trace_ref=TraceRef(trace_id="trace:research", digest=_digest("r")),
        artifact_refs=(ArtifactRef(artifact_id="artifact:source", digest=_digest("s")),),
    )
    return RunResolution(
        tenant_ref="tenant:demo",
        idempotency_key="request:001",
        request_digest=request_digest,
        authorization=authorization,
        job=job,
        tasks=(task,),
        tool_bindings=(tool,),
        invocations=(invocation,),
        trace_ref=invocation.trace_ref,
        artifact_refs=invocation.artifact_refs,
    )


def _admission_request(resolution: RunResolution) -> NativeAdmissionRequest:
    work_item_id = native_work_item_id(resolution.run_id)
    item = NativeWorkItemAdmission(
        run_id=resolution.run_id,
        work_item_id=work_item_id,
        tenant_ref=resolution.tenant_ref,
        idempotency_key=resolution.idempotency_key,
        kind="run:research",
        payload_ref="artifact:request",
        payload_digest=_digest("p"),
        request_digest=resolution.request_digest,
        resolution_digest=resolution.resolution_digest,
    )
    return NativeAdmissionRequest(resolution=resolution, work_item=item)


def test_policy_pins_exact_versions_and_rejects_budget_or_privilege_escalation() -> None:
    policy = _policy()
    assert policy.ref.digest == policy.policy_digest
    with pytest.raises(ValidationError, match="policy_rule_budget_escalation"):
        PolicyVersion(
            policy_id="policy:bad",
            version="1.0.0",
            issuer_ref="authority:policy",
            budget=_budget(max_tokens=10),
            rules=(
                PolicyRule(
                    operation="run:bad",
                    effect="allow",
                    budget=_budget(max_tokens=20),
                ),
            ),
        )
    with pytest.raises(ValidationError):
        CapabilityBinding(
            kind="tool",
            binding_id="tool:search",
            version="latest",
            digest=_digest("a"),
        )


def test_stale_approval_and_expired_signed_exception_fail_closed() -> None:
    policy = _policy(effect="deny")
    request_digest = _digest("q")
    approval = _approval(policy, request_digest=request_digest)
    authority = _authority(policy, approval)
    with pytest.raises(PolicyDomainError, match="policy_denied"):
        authority.authorize(
            policy_ref=policy.ref,
            operation="run:research",
            request_digest=request_digest,
            bindings=(_binding(),),
            requested_budget=_budget(),
            approval=approval,
            now=100,
        )

    exception = PolicyException(
        exception_id="exception:research",
        policy_ref=policy.ref,
        operation="run:research",
        request_digest=request_digest,
        scope_ref="scope:breakglass",
        budget=_budget(),
        allow_denied_operation=True,
        issued_at=90,
        expires_at=99,
        signer_ref="principal:breakglass",
        signature="sig:exception",
    )
    verifier = ExactSignatureVerifier(
        ((exception.signer_ref, exception.exception_digest, exception.signature),)
    )
    authority = PolicyAuthority(InMemoryPolicyRepository(), verifier=verifier, clock=lambda: 100)
    authority.publish(policy)
    authority.record_exception(exception)
    with pytest.raises(PolicyDomainError, match="exception_expired_or_not_yet_valid"):
        authority.authorize(
            policy_ref=policy.ref,
            operation="run:research",
            request_digest=request_digest,
            bindings=(_binding(),),
            requested_budget=_budget(),
            exception=exception,
            now=100,
        )


def test_admission_is_one_run_one_work_item_and_duplicate_delivery_is_idempotent() -> None:
    policy = _policy()
    request_digest = _digest("q")
    approval = _approval(policy, request_digest=request_digest)
    authority = _authority(policy, approval)
    resolution = _resolution(authority, request_digest=request_digest)
    request = _admission_request(resolution)
    admission = InMemoryNativeAdmission(
        audit=AuditChain(max_events=8),
        clock=lambda: 100,
        authorization_verifier=authority,
    )

    first = admission.admit_once(request)
    second = admission.admit_once(request)
    assert first.created is True
    assert second.created is False
    assert admission.read(resolution.run_id).resolution == resolution

    # Drift the request digest COHERENTLY: RunResolution's own validator
    # requires authorization.request_digest == request_digest, so updating only
    # the outer field builds an internally-invalid object that is rejected
    # before the replay-drift check is ever reached. Update both so the object
    # is valid but genuinely differs from the stored one.
    drifted_digest = _digest("d")
    altered = resolution.model_copy(
        update={
            "request_digest": drifted_digest,
            "authorization": resolution.authorization.model_copy(
                update={"request_digest": drifted_digest}
            ),
        }
    )
    # resolution_digest is a computed property over the WHOLE resolution, so
    # drifting the request digest necessarily changes it too. NativeAdmission
    # Request requires the work item to carry BOTH matching values, so the item
    # has to track both or the request is rejected as incoherent before the
    # replay-drift path runs.
    altered_item = request.work_item.model_copy(
        update={
            "request_digest": altered.request_digest,
            "resolution_digest": altered.resolution_digest,
        }
    )
    with pytest.raises(ReplayDriftError, match="replay_or_body_drift"):
        admission.admit_once(
            NativeAdmissionRequest(resolution=altered, work_item=altered_item)
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda _: admission.admit_once(request), range(4)))
    assert sum(receipt.created for receipt in receipts) == 0


def test_audit_chain_and_observations_fail_on_discontinuity_or_regression() -> None:
    chain = AuditChain(max_events=2)
    chain.append(
        kind="run.admitted",
        subject_ref="run:one",
        payload_digest=_digest("a"),
        observed_at=1,
    )
    chain.append(
        kind="work_item.admitted",
        subject_ref="work_item:one",
        payload_digest=_digest("b"),
        observed_at=1,
    )
    assert chain.verify()
    with pytest.raises(ValueError, match="audit_capacity_exceeded"):
        chain.append(
            kind="run.duplicate",
            subject_ref="run:one",
            payload_digest=_digest("c"),
            observed_at=2,
        )

    ledger = InMemoryObservationLedger()
    row = RelationalObservation(
        observation_id="observation:one",
        run_id="run:one",
        work_item_id="work_item:one",
        sequence=2,
        observed_state="pending",
        source_ref="mirror:relational",
        evidence_digest=_digest("e"),
        observed_at=1,
    )
    assert ledger.append(row) is True
    assert ledger.append(row) is False
    # A regressed sequence must arrive as a DISTINCT observation. Reusing
    # observation:one with different content trips the identity-drift check
    # first (it is evaluated before the sequence check, and is the stronger
    # violation), so the sequence path would never be exercised -- the
    # run-identity case immediately below already follows this convention.
    with pytest.raises(ObservationDiscontinuityError, match="sequence_regressed"):
        ledger.append(
            row.model_copy(
                update={"observation_id": "observation:regressed", "sequence": 1}
            )
        )
    with pytest.raises(ObservationDiscontinuityError, match="run_identity_drift"):
        ledger.append(
            row.model_copy(
                update={
                    "observation_id": "observation:other-run",
                    "run_id": "run:other",
                    "sequence": 3,
                    "observed_at": 2,
                }
            )
        )
    with pytest.raises(ObservationDiscontinuityError, match="time_regressed"):
        ledger.append(
            row.model_copy(
                update={
                    "observation_id": "observation:old-time",
                    "sequence": 3,
                    "observed_at": 0,
                }
            )
        )


def test_definition_side_objects_do_not_claim_or_complete_work() -> None:
    policy = _policy()
    approval = _approval(policy, request_digest=_digest("q"))
    admission = InMemoryNativeAdmission()
    assert not hasattr(admission, "claim_work_item")
    assert not hasattr(admission, "renew_work_item_lease")
    assert not hasattr(admission, "commit_work_item_result")
    assert not hasattr(approval, "result")
