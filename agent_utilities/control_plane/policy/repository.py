"""Policy repository and approval/exception authority seams."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Iterable
from typing import Protocol, runtime_checkable

from .models import (
    ApprovalDecision,
    ApprovalRef,
    CapabilityBinding,
    ExceptionRef,
    ExecutionBudget,
    PolicyAuthorization,
    PolicyException,
    PolicyRef,
    PolicyRule,
    PolicyVersion,
    SignatureVerifier,
)

__all__ = [
    "ExactSignatureVerifier",
    "InMemoryPolicyRepository",
    "PolicyAuthority",
    "PolicyConflictError",
    "PolicyDomainError",
    "PolicyNotFoundError",
    "PolicyRepository",
]

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class PolicyDomainError(ValueError):
    """Base fail-closed policy-domain error."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(code if not detail else f"{code}: {detail}")


class PolicyConflictError(PolicyDomainError):
    """An immutable version, approval, or exception identity drifted."""


class PolicyNotFoundError(PolicyDomainError):
    """An exact policy version or approval/exception record is missing."""


@runtime_checkable
class PolicyRepository(Protocol):
    def put(self, policy: PolicyVersion) -> None:
        """Store one immutable policy version."""

    def get(self, policy_id: str, version: str) -> PolicyVersion | None:
        """Read one exact policy version; aliases are not resolved."""


class InMemoryPolicyRepository:
    """Deterministic immutable policy catalog for local composition and tests."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._policies: dict[tuple[str, str], PolicyVersion] = {}

    def put(self, policy: PolicyVersion) -> None:
        key = (policy.policy_id, policy.version)
        with self._lock:
            prior = self._policies.get(key)
            if prior is not None and prior.policy_digest != policy.policy_digest:
                raise PolicyConflictError("immutable_policy_version_conflict")
            self._policies[key] = policy

    def get(self, policy_id: str, version: str) -> PolicyVersion | None:
        with self._lock:
            return self._policies.get((policy_id, version))


class ExactSignatureVerifier:
    """Fixture-friendly verifier keyed by exact signer and signed digest.

    Production callers should inject an Ed25519/OpenBao-backed verifier.  This
    class stores only opaque signature strings and never key material.
    """

    def __init__(self, signatures: Iterable[tuple[str, str, str]] = ()) -> None:
        self._signatures = {
            (signer_ref, signed_digest): signature
            for signer_ref, signed_digest, signature in signatures
        }

    def verify(self, signer_ref: str, signed_digest: str, signature: str) -> bool:
        return self._signatures.get((signer_ref, signed_digest)) == signature


class PolicyAuthority:
    """Resolve exact policy versions and validate bounded approvals/exceptions."""

    def __init__(
        self,
        repository: PolicyRepository,
        *,
        verifier: SignatureVerifier,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self.repository = repository
        self.verifier = verifier
        self.clock = clock or (lambda: 0)
        self._lock = threading.RLock()
        self._approvals: dict[tuple[str, str], ApprovalDecision] = {}
        self._exceptions: dict[tuple[str, str], PolicyException] = {}

    def publish(self, policy: PolicyVersion) -> PolicyRef:
        self.repository.put(policy)
        return policy.ref

    def resolve(self, reference: PolicyRef) -> PolicyVersion:
        policy = self.repository.get(reference.policy_id, reference.version)
        if policy is None:
            raise PolicyNotFoundError("policy_version_missing")
        if policy.policy_digest != reference.digest:
            raise PolicyConflictError("policy_digest_mismatch")
        return policy

    def record_approval(self, approval: ApprovalDecision) -> None:
        self._verify_approval_signature(approval)
        key = (approval.policy_ref.policy_id, approval.approval_id)
        with self._lock:
            prior = self._approvals.get(key)
            if prior is not None and prior != approval:
                raise PolicyConflictError("approval_identity_conflict")
            self._approvals[key] = approval

    def record_exception(self, exception: PolicyException) -> None:
        self._verify_exception_signature(exception)
        key = (exception.policy_ref.policy_id, exception.exception_id)
        with self._lock:
            prior = self._exceptions.get(key)
            if prior is not None and prior != exception:
                raise PolicyConflictError("exception_identity_conflict")
            self._exceptions[key] = exception

    def _resolve_rule_for(self, policy: PolicyVersion, operation: str) -> PolicyRule:
        """The policy's rule for ``operation``, or raise PolicyDomainError."""
        try:
            return policy.rule_for(operation)
        except KeyError as exc:
            raise PolicyDomainError("policy_operation_unresolved") from exc

    def _check_deny_gate(
        self,
        rule: PolicyRule,
        exception_ref: ExceptionRef | None,
        exception: PolicyException | None,
    ) -> None:
        """Raise if the rule denies the operation and no valid exception overrides it."""
        if rule.effect == "deny" and (
            exception_ref is None
            or exception is None
            or not exception.allow_denied_operation
        ):
            raise PolicyDomainError("policy_denied")

    def _record_if_present(
        self,
        approval: ApprovalDecision | None,
        exception: PolicyException | None,
    ) -> None:
        if approval is not None:
            self.record_approval(approval)
        if exception is not None:
            self.record_exception(exception)

    def authorize(
        self,
        *,
        policy_ref: PolicyRef,
        operation: str,
        request_digest: str,
        bindings: tuple[CapabilityBinding, ...],
        requested_budget: ExecutionBudget,
        approval: ApprovalDecision | None = None,
        exception: PolicyException | None = None,
        now: int | None = None,
    ) -> PolicyAuthorization:
        if not isinstance(request_digest, str) or not _DIGEST_RE.fullmatch(
            request_digest
        ):
            raise PolicyDomainError("request_digest_invalid")
        policy = self.resolve(policy_ref)
        rule = self._resolve_rule_for(policy, operation)
        if not _budget_within(requested_budget, rule.budget):
            raise PolicyDomainError("budget_escalation")
        self._validate_bindings(policy, bindings)

        current = self.clock() if now is None else now
        exception_ref = self._validate_exception(
            exception,
            policy=policy,
            operation=operation,
            request_digest=request_digest,
            requested_budget=requested_budget,
            current=current,
        )
        self._check_deny_gate(rule, exception_ref, exception)

        requires_approval = rule.approval_required or policy.approval.required
        approval_ref = self._validate_approval(
            approval,
            policy=policy,
            operation=operation,
            request_digest=request_digest,
            required=requires_approval,
            current=current,
        )
        self._record_if_present(approval, exception)
        return PolicyAuthorization(
            policy_ref=policy.ref,
            operation=operation,
            request_digest=request_digest,
            bindings=bindings,
            budget=requested_budget,
            approval_ref=approval_ref,
            exception_ref=exception_ref,
            resolved_at=current,
        )

    def _resolve_recorded_approval(
        self, authorization: PolicyAuthorization
    ) -> ApprovalDecision | None:
        """Look up + verify the recorded approval matches ``authorization.approval_ref``."""
        if authorization.approval_ref is None:
            return None
        with self._lock:
            approval = self._approvals.get(
                (
                    authorization.policy_ref.policy_id,
                    authorization.approval_ref.approval_id,
                )
            )
        if (
            approval is None
            or approval.decision_digest != authorization.approval_ref.decision_digest
        ):
            raise PolicyConflictError("approval_reference_drift")
        return approval

    def _resolve_recorded_exception(
        self, authorization: PolicyAuthorization
    ) -> PolicyException | None:
        """Look up + verify the recorded exception matches ``authorization.exception_ref``."""
        if authorization.exception_ref is None:
            return None
        with self._lock:
            exception = self._exceptions.get(
                (
                    authorization.policy_ref.policy_id,
                    authorization.exception_ref.exception_id,
                )
            )
        if (
            exception is None
            or exception.exception_digest
            != authorization.exception_ref.exception_digest
        ):
            raise PolicyConflictError("exception_reference_drift")
        return exception

    def verify_authorization(
        self,
        authorization: PolicyAuthorization,
        *,
        now: int | None = None,
    ) -> None:
        """Revalidate persisted authority without resolving a newer policy."""

        policy = self.resolve(authorization.policy_ref)
        rule = self._resolve_rule_for(policy, authorization.operation)
        if not _budget_within(authorization.budget, rule.budget):
            raise PolicyDomainError("budget_escalation")
        self._validate_bindings(policy, authorization.bindings)
        current = self.clock() if now is None else now

        approval = self._resolve_recorded_approval(authorization)
        exception = self._resolve_recorded_exception(authorization)

        self._validate_exception(
            exception,
            policy=policy,
            operation=authorization.operation,
            request_digest=authorization.request_digest,
            requested_budget=authorization.budget,
            current=current,
        )
        if rule.effect == "deny" and (
            exception is None or not exception.allow_denied_operation
        ):
            raise PolicyDomainError("policy_denied")
        self._validate_approval(
            approval,
            policy=policy,
            operation=authorization.operation,
            request_digest=authorization.request_digest,
            required=rule.approval_required or policy.approval.required,
            current=current,
        )

    def _validate_bindings(
        self,
        policy: PolicyVersion,
        bindings: tuple[CapabilityBinding, ...],
    ) -> None:
        allowed = {
            (
                binding.kind,
                binding.binding_id,
                binding.version,
                binding.digest,
                binding.privilege,
            )
            for binding in policy.bindings
        }
        for binding in bindings:
            if (
                binding.kind,
                binding.binding_id,
                binding.version,
                binding.digest,
                binding.privilege,
            ) not in allowed:
                raise PolicyDomainError("binding_unresolved_or_privilege_escalated")

    def _check_approval_identity(
        self,
        approval: ApprovalDecision,
        policy: PolicyVersion,
        operation: str,
        request_digest: str,
    ) -> None:
        if approval.policy_ref != policy.ref:
            raise PolicyConflictError("approval_policy_reference_mismatch")
        if approval.operation != operation or approval.request_digest != request_digest:
            raise PolicyConflictError("approval_request_drift")

    def _check_approval_freshness(
        self, approval: ApprovalDecision, policy: PolicyVersion, current: int
    ) -> None:
        if approval.decision != "approved":
            raise PolicyDomainError("approval_denied")
        if current < approval.issued_at or current >= approval.expires_at:
            raise PolicyDomainError("approval_expired_or_not_yet_valid")
        if approval.expires_at - approval.issued_at > policy.approval.max_age_seconds:
            raise PolicyDomainError("approval_window_exceeded")
        if len(approval.approver_refs) < policy.approval.min_approvers:
            raise PolicyDomainError("approval_quorum_missing")

    def _validate_approval(
        self,
        approval: ApprovalDecision | None,
        *,
        policy: PolicyVersion,
        operation: str,
        request_digest: str,
        required: bool,
        current: int,
    ) -> ApprovalRef | None:
        if approval is None:
            if required:
                raise PolicyDomainError("approval_required")
            return None
        self._check_approval_identity(approval, policy, operation, request_digest)
        self._check_approval_freshness(approval, policy, current)
        self._verify_approval_signature(approval)
        return approval.ref

    def _validate_exception(
        self,
        exception: PolicyException | None,
        *,
        policy: PolicyVersion,
        operation: str,
        request_digest: str,
        requested_budget: ExecutionBudget,
        current: int,
    ) -> ExceptionRef | None:
        if exception is None:
            return None
        if exception.policy_ref != policy.ref:
            raise PolicyConflictError("exception_policy_reference_mismatch")
        if (
            exception.operation != operation
            or exception.request_digest != request_digest
        ):
            raise PolicyConflictError("exception_request_drift")
        if current < exception.issued_at or current >= exception.expires_at:
            raise PolicyDomainError("exception_expired_or_not_yet_valid")
        if exception.expires_at - exception.issued_at > 86_400:
            raise PolicyDomainError("exception_window_exceeded")
        if not _budget_within(requested_budget, exception.budget):
            raise PolicyDomainError("exception_budget_escalation")
        self._verify_exception_signature(exception)
        return exception.ref

    def _verify_approval_signature(self, approval: ApprovalDecision) -> None:
        if not self.verifier.verify(
            approval.signer_ref,
            approval.decision_digest,
            approval.signature,
        ):
            raise PolicyDomainError("approval_signature_invalid")

    def _verify_exception_signature(self, exception: PolicyException) -> None:
        if not self.verifier.verify(
            exception.signer_ref,
            exception.exception_digest,
            exception.signature,
        ):
            raise PolicyDomainError("exception_signature_invalid")


def _budget_within(requested: ExecutionBudget, ceiling: ExecutionBudget) -> bool:
    return all(
        getattr(requested, name) <= getattr(ceiling, name)
        for name in (
            "max_tasks",
            "max_depth",
            "max_fanout",
            "max_tool_calls",
            "max_tokens",
            "max_seconds",
            "max_cost_micros",
        )
    )
