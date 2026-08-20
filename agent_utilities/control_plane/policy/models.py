"""Strict immutable policy and approval-domain models.

The policy plane stores references and digests, never credentials, policy
source text, tool arguments, request bodies, or execution results.  A policy
version is content-addressed; all capability bindings and approvals are exact
``id + semantic-version + digest`` pins.  Runtime execution is intentionally
outside this module.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Annotated, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "ABSOLUTE_MAX_COST_MICROS",
    "ABSOLUTE_MAX_DEPTH",
    "ABSOLUTE_MAX_SECONDS",
    "ABSOLUTE_MAX_TASKS",
    "ABSOLUTE_MAX_TOKENS",
    "ApprovalDecision",
    "ApprovalRef",
    "ApprovalRequirement",
    "CapabilityBinding",
    "ExecutionBudget",
    "ExceptionRef",
    "PolicyAuthorization",
    "PolicyDecision",
    "PolicyException",
    "PolicyRef",
    "PolicyRule",
    "PolicyVersion",
    "SignatureVerifier",
    "canonical_digest",
    "policy_digest",
]


ABSOLUTE_MAX_TASKS = 256
ABSOLUTE_MAX_DEPTH = 64
ABSOLUTE_MAX_TOKENS = 4_000_000
ABSOLUTE_MAX_SECONDS = 86_400
ABSOLUTE_MAX_COST_MICROS = 10_000_000

_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9:_./-]{0,127}$")
_REF_RE = re.compile(r"^[A-Za-z][A-Za-z0-9:_./-]{0,255}$")
_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._:/-]{0,255}$")
_VERSION_RE = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SIGNATURE_RE = re.compile(r"^sig:[A-Za-z0-9._:-]{3,255}$")
_SENSITIVE_TEXT_RE = re.compile(
    r"\b(?:argument|body|credential|password|payload|prompt|result|secret|token)\b",
    re.IGNORECASE,
)

StableId: TypeAlias = Annotated[str, Field(pattern=_ID_RE.pattern, min_length=1)]
OpaqueRef: TypeAlias = Annotated[str, Field(pattern=_REF_RE.pattern, min_length=1)]
StableKey: TypeAlias = Annotated[str, Field(pattern=_KEY_RE.pattern, min_length=1)]
Version: TypeAlias = Annotated[str, Field(pattern=_VERSION_RE.pattern)]
Digest: TypeAlias = Annotated[str, Field(pattern=_DIGEST_RE.pattern)]
Signature: TypeAlias = Annotated[str, Field(pattern=_SIGNATURE_RE.pattern)]
Timestamp: TypeAlias = Annotated[int, Field(ge=0)]
Privilege: TypeAlias = Literal["read", "write", "admin"]
BindingKind: TypeAlias = Literal["agent", "skill", "tool", "delegation"]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=False,
        strict=True,
    )


def _nonempty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name}_empty")
    if value != value.strip():
        raise ValueError(f"{field_name}_whitespace")
    return value


def _safe_summary(value: str) -> str:
    value = _nonempty(value, "summary")
    if len(value) > 256:
        raise ValueError("summary_too_long")
    if _SENSITIVE_TEXT_RE.search(value):
        raise ValueError("summary_sensitive_content")
    return value


def _canonical(value: object) -> object:
    if isinstance(value, BaseModel):
        return _canonical(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


def canonical_digest(value: object) -> str:
    payload = json.dumps(
        _canonical(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _within(requested: ExecutionBudget, ceiling: ExecutionBudget) -> bool:
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


class ExecutionBudget(_FrozenModel):
    """Hard-bounded requested or policy-approved execution ceiling."""

    max_tasks: int = Field(ge=1, le=ABSOLUTE_MAX_TASKS)
    max_depth: int = Field(ge=1, le=ABSOLUTE_MAX_DEPTH)
    max_fanout: int = Field(ge=1, le=ABSOLUTE_MAX_TASKS)
    max_tool_calls: int = Field(ge=0, le=ABSOLUTE_MAX_TASKS)
    max_tokens: int = Field(ge=1, le=ABSOLUTE_MAX_TOKENS)
    max_seconds: int = Field(ge=1, le=ABSOLUTE_MAX_SECONDS)
    max_cost_micros: int = Field(ge=0, le=ABSOLUTE_MAX_COST_MICROS)


class CapabilityBinding(_FrozenModel):
    """Exact versioned capability binding; no alias or executable content."""

    kind: BindingKind
    binding_id: StableId
    version: Version
    digest: Digest
    privilege: Privilege = "read"

    @field_validator("binding_id")
    @classmethod
    def _no_alias(cls, value: str) -> str:
        if value.casefold() in {"latest", "current", "default"} or "@" in value:
            raise ValueError("binding_alias_forbidden")
        return value


class PolicyRef(_FrozenModel):
    """Exact policy version pin carried through every resolved run."""

    policy_id: StableId
    version: Version
    digest: Digest


class ApprovalRequirement(_FrozenModel):
    """Policy-declared approval requirement."""

    required: bool = True
    min_approvers: int = Field(default=1, ge=0, le=16)
    max_age_seconds: int = Field(default=900, ge=1, le=86_400)
    approver_scope: OpaqueRef = "scope:operator"

    @model_validator(mode="after")
    def _required_has_approvers(self) -> ApprovalRequirement:
        if self.required and self.min_approvers < 1:
            raise ValueError("approval_approver_count_missing")
        if not self.required and self.min_approvers != 0:
            raise ValueError("optional_approval_approver_count_invalid")
        return self


class PolicyRule(_FrozenModel):
    """Structured operation rule; policy source text never enters the model."""

    operation: StableKey
    effect: Literal["allow", "deny"]
    approval_required: bool = True
    budget: ExecutionBudget


class PolicyVersion(_FrozenModel):
    """Immutable, content-addressed policy version and exact bindings."""

    policy_id: StableId
    version: Version
    issuer_ref: OpaqueRef
    budget: ExecutionBudget
    rules: tuple[PolicyRule, ...] = Field(min_length=1, max_length=256)
    bindings: tuple[CapabilityBinding, ...] = Field(default=(), max_length=256)
    max_privilege: Privilege = "read"
    approval: ApprovalRequirement = Field(default_factory=ApprovalRequirement)
    summary: str = "immutable policy version"

    @field_validator("summary")
    @classmethod
    def _summary_is_safe(cls, value: str) -> str:
        return _safe_summary(value)

    @model_validator(mode="after")
    def _validate_policy(self) -> PolicyVersion:
        rule_names = [rule.operation for rule in self.rules]
        if len(rule_names) != len(set(rule_names)):
            raise ValueError("policy_operation_ambiguous")
        for rule in self.rules:
            if not _within(rule.budget, self.budget):
                raise ValueError("policy_rule_budget_escalation")

        binding_keys = [
            (binding.kind, binding.binding_id, binding.version)
            for binding in self.bindings
        ]
        if len(binding_keys) != len(set(binding_keys)):
            raise ValueError("policy_binding_ambiguous")
        privilege_level = {"read": 0, "write": 1, "admin": 2}
        ceiling = privilege_level[self.max_privilege]
        if any(
            privilege_level[binding.privilege] > ceiling for binding in self.bindings
        ):
            raise ValueError("policy_privilege_escalation")
        return self

    @property
    def ref(self) -> PolicyRef:
        return PolicyRef(
            policy_id=self.policy_id,
            version=self.version,
            digest=self.policy_digest,
        )

    @property
    def policy_digest(self) -> str:
        return policy_digest(self)

    def rule_for(self, operation: str) -> PolicyRule:
        for rule in self.rules:
            if rule.operation == operation:
                return rule
        raise KeyError(operation)


PolicyDecision = Literal["approved", "denied"]


class ApprovalDecision(_FrozenModel):
    """Immutable, time-bounded approval over one exact request digest."""

    approval_id: StableId
    policy_ref: PolicyRef
    operation: StableKey
    request_digest: Digest
    decision: PolicyDecision
    approver_refs: tuple[OpaqueRef, ...] = Field(min_length=0, max_length=16)
    issued_at: Timestamp
    expires_at: Timestamp
    signer_ref: OpaqueRef
    signature: Signature

    @model_validator(mode="after")
    def _validate_decision(self) -> ApprovalDecision:
        if self.expires_at <= self.issued_at:
            raise ValueError("approval_expiry_invalid")
        if len(self.approver_refs) != len(set(self.approver_refs)):
            raise ValueError("approval_approver_duplicate")
        if self.decision == "approved" and not self.approver_refs:
            raise ValueError("approval_approver_missing")
        return self

    @property
    def decision_digest(self) -> str:
        return canonical_digest(
            self.model_dump(mode="json", exclude={"signature"}, exclude_none=True)
        )

    @property
    def ref(self) -> ApprovalRef:
        return ApprovalRef(
            approval_id=self.approval_id,
            decision_digest=self.decision_digest,
        )


class PolicyException(_FrozenModel):
    """Signed, bounded exception tied to one policy operation and request."""

    exception_id: StableId
    policy_ref: PolicyRef
    operation: StableKey
    request_digest: Digest
    scope_ref: OpaqueRef
    budget: ExecutionBudget
    allow_denied_operation: bool = False
    issued_at: Timestamp
    expires_at: Timestamp
    signer_ref: OpaqueRef
    signature: Signature

    @model_validator(mode="after")
    def _validate_expiry(self) -> PolicyException:
        if self.expires_at <= self.issued_at:
            raise ValueError("exception_expiry_invalid")
        return self

    @property
    def exception_digest(self) -> str:
        return canonical_digest(
            self.model_dump(mode="json", exclude={"signature"}, exclude_none=True)
        )

    @property
    def ref(self) -> ExceptionRef:
        return ExceptionRef(
            exception_id=self.exception_id,
            exception_digest=self.exception_digest,
        )


class ApprovalRef(_FrozenModel):
    approval_id: StableId
    decision_digest: Digest


class ExceptionRef(_FrozenModel):
    exception_id: StableId
    exception_digest: Digest


class PolicyAuthorization(_FrozenModel):
    """Exact admission proof embedded in a resolved run."""

    policy_ref: PolicyRef
    operation: StableKey
    request_digest: Digest
    bindings: tuple[CapabilityBinding, ...] = Field(max_length=256)
    budget: ExecutionBudget
    approval_ref: ApprovalRef | None = None
    exception_ref: ExceptionRef | None = None
    resolved_at: Timestamp

    @property
    def authorization_digest(self) -> str:
        return canonical_digest(self)


class SignatureVerifier(Protocol):
    """External signature verification seam; key material never enters models."""

    def verify(self, signer_ref: str, signed_digest: str, signature: str) -> bool:
        """Return true only when the exact signature authenticates the digest."""


def policy_digest(policy: PolicyVersion) -> str:
    """Deterministic digest of all policy rules, budgets, and exact bindings."""

    return canonical_digest(policy.model_dump(mode="json", exclude_none=True))
