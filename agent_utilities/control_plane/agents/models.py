"""Typed agent-plane release and resolution contracts.

This module is a policy boundary, not an execution engine.  It describes
stable agent identities, digest-pinned releases, exact artifact bindings,
release pointers, approvals, policy ceilings, evaluation evidence, and
privacy-safe read projections.  Runtime dispatch remains owned by the
existing orchestration path.

(CONCEPT:AU-AHE.harness.unified-artifact-lineage,
AU-ORCH.dispatch.kg-governed-agent-swarm,
AU-OS.config.desired-state-fleet-reconciler)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

from agent_utilities.protocols.epistemic_operations import ProtocolModel

Identifier: TypeAlias = Annotated[
    str,
    Field(
        min_length=1,
        max_length=192,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$",
    ),
]
Digest: TypeAlias = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
VersionText: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:+-]*$"),
]
Timestamp: TypeAlias = Annotated[
    str,
    Field(
        min_length=20,
        max_length=64,
        pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|z|[+-][0-9]{2}:[0-9]{2})$",
    ),
]

BindingKind = Literal[
    "configuration", "model", "prompt", "skill", "tool", "connector", "workflow"
]
ReleaseTrack = Literal["stable", "beta", "edge"]
ApprovalOutcome = Literal["pending", "approved", "denied", "revoked"]
ReleaseOperation = Literal["promote", "rollback"]

_MAX_BINDINGS = 256
_MAX_DEPENDENCIES = 64
_MAX_GRANTS = 64
_MAX_PAGE_SIZE = 100

_FORBIDDEN_REF_PREFIXES = (
    "http:",
    "https:",
    "file:",
    "env:",
    "secret:",
    "vault:",
    "body:",
    "result:",
    "data:",
    "base64:",
)
_INLINE_MARKERS = (
    "password=",
    "secret=",
    "token=",
    "authorization:",
    "-----begin",
    '{"',
    "[{",
)


def _canonical(value: str) -> str:
    return str(value).strip().casefold()


def _digest_payload(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _opaque_ref(value: str, field_name: str) -> str:
    lowered = value.casefold()
    if lowered.startswith(_FORBIDDEN_REF_PREFIXES) or any(
        marker in lowered for marker in _INLINE_MARKERS
    ):
        raise ValueError(f"{field_name} must be an opaque controlled reference")
    return value


def agent_id_for(publisher: str, package_name: str, agent_name: str) -> str:
    """Return a stable identity independent of tenant, host, or runtime state."""

    payload = "\x1f".join(
        (_canonical(publisher), _canonical(package_name), _canonical(agent_name))
    )
    return "agent:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def binding_id_for(
    kind: BindingKind,
    logical_id: str,
    version: str,
    artifact_digest: str,
    schema_digest: str,
) -> str:
    payload = "\x1f".join(
        (
            _canonical(kind),
            _canonical(logical_id),
            version,
            artifact_digest,
            schema_digest,
        )
    )
    return "agent-binding:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pointer_id_for(agent_id: str, channel: ReleaseTrack) -> str:
    return (
        "agent-pointer:"
        + hashlib.sha256(
            "\x1f".join((_canonical(agent_id), channel)).encode("utf-8")
        ).hexdigest()
    )


def pointer_digest_for(
    agent_id: str,
    channel: ReleaseTrack,
    version_id: str,
    revision: int,
    previous_version_id: str | None,
) -> str:
    return _digest_payload(
        {
            "agent_id": agent_id,
            "channel": channel,
            "version_id": version_id,
            "revision": revision,
            "previous_version_id": previous_version_id,
        }
    )


class AgentIdentity(ProtocolModel):
    """Stable package identity for one named agent."""

    identity_version: Literal["agent-identity.v1"]
    agent_id: Identifier
    publisher: Identifier
    package_name: Identifier
    agent_name: Identifier

    @model_validator(mode="after")
    def id_is_derived(self) -> AgentIdentity:
        if self.agent_id != agent_id_for(
            self.publisher, self.package_name, self.agent_name
        ):
            raise ValueError("agent id is not derived from its identity fields")
        return self


class ArtifactBinding(ProtocolModel):
    """One exact, digest-pinned configuration/model/prompt/capability binding."""

    binding_version: Literal["agent-artifact-binding.v1"]
    binding_id: Identifier
    kind: BindingKind
    logical_id: Identifier
    version: VersionText
    artifact_ref: Identifier
    artifact_digest: Digest
    schema_digest: Digest
    depends_on: tuple[Identifier, ...] = Field(default=(), max_length=_MAX_DEPENDENCIES)

    @model_validator(mode="after")
    def binding_is_exact(self) -> ArtifactBinding:
        _opaque_ref(self.artifact_ref, "artifact_ref")
        expected = binding_id_for(
            self.kind,
            self.logical_id,
            self.version,
            self.artifact_digest,
            self.schema_digest,
        )
        if self.binding_id != expected:
            raise ValueError("agent binding id is not content-derived")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("agent binding dependencies must be unique")
        if tuple(sorted(self.depends_on)) != self.depends_on:
            raise ValueError("agent binding dependencies must be sorted")
        if self.binding_id in self.depends_on:
            raise ValueError("agent binding cannot depend on itself")
        return self


class AgentBindings(ProtocolModel):
    """Normalized exact binding set for one agent release."""

    bindings_version: Literal["agent-bindings.v1"]
    configuration: ArtifactBinding
    model: ArtifactBinding
    prompt: ArtifactBinding
    skills: tuple[ArtifactBinding, ...] = Field(default=(), max_length=_MAX_BINDINGS)
    tools: tuple[ArtifactBinding, ...] = Field(default=(), max_length=_MAX_BINDINGS)
    connectors: tuple[ArtifactBinding, ...] = Field(
        default=(), max_length=_MAX_BINDINGS
    )
    workflows: tuple[ArtifactBinding, ...] = Field(default=(), max_length=_MAX_BINDINGS)

    @property
    def all_bindings(self) -> tuple[ArtifactBinding, ...]:
        values = (
            self.configuration,
            self.model,
            self.prompt,
            *self.skills,
            *self.tools,
            *self.connectors,
            *self.workflows,
        )
        return tuple(sorted(values, key=lambda item: item.binding_id))

    @property
    def binding_ids(self) -> tuple[str, ...]:
        return tuple(item.binding_id for item in self.all_bindings)

    @model_validator(mode="after")
    def bindings_are_normalized_and_acyclic(self) -> AgentBindings:
        groups = (
            ("configuration", (self.configuration,)),
            ("model", (self.model,)),
            ("prompt", (self.prompt,)),
            ("skill", self.skills),
            ("tool", self.tools),
            ("connector", self.connectors),
            ("workflow", self.workflows),
        )
        all_items = self.all_bindings
        ids = [item.binding_id for item in all_items]
        if len(set(ids)) != len(ids):
            raise ValueError("agent binding identities must be unique")
        for expected_kind, items in groups:
            if any(item.kind != expected_kind for item in items):
                raise ValueError("agent binding kind does not match its binding slot")
            logical_ids = [item.logical_id for item in items]
            if len(set(logical_ids)) != len(logical_ids):
                raise ValueError(
                    "agent binding logical ids have an alias/version ambiguity"
                )
            if tuple(sorted(items, key=lambda item: item.binding_id)) != items:
                raise ValueError("agent binding groups must be sorted by binding id")

        known = set(ids)
        dependencies = {item.binding_id: set(item.depends_on) for item in all_items}
        if any(
            dependency not in known
            for values in dependencies.values()
            for dependency in values
        ):
            raise ValueError("agent binding dependency references an unknown binding")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(binding_id: str) -> None:
            if binding_id in visiting:
                raise ValueError("agent binding dependency graph contains a cycle")
            if binding_id in visited:
                return
            visiting.add(binding_id)
            for dependency in dependencies[binding_id]:
                visit(dependency)
            visiting.remove(binding_id)
            visited.add(binding_id)

        for binding_id in ids:
            visit(binding_id)
        return self


def binding_set_digest(bindings: AgentBindings) -> str:
    return _digest_payload(
        [item.model_dump(mode="json") for item in bindings.all_bindings]
    )


class TeamPolicyRef(ProtocolModel):
    """Bounded team policy reference; policy bodies stay in the authority."""

    policy_version: Literal["agent-team-policy-ref.v1"]
    policy_ref: Identifier
    policy_digest: Digest
    team_ref: Identifier
    max_members: int = Field(ge=1, le=_MAX_BINDINGS)
    allowed_binding_ids: tuple[Identifier, ...] = Field(
        default=(), max_length=_MAX_BINDINGS
    )

    @model_validator(mode="after")
    def policy_ref_is_opaque(self) -> TeamPolicyRef:
        _opaque_ref(self.policy_ref, "policy_ref")
        _opaque_ref(self.team_ref, "team_ref")
        if len(set(self.allowed_binding_ids)) != len(self.allowed_binding_ids):
            raise ValueError("team policy capability allow-list must be unique")
        if tuple(sorted(self.allowed_binding_ids)) != self.allowed_binding_ids:
            raise ValueError("team policy capability allow-list must be sorted")
        return self


class DelegationPolicyRef(ProtocolModel):
    """Delegation ceiling reference used before any dispatch can be planned."""

    policy_version: Literal["agent-delegation-policy-ref.v1"]
    policy_ref: Identifier
    policy_digest: Digest
    max_depth: int = Field(ge=0, le=64)
    max_children: int = Field(ge=0, le=_MAX_BINDINGS)
    max_budget_micros: int = Field(ge=0, le=10**12)

    @model_validator(mode="after")
    def policy_ref_is_opaque(self) -> DelegationPolicyRef:
        _opaque_ref(self.policy_ref, "policy_ref")
        return self


class BudgetPolicyRef(ProtocolModel):
    """Token, cost, and wall-time ceilings referenced by a release."""

    policy_version: Literal["agent-budget-policy-ref.v1"]
    policy_ref: Identifier
    policy_digest: Digest
    max_tokens: int = Field(ge=0, le=10**9)
    max_cost_micros: int = Field(ge=0, le=10**12)
    max_time_ms: int = Field(ge=0, le=10**9)

    @model_validator(mode="after")
    def policy_ref_is_opaque(self) -> BudgetPolicyRef:
        _opaque_ref(self.policy_ref, "policy_ref")
        return self


def policy_set_digest_for(
    team: TeamPolicyRef,
    delegation: DelegationPolicyRef,
    budget: BudgetPolicyRef,
) -> str:
    return _digest_payload(
        {
            "team": team.model_dump(mode="json"),
            "delegation": delegation.model_dump(mode="json"),
            "budget": budget.model_dump(mode="json"),
        }
    )


class PolicySet(ProtocolModel):
    """Immutable policy references and bounded ceilings for one release."""

    policies_version: Literal["agent-policy-set.v1"]
    team: TeamPolicyRef
    delegation: DelegationPolicyRef
    budget: BudgetPolicyRef
    policy_set_digest: Digest

    @model_validator(mode="after")
    def policy_set_is_self_consistent(self) -> PolicySet:
        expected = policy_set_digest_for(self.team, self.delegation, self.budget)
        if self.policy_set_digest != expected:
            raise ValueError("agent policy-set digest does not match its references")
        return self


class EvaluationDatasetRef(ProtocolModel):
    """Reference to an immutable evaluation dataset artifact, never its rows."""

    artifact_version: Literal["agent-evaluation-dataset-ref.v1"]
    dataset_ref: Identifier
    dataset_version: VersionText
    dataset_digest: Digest
    split: Literal["train", "validation", "test", "held_out"]

    @model_validator(mode="after")
    def dataset_ref_is_opaque(self) -> EvaluationDatasetRef:
        _opaque_ref(self.dataset_ref, "dataset_ref")
        return self


class EvaluationRunRef(ProtocolModel):
    """Reference to a completed evaluation run artifact, never its results."""

    artifact_version: Literal["agent-evaluation-run-ref.v1"]
    run_ref: Identifier
    run_version: VersionText
    run_digest: Digest
    dataset_digest: Digest
    outcome: Literal["passed"]

    @model_validator(mode="after")
    def run_ref_is_opaque(self) -> EvaluationRunRef:
        _opaque_ref(self.run_ref, "run_ref")
        return self


def evaluation_evidence_digest_for(
    datasets: Iterable[EvaluationDatasetRef], runs: Iterable[EvaluationRunRef]
) -> str:
    return _digest_payload(
        {
            "datasets": [
                item.model_dump(mode="json")
                for item in sorted(datasets, key=lambda item: item.dataset_ref)
            ],
            "runs": [
                item.model_dump(mode="json")
                for item in sorted(runs, key=lambda item: item.run_ref)
            ],
        }
    )


class EvaluationEvidence(ProtocolModel):
    """Bounded dataset/run references required for release admission."""

    evidence_version: Literal["agent-evaluation-evidence.v1"]
    datasets: tuple[EvaluationDatasetRef, ...] = Field(min_length=1, max_length=64)
    runs: tuple[EvaluationRunRef, ...] = Field(min_length=1, max_length=64)
    evidence_digest: Digest

    @model_validator(mode="after")
    def evidence_is_complete(self) -> EvaluationEvidence:
        if len({item.dataset_ref for item in self.datasets}) != len(self.datasets):
            raise ValueError("evaluation dataset references must be unique")
        if len({item.run_ref for item in self.runs}) != len(self.runs):
            raise ValueError("evaluation run references must be unique")
        if (
            tuple(sorted(self.datasets, key=lambda item: item.dataset_ref))
            != self.datasets
        ):
            raise ValueError("evaluation dataset references must be sorted")
        if tuple(sorted(self.runs, key=lambda item: item.run_ref)) != self.runs:
            raise ValueError("evaluation run references must be sorted")
        dataset_digests = {item.dataset_digest for item in self.datasets}
        if any(item.dataset_digest not in dataset_digests for item in self.runs):
            raise ValueError("evaluation run references an unknown dataset artifact")
        expected = evaluation_evidence_digest_for(self.datasets, self.runs)
        if self.evidence_digest != expected:
            raise ValueError("evaluation evidence digest does not match its references")
        return self


class AgentVersion(ProtocolModel):
    """Immutable agent release with exact bindings, policies, and evaluation proof."""

    version_record_version: Literal["agent-version.v1"]
    version_id: Identifier
    agent_id: Identifier
    version: VersionText
    bindings: AgentBindings
    policies: PolicySet
    evaluation: EvaluationEvidence

    @model_validator(mode="after")
    def version_is_content_addressed(self) -> AgentVersion:
        binding_ids = set(self.bindings.binding_ids)
        if any(
            binding_id not in binding_ids
            for binding_id in self.policies.team.allowed_binding_ids
        ):
            raise ValueError("team policy references an unknown agent binding")
        expected = agent_version_id_for(
            self.agent_id,
            self.version,
            binding_set_digest(self.bindings),
            self.policies.policy_set_digest,
            self.evaluation.evidence_digest,
        )
        if self.version_id != expected:
            raise ValueError("agent version id is not derived from its release content")
        return self


def agent_version_id_for(
    agent_id: str,
    version: str,
    bindings_digest: str,
    policy_digest: str,
    evaluation_digest: str,
) -> str:
    payload = "\x1f".join(
        (agent_id, version, bindings_digest, policy_digest, evaluation_digest)
    )
    return "agent-version:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AgentReleasePointer(ProtocolModel):
    """CAS-managed channel pointer; one pointer exists per agent/channel key."""

    pointer_version: Literal["agent-release-pointer.v1"]
    pointer_id: Identifier
    agent_id: Identifier
    channel: ReleaseTrack
    version_id: Identifier
    revision: int = Field(ge=1)
    previous_version_id: Identifier | None = None
    pointer_digest: Digest

    @model_validator(mode="after")
    def pointer_is_content_addressed(self) -> AgentReleasePointer:
        if self.pointer_id != pointer_id_for(self.agent_id, self.channel):
            raise ValueError("agent release pointer id is not stable")
        if self.previous_version_id == self.version_id:
            raise ValueError("agent release pointer rollback target is unchanged")
        expected = pointer_digest_for(
            self.agent_id,
            self.channel,
            self.version_id,
            self.revision,
            self.previous_version_id,
        )
        if self.pointer_digest != expected:
            raise ValueError("agent release pointer digest does not match its state")
        return self


class ReleaseMutation(ProtocolModel):
    """Explicit CAS request for promotion or rollback; never an implicit move."""

    mutation_version: Literal["agent-release-mutation.v1"]
    agent_id: Identifier
    channel: ReleaseTrack
    operation: ReleaseOperation
    expected_revision: int = Field(ge=0)
    expected_version_id: Identifier | None = None
    next_version_id: Identifier
    change_ref: Identifier

    @model_validator(mode="after")
    def refs_are_opaque(self) -> ReleaseMutation:
        _opaque_ref(self.change_ref, "change_ref")
        return self


class ApprovalRecord(ProtocolModel):
    """Version/policy/evaluation-bound approval required before resolution."""

    approval_version: Literal["agent-approval.v1"]
    approval_id: Identifier
    agent_id: Identifier
    version_id: Identifier
    channel: ReleaseTrack
    policy_set_digest: Digest
    evaluation_digest: Digest
    outcome: ApprovalOutcome
    approved_at: Timestamp
    expires_at: Timestamp
    approver_ref: Identifier
    evidence_ref: Identifier
    approval_digest: Digest

    @model_validator(mode="after")
    def approval_is_opaque_and_bound(self) -> ApprovalRecord:
        _opaque_ref(self.approval_id, "approval_id")
        _opaque_ref(self.approver_ref, "approver_ref")
        _opaque_ref(self.evidence_ref, "evidence_ref")
        expected = _digest_payload(
            {
                "approval_id": self.approval_id,
                "agent_id": self.agent_id,
                "version_id": self.version_id,
                "channel": self.channel,
                "policy_set_digest": self.policy_set_digest,
                "evaluation_digest": self.evaluation_digest,
                "outcome": self.outcome,
                "approved_at": self.approved_at,
                "expires_at": self.expires_at,
                "approver_ref": self.approver_ref,
                "evidence_ref": self.evidence_ref,
            }
        )
        if self.approval_digest != expected:
            raise ValueError("approval digest does not match its signed decision")
        return self


def scope_digest_for(
    tenant_id: str, principal_id: str, grant_digests: Iterable[str]
) -> str:
    return _digest_payload(
        {
            "tenant_id": tenant_id,
            "principal_id": principal_id,
            "grant_digests": sorted(set(grant_digests)),
        }
    )


class AccessScope(ProtocolModel):
    """Tenant/principal/grant boundary for all reads and resolution."""

    scope_version: Literal["agent-access-scope.v1"]
    tenant_id: Identifier
    principal_id: Identifier
    grant_digests: tuple[Digest, ...] = Field(default=(), max_length=_MAX_GRANTS)
    scope_digest: Digest

    @model_validator(mode="after")
    def scope_is_self_consistent(self) -> AccessScope:
        if len(set(self.grant_digests)) != len(self.grant_digests):
            raise ValueError("agent scope grants must be unique")
        if tuple(sorted(self.grant_digests)) != self.grant_digests:
            raise ValueError("agent scope grants must be sorted")
        if self.scope_digest != scope_digest_for(
            self.tenant_id, self.principal_id, self.grant_digests
        ):
            raise ValueError("agent scope digest does not match its subject")
        return self


class AgentResolutionRequest(ProtocolModel):
    """Pre-dispatch request containing only bounded references and budget asks."""

    request_version: Literal["agent-resolution-request.v1"]
    agent_id: Identifier
    scope: AccessScope
    channel: ReleaseTrack | None = None
    version_id: Identifier | None = None
    approval_id: Identifier
    requested_binding_ids: tuple[Identifier, ...] = Field(
        default=(), max_length=_MAX_BINDINGS
    )
    team_ref: Identifier | None = None
    team_member_count: int = Field(default=1, ge=1, le=_MAX_BINDINGS)
    delegation_depth: int = Field(default=0, ge=0, le=64)
    delegation_children: int = Field(default=0, ge=0, le=_MAX_BINDINGS)
    requested_tokens: int = Field(default=0, ge=0, le=10**9)
    requested_cost_micros: int = Field(default=0, ge=0, le=10**12)
    requested_time_ms: int = Field(default=0, ge=0, le=10**9)
    at: Timestamp

    @model_validator(mode="after")
    def selector_and_capabilities_are_unambiguous(self) -> AgentResolutionRequest:
        if (self.channel is None) == (self.version_id is None):
            raise ValueError("resolution must select exactly one channel or version")
        _opaque_ref(self.approval_id, "approval_id")
        if self.team_ref is not None:
            _opaque_ref(self.team_ref, "team_ref")
        if len(set(self.requested_binding_ids)) != len(self.requested_binding_ids):
            raise ValueError("requested agent bindings must be unique")
        if tuple(sorted(self.requested_binding_ids)) != self.requested_binding_ids:
            raise ValueError("requested agent bindings must be sorted")
        return self


def resolution_digest_for(value: Mapping[str, object]) -> str:
    """Return the deterministic digest of a fully validated resolution input."""

    return _digest_payload(dict(value))


class ResolvedAgentPlan(ProtocolModel):
    """Privacy-safe, dispatch-neutral result of pre-dispatch resolution."""

    plan_version: Literal["resolved-agent-plan.v1"]
    agent_id: Identifier
    version_id: Identifier
    channel: ReleaseTrack
    pointer_digest: Digest
    approval_id: Identifier
    binding_ids: tuple[Identifier, ...] = Field(min_length=3, max_length=_MAX_BINDINGS)
    policy_set_digest: Digest
    evaluation_digest: Digest
    resolution_digest: Digest


class AgentGraphProjection(ProtocolModel):
    """Allow-listed graph projection with no bodies, results, or authority tokens."""

    projection_version: Literal["agent-graph-projection.v1"]
    tenant_id: Identifier
    agent_id: Identifier
    version_id: Identifier
    channel: ReleaseTrack
    pointer_revision: int = Field(ge=1)
    binding_count: int = Field(ge=3, le=_MAX_BINDINGS)
    binding_set_digest: Digest
    policy_set_digest: Digest
    evaluation_digest: Digest
    approval_state: Literal["missing", "approved", "stale", "denied"]


class AgentKeysetCursor(ProtocolModel):
    """Scope-bound keyset cursor; opaque transport tokens stay in adapters."""

    cursor_version: Literal["agent-keyset-cursor.v1"]
    scope_digest: Digest
    after_agent_id: Identifier
    after_version_id: Identifier


class AgentListRequest(ProtocolModel):
    """Bounded agent list request."""

    scope: AccessScope
    limit: int = Field(default=50, ge=1, le=_MAX_PAGE_SIZE)
    cursor: AgentKeysetCursor | None = None

    @model_validator(mode="after")
    def cursor_is_scope_bound(self) -> AgentListRequest:
        if (
            self.cursor is not None
            and self.cursor.scope_digest != self.scope.scope_digest
        ):
            raise ValueError("agent cursor is bound to another visibility scope")
        return self


class AgentPageItem(ProtocolModel):
    """Bounded privacy-safe summary for one released agent."""

    tenant_id: Identifier
    agent_id: Identifier
    version_id: Identifier
    channel: ReleaseTrack
    pointer_revision: int = Field(ge=1)
    binding_count: int = Field(ge=3, le=_MAX_BINDINGS)
    binding_set_digest: Digest
    policy_set_digest: Digest
    evaluation_digest: Digest
    approval_state: Literal["missing", "approved", "stale", "denied"]


class AgentPage(ProtocolModel):
    """Bounded page returned by a repository adapter."""

    page_version: Literal["agent-page.v1"]
    items: tuple[AgentPageItem, ...] = Field(max_length=_MAX_PAGE_SIZE)
    next_cursor: AgentKeysetCursor | None = None

    @model_validator(mode="after")
    def page_is_bounded(self) -> AgentPage:
        if self.next_cursor is not None and self.items == ():
            raise ValueError("an empty agent page cannot advance a cursor")
        return self


class AgentRegistration(ProtocolModel):
    """Atomic identity/version bundle passed to a repository adapter."""

    registration_version: Literal["agent-registration.v1"]
    identity: AgentIdentity
    version: AgentVersion

    @model_validator(mode="after")
    def registration_is_consistent(self) -> AgentRegistration:
        if self.version.agent_id != self.identity.agent_id:
            raise ValueError("agent release belongs to another identity")
        return self
