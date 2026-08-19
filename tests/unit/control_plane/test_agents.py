"""Focused NE-085 agent-plane contract fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from agent_utilities.control_plane.agents import (
    AccessScope,
    AgentBindings,
    AgentControlPlane,
    AgentIdentity,
    AgentKeysetCursor,
    AgentListRequest,
    AgentPage,
    AgentPageItem,
    AgentRegistration,
    AgentReleasePointer,
    AgentResolutionError,
    AgentResolutionRequest,
    AgentVersion,
    ApprovalRecord,
    ArtifactBinding,
    BindingKind,
    BudgetPolicyRef,
    DelegationPolicyRef,
    EvaluationDatasetRef,
    EvaluationEvidence,
    EvaluationRunRef,
    PolicySet,
    ReleaseMutation,
    RepositoryContractError,
    TeamPolicyRef,
    agent_id_for,
    agent_version_id_for,
    binding_id_for,
    binding_set_digest,
    evaluation_evidence_digest_for,
    pointer_digest_for,
    pointer_id_for,
    policy_set_digest_for,
    scope_digest_for,
)


def _digest(char: str) -> str:
    return "sha256:" + char * 64


def _binding(
    kind: BindingKind,
    logical_id: str,
    version: str = "1.0.0",
    *,
    depends_on: tuple[str, ...] = (),
) -> ArtifactBinding:
    artifact_digest = _digest("a")
    schema_digest = _digest("b")
    return ArtifactBinding(
        binding_version="agent-artifact-binding.v1",
        binding_id=binding_id_for(
            kind, logical_id, version, artifact_digest, schema_digest
        ),
        kind=kind,
        logical_id=logical_id,
        version=version,
        artifact_ref=f"artifact:{kind}:{logical_id}",
        artifact_digest=artifact_digest,
        schema_digest=schema_digest,
        depends_on=depends_on,
    )


def _bindings() -> AgentBindings:
    return AgentBindings(
        bindings_version="agent-bindings.v1",
        configuration=_binding("configuration", "agent-config"),
        model=_binding("model", "reasoning-model"),
        prompt=_binding("prompt", "agent-system"),
        skills=(_binding("skill", "research"),),
        tools=(_binding("tool", "graph.query"),),
        connectors=(_binding("connector", "gitlab"),),
        workflows=(_binding("workflow", "research-flow"),),
    )


def _policies() -> PolicySet:
    team = TeamPolicyRef(
        policy_version="agent-team-policy-ref.v1",
        policy_ref="policy:team:research",
        policy_digest=_digest("c"),
        team_ref="team:research",
        max_members=4,
    )
    delegation = DelegationPolicyRef(
        policy_version="agent-delegation-policy-ref.v1",
        policy_ref="policy:delegation:research",
        policy_digest=_digest("d"),
        max_depth=2,
        max_children=4,
        max_budget_micros=100,
    )
    budget = BudgetPolicyRef(
        policy_version="agent-budget-policy-ref.v1",
        policy_ref="policy:budget:research",
        policy_digest=_digest("e"),
        max_tokens=10_000,
        max_cost_micros=100,
        max_time_ms=60_000,
    )
    return PolicySet(
        policies_version="agent-policy-set.v1",
        team=team,
        delegation=delegation,
        budget=budget,
        policy_set_digest=policy_set_digest_for(team, delegation, budget),
    )


def _evaluation() -> EvaluationEvidence:
    dataset = EvaluationDatasetRef(
        artifact_version="agent-evaluation-dataset-ref.v1",
        dataset_ref="dataset:research-heldout",
        dataset_version="1.0.0",
        dataset_digest=_digest("f"),
        split="held_out",
    )
    run = EvaluationRunRef(
        artifact_version="agent-evaluation-run-ref.v1",
        run_ref="run:research:1",
        run_version="1.0.0",
        run_digest=_digest("0"),
        dataset_digest=dataset.dataset_digest,
        outcome="passed",
    )
    return EvaluationEvidence(
        evidence_version="agent-evaluation-evidence.v1",
        datasets=(dataset,),
        runs=(run,),
        evidence_digest=evaluation_evidence_digest_for((dataset,), (run,)),
    )


@dataclass
class _MemoryRepository:
    identities: dict[str, AgentIdentity] = field(default_factory=dict)
    versions: dict[str, AgentVersion] = field(default_factory=dict)
    pointers: dict[tuple[str, str], AgentReleasePointer] = field(default_factory=dict)
    approvals: dict[tuple[str, str, str], ApprovalRecord] = field(default_factory=dict)
    page: AgentPage | None = None

    def put_registration(self, registration: AgentRegistration) -> None:
        self.put_identity(registration.identity)
        self.put_version(registration.version)

    def put_identity(self, identity: AgentIdentity) -> None:
        self.identities[identity.agent_id] = identity

    def put_version(self, version: AgentVersion) -> None:
        self.versions[version.version_id] = version

    def get_version(self, scope: AccessScope, version_id: str) -> AgentVersion | None:
        del scope
        return self.versions.get(version_id)

    def get_release_pointer(
        self, scope: AccessScope, agent_id: str, channel: str
    ) -> AgentReleasePointer | None:
        del scope
        return self.pointers.get((agent_id, channel))

    def get_release_pointer_for_version(
        self, scope: AccessScope, agent_id: str, version_id: str
    ) -> AgentReleasePointer | None:
        del scope
        matches = [
            pointer
            for (pointer_agent, _channel), pointer in self.pointers.items()
            if pointer_agent == agent_id and pointer.version_id == version_id
        ]
        if len(matches) > 1:
            raise RepositoryContractError("ambiguous active release pointer")
        return matches[0] if matches else None

    def compare_and_swap_release(
        self,
        scope: AccessScope,
        mutation: ReleaseMutation,
        pointer: AgentReleasePointer,
    ) -> AgentReleasePointer:
        del scope
        key = (mutation.agent_id, mutation.channel)
        current = self.pointers.get(key)
        if (current.revision if current else 0) != mutation.expected_revision:
            raise RepositoryContractError("release pointer CAS conflict")
        if (current.version_id if current else None) != mutation.expected_version_id:
            raise RepositoryContractError("release pointer CAS version conflict")
        self.pointers[key] = pointer
        return pointer

    def get_approval(
        self,
        scope: AccessScope,
        agent_id: str,
        version_id: str,
        approval_id: str,
    ) -> ApprovalRecord | None:
        del scope
        return self.approvals.get((agent_id, version_id, approval_id))

    def list_agents(self, request: AgentListRequest) -> AgentPage:
        del request
        assert self.page is not None
        return self.page

    def cursor_for(
        self,
        scope: AccessScope,
        after_agent_id: str,
        after_version_id: str,
    ) -> AgentKeysetCursor:
        return AgentKeysetCursor(
            cursor_version="agent-keyset-cursor.v1",
            scope_digest=scope.scope_digest,
            after_agent_id=after_agent_id,
            after_version_id=after_version_id,
        )


def _fixture() -> tuple[
    _MemoryRepository,
    AgentControlPlane,
    AccessScope,
    AgentIdentity,
    AgentVersion,
    AgentReleasePointer,
    ApprovalRecord,
]:
    identity = AgentIdentity(
        identity_version="agent-identity.v1",
        agent_id=agent_id_for("knuckles", "agent-pack", "researcher"),
        publisher="knuckles",
        package_name="agent-pack",
        agent_name="researcher",
    )
    bindings = _bindings()
    policies = _policies()
    evaluation = _evaluation()
    version = AgentVersion(
        version_record_version="agent-version.v1",
        version_id=agent_version_id_for(
            identity.agent_id,
            "1.0.0",
            binding_set_digest(bindings),
            policies.policy_set_digest,
            evaluation.evidence_digest,
        ),
        agent_id=identity.agent_id,
        version="1.0.0",
        bindings=bindings,
        policies=policies,
        evaluation=evaluation,
    )
    scope = AccessScope(
        scope_version="agent-access-scope.v1",
        tenant_id="tenant:alpha",
        principal_id="principal:operator",
        grant_digests=(_digest("1"),),
        scope_digest=scope_digest_for(
            "tenant:alpha", "principal:operator", (_digest("1"),)
        ),
    )
    pointer = AgentReleasePointer(
        pointer_version="agent-release-pointer.v1",
        pointer_id=pointer_id_for(identity.agent_id, "stable"),
        agent_id=identity.agent_id,
        channel="stable",
        version_id=version.version_id,
        revision=1,
        pointer_digest=pointer_digest_for(
            identity.agent_id, "stable", version.version_id, 1, None
        ),
    )
    approval_id = "approval:research:1"
    approved_at = "2026-08-19T00:00:00Z"
    expires_at = "2026-08-20T00:00:00Z"
    approval_digest_payload = {
        "approval_id": approval_id,
        "agent_id": identity.agent_id,
        "version_id": version.version_id,
        "channel": "stable",
        "policy_set_digest": policies.policy_set_digest,
        "evaluation_digest": evaluation.evidence_digest,
        "outcome": "approved",
        "approved_at": approved_at,
        "expires_at": expires_at,
        "approver_ref": "principal:reviewer",
        "evidence_ref": "evidence:research:1",
    }
    from agent_utilities.control_plane.agents.models import _digest_payload

    approval = ApprovalRecord(
        approval_version="agent-approval.v1",
        approval_id=approval_id,
        agent_id=identity.agent_id,
        version_id=version.version_id,
        channel="stable",
        policy_set_digest=policies.policy_set_digest,
        evaluation_digest=evaluation.evidence_digest,
        outcome="approved",
        approved_at=approved_at,
        expires_at=expires_at,
        approver_ref="principal:reviewer",
        evidence_ref="evidence:research:1",
        approval_digest=_digest_payload(approval_digest_payload),
    )
    repository = _MemoryRepository(
        pointers={(identity.agent_id, "stable"): pointer},
        approvals={(identity.agent_id, version.version_id, approval_id): approval},
    )
    plane = AgentControlPlane(repository)
    plane.register(
        AgentRegistration(
            registration_version="agent-registration.v1",
            identity=identity,
            version=version,
        )
    )
    return repository, plane, scope, identity, version, pointer, approval


def _request(
    scope: AccessScope,
    identity: AgentIdentity,
    version: AgentVersion,
    approval: ApprovalRecord,
    **overrides: object,
) -> AgentResolutionRequest:
    values: dict[str, object] = {
        "request_version": "agent-resolution-request.v1",
        "agent_id": identity.agent_id,
        "scope": scope,
        "channel": "stable",
        "approval_id": approval.approval_id,
        "at": "2026-08-19T01:00:00Z",
    }
    values.update(overrides)
    return AgentResolutionRequest(**values)


def test_identity_version_and_binding_contracts_are_stable() -> None:
    _, _, _, identity, version, _, _ = _fixture()
    assert identity.agent_id == agent_id_for("knuckles", "agent-pack", "researcher")
    with pytest.raises(ValidationError):
        AgentIdentity.model_validate(
            {**identity.model_dump(mode="json"), "unknown_key": "rejected"}
        )
    with pytest.raises(ValidationError):
        version.model_validate({**version.model_dump(mode="json"), "version": "2.0.0"})
    with pytest.raises(ValidationError):
        ArtifactBinding(
            binding_version="agent-artifact-binding.v1",
            binding_id="binding:bad",
            kind="prompt",
            logical_id="prompt",
            version="1.0.0",
            artifact_ref="secret:inline-body",
            artifact_digest=_digest("a"),
            schema_digest=_digest("b"),
        )


def test_binding_cycles_and_alias_version_ambiguity_fail_closed() -> None:
    config = _binding("configuration", "config")
    model = _binding("model", "model")
    prompt = _binding("prompt", "prompt")
    first = _binding("workflow", "first")
    second = _binding("workflow", "second", depends_on=(first.binding_id,))
    first_cycle = _binding("workflow", "first", depends_on=(second.binding_id,))
    with pytest.raises(ValidationError):
        AgentBindings(
            bindings_version="agent-bindings.v1",
            configuration=config,
            model=model,
            prompt=prompt,
            workflows=(first_cycle, second),
        )
    duplicate_a = _binding("tool", "same-tool", "1.0.0")
    duplicate_b = _binding("tool", "same-tool", "2.0.0")
    with pytest.raises(ValidationError):
        AgentBindings(
            bindings_version="agent-bindings.v1",
            configuration=config,
            model=model,
            prompt=prompt,
            tools=(duplicate_a, duplicate_b),
        )


def test_resolution_is_deterministic_and_capability_bound() -> None:
    _, plane, scope, identity, version, _, approval = _fixture()
    first = plane.resolve(_request(scope, identity, version, approval))
    second = plane.resolve(_request(scope, identity, version, approval))
    assert first == second
    assert first.resolution_digest.startswith("sha256:")
    with pytest.raises(AgentResolutionError, match="capability_mismatch"):
        plane.resolve(
            _request(
                scope,
                identity,
                version,
                approval,
                requested_binding_ids=("agent-binding:missing",),
            )
        )


def test_stale_approval_and_budget_or_delegation_escalation_are_denied() -> None:
    _, plane, scope, identity, version, _, approval = _fixture()
    with pytest.raises(AgentResolutionError, match="token_budget_escalation"):
        plane.resolve(
            _request(scope, identity, version, approval, requested_tokens=10_001)
        )
    with pytest.raises(AgentResolutionError, match="delegation_depth_escalation"):
        plane.resolve(
            _request(scope, identity, version, approval, delegation_depth=3)
        )
    with pytest.raises(AgentResolutionError, match="approval_required"):
        plane.resolve(
            _request(scope, identity, version, approval, approval_id="approval:missing")
        )


def test_release_pointer_cas_and_rollback_are_explicit() -> None:
    repository, plane, scope, identity, version, pointer, approval = _fixture()
    mutation = ReleaseMutation(
        mutation_version="agent-release-mutation.v1",
        agent_id=identity.agent_id,
        channel="stable",
        operation="promote",
        expected_revision=pointer.revision,
        expected_version_id=pointer.version_id,
        next_version_id=version.version_id,
        change_ref="change:noop",
    )
    with pytest.raises(AgentResolutionError, match="release_pointer_noop"):
        plane.promote(
            scope,
            mutation,
            approval_id=approval.approval_id,
            at="2026-08-19T01:00:00Z",
        )
    with pytest.raises(AgentResolutionError, match="release_pointer_revision_conflict"):
        plane.promote(
            scope,
            mutation.model_copy(update={"expected_revision": 0}),
            approval_id=approval.approval_id,
            at="2026-08-19T01:00:00Z",
        )

    replacement = pointer.model_copy(
        update={
            "revision": 2,
            "previous_version_id": "agent-version:previous",
            "pointer_digest": pointer_digest_for(
                identity.agent_id,
                "stable",
                version.version_id,
                2,
                "agent-version:previous",
            ),
        }
    )
    repository.pointers[(identity.agent_id, "stable")] = replacement
    rollback = ReleaseMutation(
        mutation_version="agent-release-mutation.v1",
        agent_id=identity.agent_id,
        channel="stable",
        operation="rollback",
        expected_revision=2,
        expected_version_id=version.version_id,
        next_version_id="agent-version:previous",
        change_ref="change:rollback",
    )
    with pytest.raises(AgentResolutionError, match="agent_version_unavailable"):
        plane.promote(
            scope,
            rollback,
            approval_id=approval.approval_id,
            at="2026-08-19T01:00:00Z",
        )


def test_scope_bound_page_and_projection_are_privacy_safe() -> None:
    repository, plane, scope, identity, version, pointer, approval = _fixture()
    projection = plane.project(
        scope,
        identity.agent_id,
        "stable",
        approval_id=approval.approval_id,
        at="2026-08-19T01:00:00Z",
    )
    assert projection.approval_state == "approved"
    assert "secret" not in projection.model_dump_json()
    repository.page = AgentPage(
        page_version="agent-page.v1",
        items=(
            AgentPageItem(
                tenant_id=scope.tenant_id,
                agent_id=identity.agent_id,
                version_id=version.version_id,
                channel=pointer.channel,
                pointer_revision=pointer.revision,
                binding_count=7,
                binding_set_digest=binding_set_digest(version.bindings),
                policy_set_digest=version.policies.policy_set_digest,
                evaluation_digest=version.evaluation.evidence_digest,
                approval_state="approved",
            ),
        ),
    )
    page = plane.list(AgentListRequest(scope=scope, limit=1))
    assert len(page.items) == 1
    wrong_cursor = AgentKeysetCursor(
        cursor_version="agent-keyset-cursor.v1",
        scope_digest=_digest("9"),
        after_agent_id=identity.agent_id,
        after_version_id=version.version_id,
    )
    with pytest.raises(ValidationError):
        AgentListRequest(scope=scope, cursor=wrong_cursor)
