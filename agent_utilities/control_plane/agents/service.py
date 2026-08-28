"""Agent-plane release admission over a typed repository seam.

The service resolves and validates an agent release before dispatch, but it
does not import an executor, spawn a process, call a model, or invoke a tool.
That separation keeps release governance authoritative and lets the existing
orchestration path consume a small, deterministic plan.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field

from agent_utilities.protocols.epistemic_operations import ProtocolModel

from .models import (
    AccessScope,
    AgentGraphProjection,
    AgentListRequest,
    AgentPage,
    AgentPageItem,
    AgentRegistration,
    AgentReleasePointer,
    AgentResolutionRequest,
    AgentVersion,
    ApprovalRecord,
    ReleaseMutation,
    ReleaseTrack,
    ResolvedAgentPlan,
    binding_set_digest,
    pointer_digest_for,
    pointer_id_for,
    resolution_digest_for,
)
from .repository import AgentRepository, RepositoryContractError


class AgentResolutionError(ValueError):
    """Stable fail-closed pre-dispatch resolution error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class AgentPromotionResult(ProtocolModel):
    """Pointer mutation result; it contains no runtime dispatch handle."""

    result_version: Literal["agent-promotion-result.v1"]
    pointer: AgentReleasePointer
    operation: Literal["promote", "rollback"]
    change_ref: str = Field(min_length=1, max_length=192)


def _parse_timestamp(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value[-1:] in {"Z", "z"} else value
    parsed = datetime.fromisoformat(normalized)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _approval_state(
    approval: ApprovalRecord | None,
    *,
    agent_id: str,
    version_id: str,
    channel: str,
    at: str | None = None,
) -> Literal["missing", "approved", "stale", "denied"]:
    if approval is None:
        return "missing"
    if (
        approval.agent_id != agent_id
        or approval.version_id != version_id
        or approval.channel != channel
    ):
        return "stale"
    if approval.outcome != "approved":
        return "denied"
    if at is None:
        return "approved"
    try:
        if _parse_timestamp(approval.expires_at) <= _parse_timestamp(at):
            return "stale"
        if _parse_timestamp(approval.approved_at) > _parse_timestamp(at):
            return "stale"
    except ValueError:
        return "stale"
    return "approved"


class AgentControlPlane:
    """Resolve governed agent releases without becoming a dispatch authority."""

    def __init__(self, repository: AgentRepository) -> None:
        self._repository = repository

    def register(self, registration: AgentRegistration) -> None:
        """Retain one immutable identity/release bundle through the repository."""

        self._repository.put_registration(registration)

    def _resolve_pointer_for_request(
        self,
        request: AgentResolutionRequest,
    ) -> AgentReleasePointer:
        if request.channel is not None:
            pointer = self._repository.get_release_pointer(
                request.scope, request.agent_id, request.channel
            )
        else:
            pointer = self._repository.get_release_pointer_for_version(
                request.scope, request.agent_id, request.version_id or ""
            )
        if pointer is None:
            raise AgentResolutionError("release_pointer_unavailable")
        if pointer.agent_id != request.agent_id:
            raise RepositoryContractError(
                "repository returned a cross-agent release pointer"
            )
        if request.channel is not None and pointer.channel != request.channel:
            raise RepositoryContractError(
                "repository returned a cross-channel release pointer"
            )
        if request.version_id is not None and pointer.version_id != request.version_id:
            raise AgentResolutionError("version_not_active_on_selected_pointer")
        return pointer

    def _resolve_version_for_pointer(
        self,
        request: AgentResolutionRequest,
        pointer: AgentReleasePointer,
    ) -> AgentVersion:
        version = self._repository.get_version(request.scope, pointer.version_id)
        if version is None:
            raise AgentResolutionError("agent_version_unavailable")
        if version.agent_id != request.agent_id:
            raise RepositoryContractError("repository returned a cross-agent release")
        return version

    def _pointer_for_request(
        self,
        request: AgentResolutionRequest,
    ) -> tuple[AgentReleasePointer, AgentVersion]:
        pointer = self._resolve_pointer_for_request(request)
        version = self._resolve_version_for_pointer(request, pointer)
        return pointer, version

    @staticmethod
    def _validate_approval(
        request: AgentResolutionRequest,
        pointer: AgentReleasePointer,
        version: AgentVersion,
        approval: ApprovalRecord | None,
    ) -> None:
        state = _approval_state(
            approval,
            agent_id=version.agent_id,
            version_id=version.version_id,
            channel=pointer.channel,
            at=request.at,
        )
        if state == "missing":
            raise AgentResolutionError("approval_required")
        if state == "stale":
            raise AgentResolutionError("approval_stale")
        if state == "denied":
            raise AgentResolutionError("approval_denied")
        if approval is None:
            raise AgentResolutionError("approval_required")
        if approval.approval_id != request.approval_id:
            raise AgentResolutionError("approval_stale")
        if (
            approval.policy_set_digest != version.policies.policy_set_digest
            or approval.evaluation_digest != version.evaluation.evidence_digest
        ):
            raise AgentResolutionError("approval_release_evidence_mismatch")

    @staticmethod
    def _resolve_selected_bindings(
        request: AgentResolutionRequest,
        version: AgentVersion,
    ) -> tuple[str, ...]:
        all_binding_ids = set(version.bindings.binding_ids)
        selected = (
            tuple(request.requested_binding_ids)
            if request.requested_binding_ids
            else version.bindings.binding_ids
        )
        if any(binding_id not in all_binding_ids for binding_id in selected):
            raise AgentResolutionError("capability_mismatch")
        return selected

    @staticmethod
    def _validate_team_policy(
        request: AgentResolutionRequest,
        version: AgentVersion,
        selected: tuple[str, ...],
    ) -> None:
        allowed = set(version.policies.team.allowed_binding_ids)
        if allowed and any(binding_id not in allowed for binding_id in selected):
            raise AgentResolutionError("team_capability_mismatch")
        if (
            request.team_ref is not None
            and request.team_ref != version.policies.team.team_ref
        ):
            raise AgentResolutionError("team_policy_mismatch")
        if request.team_member_count > version.policies.team.max_members:
            raise AgentResolutionError("team_budget_escalation")

    @staticmethod
    def _validate_delegation_policy(
        request: AgentResolutionRequest,
        version: AgentVersion,
    ) -> None:
        if request.delegation_depth > version.policies.delegation.max_depth:
            raise AgentResolutionError("delegation_depth_escalation")
        if request.delegation_children > version.policies.delegation.max_children:
            raise AgentResolutionError("delegation_fanout_escalation")
        if (
            request.requested_cost_micros
            > version.policies.delegation.max_budget_micros
        ):
            raise AgentResolutionError("delegation_budget_escalation")

    @staticmethod
    def _validate_budget_policy(
        request: AgentResolutionRequest,
        version: AgentVersion,
    ) -> None:
        budget = version.policies.budget
        if request.requested_tokens > budget.max_tokens:
            raise AgentResolutionError("token_budget_escalation")
        if request.requested_cost_micros > budget.max_cost_micros:
            raise AgentResolutionError("cost_budget_escalation")
        if request.requested_time_ms > budget.max_time_ms:
            raise AgentResolutionError("time_budget_escalation")

    @staticmethod
    def _validate_request(
        request: AgentResolutionRequest,
        version: AgentVersion,
    ) -> tuple[str, ...]:
        selected = AgentControlPlane._resolve_selected_bindings(request, version)
        AgentControlPlane._validate_team_policy(request, version, selected)
        AgentControlPlane._validate_delegation_policy(request, version)
        AgentControlPlane._validate_budget_policy(request, version)
        return tuple(sorted(selected))

    def resolve(self, request: AgentResolutionRequest) -> ResolvedAgentPlan:
        """Resolve and attest a release; never dispatch or execute it."""

        pointer, version = self._pointer_for_request(request)
        approval = self._repository.get_approval(
            request.scope,
            version.agent_id,
            version.version_id,
            request.approval_id,
        )
        self._validate_approval(request, pointer, version, approval)
        selected = self._validate_request(request, version)
        if approval is None:
            raise AgentResolutionError("approval_required")
        resolution_input = {
            "agent_id": version.agent_id,
            "version_id": version.version_id,
            "channel": pointer.channel,
            "pointer_digest": pointer.pointer_digest,
            "approval_digest": approval.approval_digest,
            "binding_ids": selected,
            "policy_set_digest": version.policies.policy_set_digest,
            "evaluation_digest": version.evaluation.evidence_digest,
            "team_ref": request.team_ref or version.policies.team.team_ref,
            "team_member_count": request.team_member_count,
            "delegation_depth": request.delegation_depth,
            "delegation_children": request.delegation_children,
            "requested_tokens": request.requested_tokens,
            "requested_cost_micros": request.requested_cost_micros,
            "requested_time_ms": request.requested_time_ms,
            "at": request.at,
        }
        return ResolvedAgentPlan(
            plan_version="resolved-agent-plan.v1",
            agent_id=version.agent_id,
            version_id=version.version_id,
            channel=pointer.channel,
            pointer_digest=pointer.pointer_digest,
            approval_id=approval.approval_id,
            binding_ids=selected,
            policy_set_digest=version.policies.policy_set_digest,
            evaluation_digest=version.evaluation.evidence_digest,
            resolution_digest=resolution_digest_for(resolution_input),
        )

    def _resolve_current_pointer_state(
        self,
        scope: AccessScope,
        mutation: ReleaseMutation,
    ) -> tuple[AgentReleasePointer | None, int, str | None]:
        current = self._repository.get_release_pointer(
            scope, mutation.agent_id, mutation.channel
        )
        current_revision = current.revision if current is not None else 0
        current_version_id = current.version_id if current is not None else None
        if mutation.expected_revision != current_revision:
            raise AgentResolutionError("release_pointer_revision_conflict")
        if mutation.expected_version_id != current_version_id:
            raise AgentResolutionError("release_pointer_version_conflict")
        if current is not None and mutation.next_version_id == current.version_id:
            raise AgentResolutionError("release_pointer_noop")
        if mutation.operation == "rollback" and (
            current is None or current.previous_version_id != mutation.next_version_id
        ):
            raise AgentResolutionError("rollback_target_not_previous_release")
        return current, current_revision, current_version_id

    def _resolve_promotion_target_version(
        self,
        scope: AccessScope,
        mutation: ReleaseMutation,
    ) -> AgentVersion:
        version = self._repository.get_version(scope, mutation.next_version_id)
        if version is None:
            raise AgentResolutionError("agent_version_unavailable")
        if version.agent_id != mutation.agent_id:
            raise RepositoryContractError(
                "repository returned a cross-agent target release"
            )
        return version

    @staticmethod
    def _build_promotion_pointer(
        mutation: ReleaseMutation,
        version: AgentVersion,
        current_revision: int,
        current_version_id: str | None,
    ) -> AgentReleasePointer:
        return AgentReleasePointer(
            pointer_version="agent-release-pointer.v1",
            pointer_id=pointer_id_for(mutation.agent_id, mutation.channel),
            agent_id=mutation.agent_id,
            channel=mutation.channel,
            version_id=version.version_id,
            revision=current_revision + 1,
            previous_version_id=current_version_id,
            pointer_digest=pointer_digest_for(
                mutation.agent_id,
                mutation.channel,
                version.version_id,
                current_revision + 1,
                current_version_id,
            ),
        )

    def promote(
        self,
        scope: AccessScope,
        mutation: ReleaseMutation,
        *,
        approval_id: str,
        at: str,
    ) -> AgentPromotionResult:
        """Apply an approved promotion/rollback through repository CAS."""

        current, current_revision, current_version_id = (
            self._resolve_current_pointer_state(scope, mutation)
        )
        version = self._resolve_promotion_target_version(scope, mutation)
        approval = self._repository.get_approval(
            scope, mutation.agent_id, version.version_id, approval_id
        )
        synthetic_request = AgentResolutionRequest(
            request_version="agent-resolution-request.v1",
            agent_id=mutation.agent_id,
            scope=scope,
            channel=mutation.channel,
            approval_id=approval_id,
            at=at,
        )
        pointer = self._build_promotion_pointer(
            mutation, version, current_revision, current_version_id
        )
        self._validate_approval(synthetic_request, pointer, version, approval)
        applied = self._repository.compare_and_swap_release(scope, mutation, pointer)
        if applied != pointer:
            raise RepositoryContractError(
                "repository returned a different release pointer"
            )
        return AgentPromotionResult(
            result_version="agent-promotion-result.v1",
            pointer=applied,
            operation=mutation.operation,
            change_ref=mutation.change_ref,
        )

    @staticmethod
    def _coerce_release_channel(channel: str) -> ReleaseTrack:
        if channel == "stable":
            return "stable"
        if channel == "beta":
            return "beta"
        if channel == "edge":
            return "edge"
        raise AgentResolutionError("release_channel_unknown")

    def _resolve_projection_pointer(
        self,
        scope: AccessScope,
        agent_id: str,
        release_channel: ReleaseTrack,
        channel: str,
    ) -> AgentReleasePointer:
        pointer = self._repository.get_release_pointer(scope, agent_id, release_channel)
        if pointer is None:
            raise AgentResolutionError("release_pointer_unavailable")
        if pointer.agent_id != agent_id or pointer.channel != channel:
            raise RepositoryContractError(
                "repository returned an invalid agent pointer"
            )
        return pointer

    def _resolve_projection_version(
        self,
        scope: AccessScope,
        agent_id: str,
        pointer: AgentReleasePointer,
    ) -> AgentVersion:
        version = self._repository.get_version(scope, pointer.version_id)
        if version is None:
            raise AgentResolutionError("agent_version_unavailable")
        if version.agent_id != agent_id:
            raise RepositoryContractError("repository returned a cross-agent release")
        return version

    def _resolve_projection_approval_state(
        self,
        scope: AccessScope,
        agent_id: str,
        version: AgentVersion,
        pointer: AgentReleasePointer,
        approval_id: str | None,
        at: str | None,
    ) -> Literal["missing", "approved", "stale", "denied"]:
        approval = (
            self._repository.get_approval(
                scope, agent_id, version.version_id, approval_id
            )
            if approval_id is not None
            else None
        )
        if approval_id is not None and (
            approval is None
            or approval.approval_id != approval_id
            or approval.policy_set_digest != version.policies.policy_set_digest
            or approval.evaluation_digest != version.evaluation.evidence_digest
        ):
            return "stale"
        return _approval_state(
            approval,
            agent_id=agent_id,
            version_id=version.version_id,
            channel=pointer.channel,
            at=at,
        )

    def project(
        self,
        scope: AccessScope,
        agent_id: str,
        channel: str,
        *,
        approval_id: str | None = None,
        at: str | None = None,
    ) -> AgentGraphProjection:
        """Return a bounded graph projection without resolving or dispatching."""

        release_channel = self._coerce_release_channel(channel)
        pointer = self._resolve_projection_pointer(
            scope, agent_id, release_channel, channel
        )
        version = self._resolve_projection_version(scope, agent_id, pointer)
        state = self._resolve_projection_approval_state(
            scope, agent_id, version, pointer, approval_id, at
        )
        return AgentGraphProjection(
            projection_version="agent-graph-projection.v1",
            tenant_id=scope.tenant_id,
            agent_id=agent_id,
            version_id=version.version_id,
            channel=pointer.channel,
            pointer_revision=pointer.revision,
            binding_count=len(version.bindings.all_bindings),
            binding_set_digest=binding_set_digest(version.bindings),
            policy_set_digest=version.policies.policy_set_digest,
            evaluation_digest=version.evaluation.evidence_digest,
            approval_state=state,
        )

    def list(self, request: AgentListRequest) -> AgentPage:
        """Return a bounded page and reject scope/cursor violations."""

        page = self._repository.list_agents(request)
        if len(page.items) > request.limit:
            raise RepositoryContractError("repository returned an oversized agent page")
        if any(item.tenant_id != request.scope.tenant_id for item in page.items):
            raise RepositoryContractError(
                "repository returned a cross-tenant agent page"
            )
        pointer_keys = {(item.agent_id, item.channel) for item in page.items}
        if len(pointer_keys) != len(page.items):
            raise RepositoryContractError(
                "repository returned duplicate agent pointers"
            )
        if (
            page.next_cursor is not None
            and page.next_cursor.scope_digest != request.scope.scope_digest
        ):
            raise RepositoryContractError("repository returned an unbound agent cursor")
        return page

    def summarize_page_item(
        self,
        scope: AccessScope,
        agent_id: str,
        channel: str,
        *,
        approval_id: str | None = None,
        at: str | None = None,
    ) -> AgentPageItem:
        """Build the same privacy-safe summary used by bounded list pages."""

        projection = self.project(
            scope, agent_id, channel, approval_id=approval_id, at=at
        )
        return AgentPageItem(
            tenant_id=projection.tenant_id,
            agent_id=projection.agent_id,
            version_id=projection.version_id,
            channel=projection.channel,
            pointer_revision=projection.pointer_revision,
            binding_count=projection.binding_count,
            binding_set_digest=projection.binding_set_digest,
            policy_set_digest=projection.policy_set_digest,
            evaluation_digest=projection.evaluation_digest,
            approval_state=projection.approval_state,
        )
