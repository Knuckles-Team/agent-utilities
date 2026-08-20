"""Immutable, tenant-scoped Web UI authority models.

The Web UI control plane stores identities and opaque references only.  Session
credentials, source content, attachment bytes, preference bodies, query text,
provider grants, and GraphOS permission mutations stay outside this package.
Every persisted identity carries a monotonic CAS version; mutable lifecycle
state is represented as an explicit retention record.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "MAX_ATTACHMENTS",
    "MAX_WIDGETS",
    "MAX_PAGE_SIZE",
    "Digest",
    "LifecycleState",
    "OpaqueRef",
    "PageSize",
    "UiPermission",
    "Visibility",
    "AccessContext",
    "AnonymousPilotSession",
    "AttachmentIdentity",
    "ContentReference",
    "ConversationIdentity",
    "Cursor",
    "DashboardIdentity",
    "EntityKind",
    "FeedbackIdentity",
    "MessageIdentity",
    "NotificationIdentity",
    "Page",
    "PageRequest",
    "PreferenceIdentity",
    "ProfileIdentity",
    "RetentionRecord",
    "SavedQueryIdentity",
    "SessionIdentity",
    "SupportIdentity",
    "TenantIdentity",
    "UiStateWriteReceipt",
    "UserIdentity",
    "WidgetIdentity",
    "WorkspaceIdentity",
    "WebUiEntity",
    "UiStateEntity",
    "canonical_digest",
]


MAX_PAGE_SIZE = 100
MAX_ATTACHMENTS = 16
MAX_WIDGETS = 64

_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9:_./-]{0,255}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

OpaqueRef: TypeAlias = Annotated[str, Field(pattern=_ID_RE.pattern, min_length=1)]
Digest: TypeAlias = Annotated[str, Field(pattern=_DIGEST_RE.pattern)]
Version: TypeAlias = Annotated[int, Field(ge=1, le=2_147_483_647)]
Timestamp: TypeAlias = Annotated[int, Field(ge=0)]
PageSize: TypeAlias = Annotated[int, Field(ge=1, le=MAX_PAGE_SIZE)]
Cursor: TypeAlias = Annotated[str, Field(pattern=r"^cursor:(0|[1-9][0-9]{0,9})$")]
UiPermission: TypeAlias = Literal["read", "write", "feedback", "support", "admin"]
LifecycleState: TypeAlias = Literal[
    "active", "retained", "deletion_pending", "deleted", "legal_hold"
]
EntityKind: TypeAlias = Literal[
    "tenant",
    "workspace",
    "user",
    "profile",
    "session",
    "conversation",
    "content",
    "attachment",
    "message",
    "preference",
    "saved_query",
    "dashboard",
    "widget",
    "notification",
    "feedback",
    "support",
]
Visibility: TypeAlias = Literal["private", "workspace"]

_FORBIDDEN_INLINE_KEYS = {
    "body",
    "content",
    "credentials",
    "grant",
    "graph_permissions",
    "password",
    "permission_grant",
    "prompt",
    "provider_grant",
    "public_share",
    "result",
    "secret",
    "text",
    "token",
    "values",
    "vector",
}


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


def _reject_alias(value: str, field_name: str) -> None:
    if value.casefold() in {"latest", "current", "default", "public"}:
        raise ValueError(f"{field_name}_alias_or_public_forbidden")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=False,
        strict=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_inline_material(cls, value: object) -> object:
        if isinstance(value, Mapping):
            keys = {
                str(key).casefold().replace("-", "_")
                for key in value
                if str(key).casefold().replace("-", "_") in _FORBIDDEN_INLINE_KEYS
            }
            if keys:
                raise ValueError("inline_ui_material_or_authority_forbidden")
        return value


class _IdentityModel(_FrozenModel):
    version: Version
    digest: Digest

    @property
    def identity_digest(self) -> str:
        return canonical_digest(self)


class AccessContext(_FrozenModel):
    """Explicit tenant/workspace/session authority for every operation."""

    tenant_ref: OpaqueRef
    workspace_ref: OpaqueRef
    actor_ref: OpaqueRef
    session_ref: OpaqueRef
    permissions: tuple[UiPermission, ...] = Field(min_length=1, max_length=5)
    authenticated: bool = True
    anonymous_pilot: bool = False
    private_boundary: Literal[True] = True

    @model_validator(mode="after")
    def _authority_is_private_and_unambiguous(self) -> AccessContext:
        if len(self.permissions) != len(set(self.permissions)):
            raise ValueError("ui_permission_duplicate")
        if self.anonymous_pilot:
            if self.authenticated:
                raise ValueError("anonymous_pilot_authenticated")
            if self.actor_ref != "actor:anonymous-pilot":
                raise ValueError("anonymous_pilot_actor_invalid")
            if "admin" in self.permissions:
                raise ValueError("anonymous_pilot_admin_forbidden")
            if any(
                permission not in {"read", "write", "feedback", "support"}
                for permission in self.permissions
            ):
                raise ValueError("anonymous_pilot_permission_invalid")
        return self

    def allows(self, permission: UiPermission) -> bool:
        return permission in self.permissions


class TenantIdentity(_IdentityModel):
    tenant_ref: OpaqueRef
    state: Literal["active", "deleted"] = "active"


class WorkspaceIdentity(_IdentityModel):
    tenant_ref: OpaqueRef
    workspace_ref: OpaqueRef
    state: Literal["active", "deleted"] = "active"

    @model_validator(mode="after")
    def _exact_refs(self) -> WorkspaceIdentity:
        _reject_alias(self.workspace_ref, "workspace")
        return self


class UserIdentity(_IdentityModel):
    tenant_ref: OpaqueRef
    workspace_ref: OpaqueRef
    user_ref: OpaqueRef
    state: Literal["active", "deleted"] = "active"


class ProfileIdentity(_IdentityModel):
    tenant_ref: OpaqueRef
    workspace_ref: OpaqueRef
    user_ref: OpaqueRef
    profile_ref: OpaqueRef
    state_ref: OpaqueRef


class SessionIdentity(_IdentityModel):
    tenant_ref: OpaqueRef
    workspace_ref: OpaqueRef
    session_ref: OpaqueRef
    user_ref: OpaqueRef
    anonymous_pilot: bool = False
    private_boundary: Literal[True] = True
    issued_at: Timestamp
    expires_at: Timestamp

    @model_validator(mode="after")
    def _session_is_bounded(self) -> SessionIdentity:
        _reject_alias(self.session_ref, "session")
        if self.expires_at <= self.issued_at:
            raise ValueError("session_expiry_invalid")
        if self.expires_at - self.issued_at > 86_400:
            raise ValueError("session_lifetime_exceeded")
        if self.anonymous_pilot and self.user_ref != "user:anonymous-pilot":
            raise ValueError("anonymous_pilot_user_invalid")
        return self


class ConversationIdentity(_IdentityModel):
    tenant_ref: OpaqueRef
    workspace_ref: OpaqueRef
    conversation_ref: OpaqueRef
    session_ref: OpaqueRef
    state: Literal["active", "archived"] = "active"


class ContentReference(_IdentityModel):
    """Opaque content identity; bytes live behind the artifact authority."""

    tenant_ref: OpaqueRef
    workspace_ref: OpaqueRef
    content_ref: OpaqueRef
    artifact_ref: OpaqueRef
    content_digest: Digest
    media_type: OpaqueRef
    size_bytes: int = Field(ge=0, le=256 * 1024 * 1024)


class AttachmentIdentity(_IdentityModel):
    tenant_ref: OpaqueRef
    workspace_ref: OpaqueRef
    attachment_ref: OpaqueRef
    content_ref: ContentReference
    attachment_digest: Digest
    media_type: OpaqueRef
    size_bytes: int = Field(ge=0, le=256 * 1024 * 1024)

    @model_validator(mode="after")
    def _content_scope_matches(self) -> AttachmentIdentity:
        if self.content_ref.tenant_ref != self.tenant_ref:
            raise ValueError("attachment_tenant_drift")
        if self.content_ref.workspace_ref != self.workspace_ref:
            raise ValueError("attachment_workspace_drift")
        return self


class MessageIdentity(_IdentityModel):
    tenant_ref: OpaqueRef
    workspace_ref: OpaqueRef
    message_ref: OpaqueRef
    conversation_ref: OpaqueRef
    sender_ref: OpaqueRef
    content_ref: ContentReference
    attachments: tuple[AttachmentIdentity, ...] = Field(max_length=MAX_ATTACHMENTS)
    state: Literal["active", "deleted"] = "active"

    @model_validator(mode="after")
    def _content_scope_matches(self) -> MessageIdentity:
        if self.content_ref.tenant_ref != self.tenant_ref:
            raise ValueError("message_content_tenant_drift")
        if self.content_ref.workspace_ref != self.workspace_ref:
            raise ValueError("message_content_workspace_drift")
        if any(
            attachment.tenant_ref != self.tenant_ref
            or attachment.workspace_ref != self.workspace_ref
            for attachment in self.attachments
        ):
            raise ValueError("message_attachment_scope_drift")
        return self


class PreferenceIdentity(_IdentityModel):
    tenant_ref: OpaqueRef
    workspace_ref: OpaqueRef
    preference_ref: OpaqueRef
    user_ref: OpaqueRef
    state_ref: OpaqueRef


class SavedQueryIdentity(_IdentityModel):
    tenant_ref: OpaqueRef
    workspace_ref: OpaqueRef
    saved_query_ref: OpaqueRef
    owner_ref: OpaqueRef
    query_ref: OpaqueRef
    visibility: Visibility = "private"

    @model_validator(mode="after")
    def _no_public_visibility(self) -> SavedQueryIdentity:
        _reject_alias(self.visibility, "query_visibility")
        return self


class WidgetIdentity(_IdentityModel):
    tenant_ref: OpaqueRef
    workspace_ref: OpaqueRef
    widget_ref: OpaqueRef
    dashboard_ref: OpaqueRef
    widget_type: OpaqueRef
    state_ref: OpaqueRef


class DashboardIdentity(_IdentityModel):
    tenant_ref: OpaqueRef
    workspace_ref: OpaqueRef
    dashboard_ref: OpaqueRef
    owner_ref: OpaqueRef
    visibility: Visibility = "private"
    widgets: tuple[WidgetIdentity, ...] = Field(max_length=MAX_WIDGETS)

    @model_validator(mode="after")
    def _widgets_are_scoped(self) -> DashboardIdentity:
        _reject_alias(self.visibility, "dashboard_visibility")
        refs = []
        for widget in self.widgets:
            if widget.tenant_ref != self.tenant_ref:
                raise ValueError("dashboard_widget_tenant_drift")
            if widget.workspace_ref != self.workspace_ref:
                raise ValueError("dashboard_widget_workspace_drift")
            if widget.dashboard_ref != self.dashboard_ref:
                raise ValueError("dashboard_widget_reference_drift")
            refs.append(widget.widget_ref)
        if len(refs) != len(set(refs)):
            raise ValueError("dashboard_widget_duplicate")
        return self


class NotificationIdentity(_IdentityModel):
    tenant_ref: OpaqueRef
    workspace_ref: OpaqueRef
    notification_ref: OpaqueRef
    recipient_ref: OpaqueRef
    content_ref: ContentReference
    state: Literal["unread", "read"] = "unread"

    @model_validator(mode="after")
    def _content_scope_matches(self) -> NotificationIdentity:
        if self.content_ref.tenant_ref != self.tenant_ref:
            raise ValueError("notification_content_tenant_drift")
        if self.content_ref.workspace_ref != self.workspace_ref:
            raise ValueError("notification_content_workspace_drift")
        return self


class FeedbackIdentity(_IdentityModel):
    tenant_ref: OpaqueRef
    workspace_ref: OpaqueRef
    feedback_ref: OpaqueRef
    actor_ref: OpaqueRef
    conversation_ref: OpaqueRef | None = None
    message_ref: OpaqueRef | None = None
    content_ref: ContentReference
    category: OpaqueRef

    @model_validator(mode="after")
    def _feedback_scope_matches(self) -> FeedbackIdentity:
        if self.conversation_ref is None and self.message_ref is None:
            raise ValueError("feedback_target_missing")
        if self.content_ref.tenant_ref != self.tenant_ref:
            raise ValueError("feedback_content_tenant_drift")
        if self.content_ref.workspace_ref != self.workspace_ref:
            raise ValueError("feedback_content_workspace_drift")
        return self


class SupportIdentity(_IdentityModel):
    tenant_ref: OpaqueRef
    workspace_ref: OpaqueRef
    support_ref: OpaqueRef
    requester_ref: OpaqueRef
    content_ref: ContentReference
    category: OpaqueRef
    state: Literal["open", "closed"] = "open"

    @model_validator(mode="after")
    def _support_scope_matches(self) -> SupportIdentity:
        if self.content_ref.tenant_ref != self.tenant_ref:
            raise ValueError("support_content_tenant_drift")
        if self.content_ref.workspace_ref != self.workspace_ref:
            raise ValueError("support_content_workspace_drift")
        return self


class RetentionRecord(_FrozenModel):
    tenant_ref: OpaqueRef
    workspace_ref: OpaqueRef
    entity_kind: EntityKind
    entity_ref: OpaqueRef
    state: LifecycleState
    version: Version
    legal_hold_ref: OpaqueRef | None = None
    resume_state: Literal["active", "retained", "deletion_pending"] | None = None
    effective_at: Timestamp

    @model_validator(mode="after")
    def _hold_state_is_consistent(self) -> RetentionRecord:
        if self.state == "legal_hold":
            if self.legal_hold_ref is None or self.resume_state is None:
                raise ValueError("legal_hold_metadata_missing")
        elif self.legal_hold_ref is not None or self.resume_state is not None:
            raise ValueError("legal_hold_metadata_unexpected")
        return self

    @property
    def retention_digest(self) -> str:
        return canonical_digest(self)


class PageRequest(_FrozenModel):
    limit: PageSize = 50
    cursor: Cursor | None = None


WebUiEntity: TypeAlias = (
    TenantIdentity
    | WorkspaceIdentity
    | UserIdentity
    | ProfileIdentity
    | SessionIdentity
    | ConversationIdentity
    | ContentReference
    | AttachmentIdentity
    | MessageIdentity
    | PreferenceIdentity
    | SavedQueryIdentity
    | DashboardIdentity
    | WidgetIdentity
    | NotificationIdentity
    | FeedbackIdentity
    | SupportIdentity
)
UiStateEntity: TypeAlias = (
    PreferenceIdentity | SavedQueryIdentity | DashboardIdentity | WidgetIdentity
)


class Page(_FrozenModel):
    items: tuple[WebUiEntity, ...] = Field(max_length=MAX_PAGE_SIZE)
    next_cursor: OpaqueRef | None = None


class UiStateWriteReceipt(_FrozenModel):
    """Proof that a UI-state write changed no GraphOS authority."""

    entity_kind: EntityKind
    entity_ref: OpaqueRef
    version: Version
    graphos_permission_mutations: Literal[0] = 0
    provider_grants: Literal[0] = 0
    public_shares: Literal[0] = 0


class AnonymousPilotSession(_FrozenModel):
    """Private-boundary pilot result containing no credential or public share."""

    session: SessionIdentity
    private_boundary: Literal[True] = True

    @model_validator(mode="after")
    def _pilot_is_private(self) -> AnonymousPilotSession:
        if not self.session.anonymous_pilot:
            raise ValueError("pilot_session_not_anonymous")
        return self
