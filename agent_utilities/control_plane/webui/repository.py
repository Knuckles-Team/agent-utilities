"""Small, deterministic Web UI repository with explicit scope and CAS."""

from __future__ import annotations

import re
from collections.abc import Callable
from threading import RLock
from time import time
from typing import Literal, TypeAlias

from pydantic import BaseModel

from .errors import (
    WebUiAuthorizationError,
    WebUiCasConflictError,
    WebUiEntityNotFoundError,
    WebUiPaginationError,
    WebUiPilotBoundaryError,
    WebUiRetentionError,
)
from .models import (
    AccessContext,
    AttachmentIdentity,
    ContentReference,
    ConversationIdentity,
    DashboardIdentity,
    EntityKind,
    FeedbackIdentity,
    LifecycleState,
    MessageIdentity,
    NotificationIdentity,
    Page,
    PageRequest,
    PreferenceIdentity,
    ProfileIdentity,
    RetentionRecord,
    SavedQueryIdentity,
    SessionIdentity,
    SupportIdentity,
    TenantIdentity,
    UiPermission,
    UserIdentity,
    WebUiEntity,
    WidgetIdentity,
    WorkspaceIdentity,
)

__all__ = ["InMemoryWebUiRepository", "entity_kind_for"]


_REF_RE = re.compile(r"^[A-Za-z][A-Za-z0-9:_./-]{0,255}$")
_CURSOR_RE = re.compile(r"^cursor:([0-9]+)$")
_ENTITY_TYPES: tuple[tuple[type[BaseModel], EntityKind], ...] = (
    (TenantIdentity, "tenant"),
    (WorkspaceIdentity, "workspace"),
    (UserIdentity, "user"),
    (ProfileIdentity, "profile"),
    (SessionIdentity, "session"),
    (ConversationIdentity, "conversation"),
    (ContentReference, "content"),
    (AttachmentIdentity, "attachment"),
    (MessageIdentity, "message"),
    (PreferenceIdentity, "preference"),
    (SavedQueryIdentity, "saved_query"),
    (DashboardIdentity, "dashboard"),
    (WidgetIdentity, "widget"),
    (NotificationIdentity, "notification"),
    (FeedbackIdentity, "feedback"),
    (SupportIdentity, "support"),
)
_PILOT_AUTHORITY_KINDS = {"tenant", "workspace", "user", "profile"}
_RETENTION_TARGETS = {
    "active",
    "retained",
    "deletion_pending",
    "deleted",
    "legal_hold",
}
_RETENTION_ADMIN_TARGETS = {"deletion_pending", "deleted", "legal_hold"}
RetentionResumeState: TypeAlias = Literal["active", "retained", "deletion_pending"]
_STANDARD_RETENTION_TRANSITIONS: dict[
    tuple[LifecycleState, LifecycleState], LifecycleState
] = {
    ("active", "retained"): "retained",
    ("active", "deletion_pending"): "deletion_pending",
    ("retained", "deletion_pending"): "deletion_pending",
    ("deletion_pending", "deleted"): "deleted",
}


def entity_kind_for(entity: WebUiEntity) -> EntityKind:
    """Return the stable catalog kind without inspecting untrusted fields."""

    for model_type, entity_kind in _ENTITY_TYPES:
        if isinstance(entity, model_type):
            return entity_kind
    raise TypeError("unsupported_webui_entity")


def _validate_kind(entity_kind: EntityKind) -> None:
    if not isinstance(entity_kind, str) or entity_kind not in {
        kind for _, kind in _ENTITY_TYPES
    }:
        raise WebUiEntityNotFoundError()


def _validate_ref(entity_ref: str) -> None:
    if not isinstance(entity_ref, str) or _REF_RE.fullmatch(entity_ref) is None:
        raise WebUiEntityNotFoundError()


def _entity_ref(entity: WebUiEntity, entity_kind: EntityKind) -> str:
    ref_field = {
        "tenant": "tenant_ref",
        "workspace": "workspace_ref",
        "user": "user_ref",
        "profile": "profile_ref",
        "session": "session_ref",
        "conversation": "conversation_ref",
        "content": "content_ref",
        "attachment": "attachment_ref",
        "message": "message_ref",
        "preference": "preference_ref",
        "saved_query": "saved_query_ref",
        "dashboard": "dashboard_ref",
        "widget": "widget_ref",
        "notification": "notification_ref",
        "feedback": "feedback_ref",
        "support": "support_ref",
    }[entity_kind]
    return str(getattr(entity, ref_field))


def _entity_scope(
    entity: WebUiEntity,
    context: AccessContext,
) -> tuple[str, str]:
    workspace_ref = getattr(entity, "workspace_ref", None)
    if workspace_ref is None:
        workspace_ref = context.workspace_ref
    return (
        str(entity.tenant_ref),
        str(workspace_ref),
    )


def _validate_expected_version(expected_version: int | None) -> None:
    if expected_version is not None and (
        isinstance(expected_version, bool)
        or not isinstance(expected_version, int)
        or expected_version < 1
    ):
        raise WebUiCasConflictError()


def _pilot_ref_mismatch(entity: WebUiEntity, context: AccessContext) -> bool:
    expected_refs = (
        ("session_ref", context.session_ref),
        ("user_ref", "user:anonymous-pilot"),
        ("owner_ref", "user:anonymous-pilot"),
        ("actor_ref", "actor:anonymous-pilot"),
        ("requester_ref", "actor:anonymous-pilot"),
    )
    return any(
        getattr(entity, field_name, None) not in (None, expected)
        for field_name, expected in expected_refs
    )


def _pilot_boundary_violation(
    entity: WebUiEntity,
    entity_kind: EntityKind,
    context: AccessContext,
) -> bool:
    if entity_kind in _PILOT_AUTHORITY_KINDS:
        return True
    if getattr(entity, "visibility", "private") != "private":
        return True
    if _pilot_ref_mismatch(entity, context):
        return True
    return isinstance(entity, SessionIdentity) and not entity.anonymous_pilot


def _validate_retention_target(target: LifecycleState) -> None:
    if not isinstance(target, str) or target not in _RETENTION_TARGETS:
        raise WebUiRetentionError()


def _retention_permission(target: LifecycleState) -> UiPermission:
    if target in _RETENTION_ADMIN_TARGETS:
        return "admin"
    return "write"


def _same_retention_state(
    current: RetentionRecord,
    target: LifecycleState,
    legal_hold_ref: str | None,
) -> RetentionRecord:
    if target == "legal_hold":
        if legal_hold_ref != current.legal_hold_ref:
            raise WebUiRetentionError("legal_hold_reference_mismatch")
    elif legal_hold_ref is not None:
        raise WebUiRetentionError("legal_hold_reference_unexpected")
    return current


def _place_legal_hold(
    current: RetentionRecord,
    legal_hold_ref: str | None,
) -> tuple[LifecycleState, str | None, RetentionResumeState]:
    if current.state == "deleted":
        raise WebUiRetentionError("legal_hold_after_delete")
    if legal_hold_ref is None:
        raise WebUiRetentionError("legal_hold_reference_required")
    if current.state not in ("active", "retained", "deletion_pending"):
        raise WebUiRetentionError("legal_hold_source_state_invalid")
    return "legal_hold", legal_hold_ref, _retention_resume_state(current.state)


def _retention_resume_state(state: LifecycleState) -> RetentionResumeState:
    """Narrow a lifecycle state before persisting legal-hold metadata."""
    if state == "active" or state == "retained" or state == "deletion_pending":
        return state
    raise WebUiRetentionError("legal_hold_source_state_invalid")


def _release_legal_hold(
    current: RetentionRecord,
    target: LifecycleState,
    legal_hold_ref: str | None,
) -> tuple[LifecycleState, str | None, RetentionResumeState | None]:
    if legal_hold_ref is not None or target != current.resume_state:
        raise WebUiRetentionError("legal_hold_release_mismatch")
    return target, None, None


def _standard_retention_transition(
    current: RetentionRecord,
    target: LifecycleState,
) -> tuple[LifecycleState, str | None, RetentionResumeState | None]:
    next_state = _STANDARD_RETENTION_TRANSITIONS.get((current.state, target))
    if next_state is None:
        raise WebUiRetentionError("retention_transition_invalid")
    return next_state, None, None


def _retention_transition(
    current: RetentionRecord,
    target: LifecycleState,
    legal_hold_ref: str | None,
) -> tuple[LifecycleState, str | None, RetentionResumeState | None]:
    if target == "legal_hold":
        return _place_legal_hold(current, legal_hold_ref)
    if current.state == "legal_hold":
        return _release_legal_hold(current, target, legal_hold_ref)
    return _standard_retention_transition(current, target)


def _updated_retention(
    current: RetentionRecord,
    *,
    next_state: LifecycleState,
    next_hold: str | None,
    next_resume: RetentionResumeState | None,
    clock: Callable[[], int],
) -> RetentionRecord:
    if current.version >= 2_147_483_647:
        raise WebUiCasConflictError()
    return RetentionRecord(
        tenant_ref=current.tenant_ref,
        workspace_ref=current.workspace_ref,
        entity_kind=current.entity_kind,
        entity_ref=current.entity_ref,
        state=next_state,
        version=current.version + 1,
        legal_hold_ref=next_hold,
        resume_state=next_resume,
        effective_at=max(0, int(clock())),
    )


class InMemoryWebUiRepository:
    """A contract repository suitable for unit fixtures and local pilots.

    The production adapter can replace this implementation without changing the
    service seam.  Keys always include tenant and workspace, and retention
    records remain separate from entity versions so deletion cannot be bypassed
    by replaying an old identity.
    """

    def __init__(self, *, clock: Callable[[], int] | None = None) -> None:
        self._clock = clock or (lambda: int(time()))
        self._entities: dict[tuple[EntityKind, str, str, str], WebUiEntity] = {}
        self._retention: dict[tuple[EntityKind, str, str, str], RetentionRecord] = {}
        self._lock = RLock()

    @staticmethod
    def _require_context(context: AccessContext, permission: UiPermission) -> None:
        if not isinstance(context, AccessContext) or not context.allows(permission):
            raise WebUiAuthorizationError()

    @classmethod
    def _require_scope(
        cls,
        context: AccessContext,
        tenant_ref: str,
        workspace_ref: str,
        *,
        permission: UiPermission,
    ) -> None:
        cls._require_context(context, permission)
        if context.tenant_ref != tenant_ref or context.workspace_ref != workspace_ref:
            raise WebUiAuthorizationError()

    @staticmethod
    def _key(
        entity_kind: EntityKind,
        tenant_ref: str,
        workspace_ref: str,
        entity_ref: str,
    ) -> tuple[EntityKind, str, str, str]:
        return entity_kind, tenant_ref, workspace_ref, entity_ref

    @staticmethod
    def _pilot_boundary(
        entity: WebUiEntity,
        entity_kind: EntityKind,
        context: AccessContext,
    ) -> None:
        if not context.anonymous_pilot:
            return
        if _pilot_boundary_violation(entity, entity_kind, context):
            raise WebUiPilotBoundaryError()

    def _initial_retention(
        self,
        *,
        entity_kind: EntityKind,
        tenant_ref: str,
        workspace_ref: str,
        entity_ref: str,
    ) -> RetentionRecord:
        return RetentionRecord(
            tenant_ref=tenant_ref,
            workspace_ref=workspace_ref,
            entity_kind=entity_kind,
            entity_ref=entity_ref,
            state="active",
            version=1,
            effective_at=max(0, int(self._clock())),
        )

    def _retention_for(
        self,
        key: tuple[EntityKind, str, str, str],
    ) -> RetentionRecord:
        current = self._retention.get(key)
        if current is not None:
            return current
        entity_kind, tenant_ref, workspace_ref, entity_ref = key
        current = self._initial_retention(
            entity_kind=entity_kind,
            tenant_ref=tenant_ref,
            workspace_ref=workspace_ref,
            entity_ref=entity_ref,
        )
        self._retention[key] = current
        return current

    def put(
        self,
        entity: WebUiEntity,
        *,
        context: AccessContext,
        expected_version: int | None = None,
    ) -> WebUiEntity:
        """Create or update an identity with strict monotonic CAS."""

        _validate_expected_version(expected_version)
        self._require_context(context, "write")
        entity_kind = entity_kind_for(entity)
        tenant_ref, workspace_ref = _entity_scope(entity, context)
        entity_ref = _entity_ref(entity, entity_kind)
        _validate_ref(entity_ref)
        self._require_scope(
            context,
            tenant_ref,
            workspace_ref,
            permission="write",
        )
        self._pilot_boundary(entity, entity_kind, context)
        key = self._key(entity_kind, tenant_ref, workspace_ref, entity_ref)
        with self._lock:
            existing = self._entities.get(key)
            if existing is None:
                if expected_version is not None or entity.version != 1:
                    raise WebUiCasConflictError()
            else:
                if self._retention_for(key).state == "deleted":
                    raise WebUiRetentionError("webui_entity_deleted")
                if existing == entity and expected_version in {
                    None,
                    existing.version - 1,
                }:
                    return existing
                if expected_version != existing.version:
                    raise WebUiCasConflictError()
                if entity.version != existing.version + 1:
                    raise WebUiCasConflictError()
            self._entities[key] = entity
            if key not in self._retention:
                self._retention[key] = self._initial_retention(
                    entity_kind=entity_kind,
                    tenant_ref=tenant_ref,
                    workspace_ref=workspace_ref,
                    entity_ref=entity_ref,
                )
            return entity

    def get(
        self,
        entity_kind: EntityKind,
        entity_ref: str,
        *,
        context: AccessContext,
    ) -> WebUiEntity | None:
        """Read only the exact tenant/workspace key authorized by context."""

        _validate_kind(entity_kind)
        _validate_ref(entity_ref)
        self._require_context(context, "read")
        self._require_scope(
            context,
            context.tenant_ref,
            context.workspace_ref,
            permission="read",
        )
        key = self._key(
            entity_kind,
            context.tenant_ref,
            context.workspace_ref,
            entity_ref,
        )
        with self._lock:
            entity = self._entities.get(key)
            if entity is None:
                return None
            retention = self._retention.get(key)
            if retention is not None and retention.state == "deleted":
                raise WebUiRetentionError("webui_entity_deleted")
            return entity

    @staticmethod
    def _offset(cursor: str | None) -> int:
        if cursor is None:
            return 0
        match = _CURSOR_RE.fullmatch(cursor)
        if match is None:
            raise WebUiPaginationError()
        offset = int(match.group(1))
        if offset > 2_147_483_647:
            raise WebUiPaginationError()
        return offset

    def page(
        self,
        entity_kind: EntityKind,
        *,
        context: AccessContext,
        request: PageRequest,
    ) -> Page:
        """Return a deterministic bounded page after scope filtering."""

        _validate_kind(entity_kind)
        if not isinstance(request, PageRequest):
            raise WebUiPaginationError()
        offset = self._offset(request.cursor)
        self._require_context(context, "read")
        self._require_scope(
            context,
            context.tenant_ref,
            context.workspace_ref,
            permission="read",
        )
        prefix = (entity_kind, context.tenant_ref, context.workspace_ref)
        with self._lock:
            rows = [
                (key[3], entity)
                for key, entity in self._entities.items()
                if key[:3] == prefix
                and (
                    self._retention.get(key) is None
                    or self._retention[key].state != "deleted"
                )
            ]
        rows.sort(key=lambda row: row[0])
        selected = rows[offset : offset + request.limit]
        end = offset + len(selected)
        next_cursor = f"cursor:{end}" if end < len(rows) else None
        return Page(
            items=tuple(entity for _, entity in selected),
            next_cursor=next_cursor,
        )

    def transition_retention(
        self,
        entity_kind: EntityKind,
        entity_ref: str,
        *,
        context: AccessContext,
        expected_version: int,
        target: LifecycleState,
        legal_hold_ref: str | None = None,
    ) -> RetentionRecord:
        """Advance one entity's retention state without bypassing a hold."""

        _validate_kind(entity_kind)
        _validate_ref(entity_ref)
        _validate_expected_version(expected_version)
        _validate_retention_target(target)
        if legal_hold_ref is not None:
            _validate_ref(legal_hold_ref)
        permission = _retention_permission(target)
        self._require_context(context, permission)
        self._require_scope(
            context,
            context.tenant_ref,
            context.workspace_ref,
            permission=permission,
        )
        key = self._key(
            entity_kind,
            context.tenant_ref,
            context.workspace_ref,
            entity_ref,
        )
        with self._lock:
            if key not in self._entities:
                raise WebUiEntityNotFoundError()
            current = self._retention_for(key)
            if current.version != expected_version:
                raise WebUiCasConflictError()
            if target == current.state:
                return _same_retention_state(current, target, legal_hold_ref)
            next_state, next_hold, next_resume = _retention_transition(
                current,
                target,
                legal_hold_ref,
            )
            updated = _updated_retention(
                current,
                next_state=next_state,
                next_hold=next_hold,
                next_resume=next_resume,
                clock=self._clock,
            )
            self._retention[key] = updated
            return updated
