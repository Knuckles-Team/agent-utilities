"""Application service for the tenant-scoped Web UI authority models."""

from __future__ import annotations

from collections.abc import Callable
from time import time
from typing import TypeVar, cast

from pydantic import BaseModel

from .errors import WebUiAuthorizationError, WebUiPilotBoundaryError
from .models import (
    AccessContext,
    AnonymousPilotSession,
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
    UiStateEntity,
    UiStateWriteReceipt,
    UserIdentity,
    WebUiEntity,
    WidgetIdentity,
    WorkspaceIdentity,
    canonical_digest,
)
from .protocols import WebUiRepository
from .repository import entity_kind_for

__all__ = ["WebUiService"]


EntityModelT = TypeVar("EntityModelT", bound=BaseModel)


class WebUiService:
    """Explicit-scope service with no GraphOS permission side effects."""

    def __init__(
        self,
        repository: WebUiRepository,
        *,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self._repository = repository
        self._clock = clock or (lambda: int(time()))

    @staticmethod
    def _pilot_guard(entity: WebUiEntity, context: AccessContext) -> None:
        if not isinstance(context, AccessContext):
            raise WebUiAuthorizationError()
        if not context.anonymous_pilot:
            return
        if getattr(entity, "visibility", "private") != "private":
            raise WebUiPilotBoundaryError()
        if getattr(entity, "session_ref", context.session_ref) != context.session_ref:
            raise WebUiPilotBoundaryError()
        if (
            getattr(entity, "user_ref", "user:anonymous-pilot")
            != "user:anonymous-pilot"
        ):
            raise WebUiPilotBoundaryError()
        if (
            getattr(entity, "owner_ref", "user:anonymous-pilot")
            != "user:anonymous-pilot"
        ):
            raise WebUiPilotBoundaryError()
        if (
            getattr(entity, "actor_ref", "actor:anonymous-pilot")
            != "actor:anonymous-pilot"
        ):
            raise WebUiPilotBoundaryError()
        if (
            getattr(entity, "requester_ref", "actor:anonymous-pilot")
            != "actor:anonymous-pilot"
        ):
            raise WebUiPilotBoundaryError()
        if isinstance(entity, SessionIdentity) and not entity.anonymous_pilot:
            raise WebUiPilotBoundaryError()

    def save_entity(
        self,
        entity: WebUiEntity,
        *,
        context: AccessContext,
        expected_version: int | None = None,
    ) -> WebUiEntity:
        """Persist one identity; tenant/workspace authority is never implicit."""

        self._pilot_guard(entity, context)
        return self._repository.put(
            entity,
            context=context,
            expected_version=expected_version,
        )

    def get_entity(
        self,
        entity_kind: EntityKind,
        entity_ref: str,
        *,
        context: AccessContext,
    ) -> WebUiEntity | None:
        """Read one identity inside the caller's exact tenant/workspace scope."""

        return self._repository.get(entity_kind, entity_ref, context=context)

    def list_entities(
        self,
        entity_kind: EntityKind,
        *,
        context: AccessContext,
        request: PageRequest,
    ) -> Page:
        """Read one bounded page; ACL filtering happens before ordering."""

        return self._repository.page(entity_kind, context=context, request=request)

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
        """Apply one policy-gated retention transition."""

        return self._repository.transition_retention(
            entity_kind,
            entity_ref,
            context=context,
            expected_version=expected_version,
            target=target,
            legal_hold_ref=legal_hold_ref,
        )

    def save_ui_state(
        self,
        entity: UiStateEntity,
        *,
        context: AccessContext,
        expected_version: int | None = None,
    ) -> UiStateWriteReceipt:
        """Save UI state while proving it cannot grant GraphOS authority."""

        saved = self.save_entity(
            entity,
            context=context,
            expected_version=expected_version,
        )
        return UiStateWriteReceipt(
            entity_kind=entity_kind_for(saved),
            entity_ref=str(getattr(saved, f"{entity_kind_for(saved)}_ref")),
            version=saved.version,
        )

    def open_anonymous_pilot(
        self,
        *,
        context: AccessContext,
        session_ref: str,
        expires_at: int,
    ) -> AnonymousPilotSession:
        """Open a private pilot session without minting any credential."""

        if not isinstance(context, AccessContext):
            raise WebUiAuthorizationError()
        if not context.anonymous_pilot or not context.allows("write"):
            raise WebUiPilotBoundaryError()
        if session_ref != context.session_ref:
            raise WebUiPilotBoundaryError()
        issued_at = max(0, int(self._clock()))
        session = SessionIdentity(
            tenant_ref=context.tenant_ref,
            workspace_ref=context.workspace_ref,
            session_ref=session_ref,
            user_ref="user:anonymous-pilot",
            anonymous_pilot=True,
            private_boundary=True,
            issued_at=issued_at,
            expires_at=expires_at,
            version=1,
            digest=canonical_digest(
                {
                    "tenant_ref": context.tenant_ref,
                    "workspace_ref": context.workspace_ref,
                    "session_ref": session_ref,
                    "user_ref": "user:anonymous-pilot",
                    "anonymous_pilot": True,
                    "private_boundary": True,
                    "issued_at": issued_at,
                    "expires_at": expires_at,
                }
            ),
        )
        stored = self.save_entity(session, context=context)
        return AnonymousPilotSession(session=cast(SessionIdentity, stored))

    def _save_typed(
        self,
        entity: EntityModelT,
        expected_type: type[EntityModelT],
        *,
        context: AccessContext,
        expected_version: int | None,
    ) -> EntityModelT:
        if not isinstance(entity, expected_type):
            raise TypeError("webui_entity_type_mismatch")
        return cast(
            EntityModelT,
            self.save_entity(
                cast(WebUiEntity, entity),
                context=context,
                expected_version=expected_version,
            ),
        )

    def save_tenant(
        self,
        entity: TenantIdentity,
        *,
        context: AccessContext,
        expected_version: int | None = None,
    ) -> TenantIdentity:
        return self._save_typed(
            entity, TenantIdentity, context=context, expected_version=expected_version
        )

    def save_workspace(
        self,
        entity: WorkspaceIdentity,
        *,
        context: AccessContext,
        expected_version: int | None = None,
    ) -> WorkspaceIdentity:
        return self._save_typed(
            entity,
            WorkspaceIdentity,
            context=context,
            expected_version=expected_version,
        )

    def save_user(
        self,
        entity: UserIdentity,
        *,
        context: AccessContext,
        expected_version: int | None = None,
    ) -> UserIdentity:
        return self._save_typed(
            entity, UserIdentity, context=context, expected_version=expected_version
        )

    def save_profile(
        self,
        entity: ProfileIdentity,
        *,
        context: AccessContext,
        expected_version: int | None = None,
    ) -> ProfileIdentity:
        return self._save_typed(
            entity, ProfileIdentity, context=context, expected_version=expected_version
        )

    def save_session(
        self,
        entity: SessionIdentity,
        *,
        context: AccessContext,
        expected_version: int | None = None,
    ) -> SessionIdentity:
        return self._save_typed(
            entity,
            SessionIdentity,
            context=context,
            expected_version=expected_version,
        )

    def save_conversation(
        self,
        entity: ConversationIdentity,
        *,
        context: AccessContext,
        expected_version: int | None = None,
    ) -> ConversationIdentity:
        return self._save_typed(
            entity,
            ConversationIdentity,
            context=context,
            expected_version=expected_version,
        )

    def save_content(
        self,
        entity: ContentReference,
        *,
        context: AccessContext,
        expected_version: int | None = None,
    ) -> ContentReference:
        return self._save_typed(
            entity,
            ContentReference,
            context=context,
            expected_version=expected_version,
        )

    def save_attachment(
        self,
        entity: AttachmentIdentity,
        *,
        context: AccessContext,
        expected_version: int | None = None,
    ) -> AttachmentIdentity:
        return self._save_typed(
            entity,
            AttachmentIdentity,
            context=context,
            expected_version=expected_version,
        )

    def save_message(
        self,
        entity: MessageIdentity,
        *,
        context: AccessContext,
        expected_version: int | None = None,
    ) -> MessageIdentity:
        return self._save_typed(
            entity,
            MessageIdentity,
            context=context,
            expected_version=expected_version,
        )

    def save_preference(
        self,
        entity: PreferenceIdentity,
        *,
        context: AccessContext,
        expected_version: int | None = None,
    ) -> PreferenceIdentity:
        return self._save_typed(
            entity,
            PreferenceIdentity,
            context=context,
            expected_version=expected_version,
        )

    def save_saved_query(
        self,
        entity: SavedQueryIdentity,
        *,
        context: AccessContext,
        expected_version: int | None = None,
    ) -> SavedQueryIdentity:
        return self._save_typed(
            entity,
            SavedQueryIdentity,
            context=context,
            expected_version=expected_version,
        )

    def save_dashboard(
        self,
        entity: DashboardIdentity,
        *,
        context: AccessContext,
        expected_version: int | None = None,
    ) -> DashboardIdentity:
        return self._save_typed(
            entity,
            DashboardIdentity,
            context=context,
            expected_version=expected_version,
        )

    def save_widget(
        self,
        entity: WidgetIdentity,
        *,
        context: AccessContext,
        expected_version: int | None = None,
    ) -> WidgetIdentity:
        return self._save_typed(
            entity, WidgetIdentity, context=context, expected_version=expected_version
        )

    def save_notification(
        self,
        entity: NotificationIdentity,
        *,
        context: AccessContext,
        expected_version: int | None = None,
    ) -> NotificationIdentity:
        return self._save_typed(
            entity,
            NotificationIdentity,
            context=context,
            expected_version=expected_version,
        )

    def save_feedback(
        self,
        entity: FeedbackIdentity,
        *,
        context: AccessContext,
        expected_version: int | None = None,
    ) -> FeedbackIdentity:
        return self._save_typed(
            entity,
            FeedbackIdentity,
            context=context,
            expected_version=expected_version,
        )

    def save_support(
        self,
        entity: SupportIdentity,
        *,
        context: AccessContext,
        expected_version: int | None = None,
    ) -> SupportIdentity:
        return self._save_typed(
            entity,
            SupportIdentity,
            context=context,
            expected_version=expected_version,
        )
