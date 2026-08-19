"""Focused contract fixtures for the Web UI control-plane authority."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from agent_utilities.control_plane.webui import (
    AccessContext,
    AttachmentIdentity,
    ContentReference,
    ConversationIdentity,
    DashboardIdentity,
    InMemoryWebUiRepository,
    PageRequest,
    PreferenceIdentity,
    SavedQueryIdentity,
    WebUiAuthorizationError,
    WebUiCasConflictError,
    WebUiEntityNotFoundError,
    WebUiPilotBoundaryError,
    WebUiRetentionError,
    WebUiService,
    WidgetIdentity,
)


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode()).hexdigest()}"


def _context(
    *,
    tenant_ref: str = "tenant:one",
    workspace_ref: str = "workspace:one",
    session_ref: str = "session:one",
) -> AccessContext:
    return AccessContext(
        tenant_ref=tenant_ref,
        workspace_ref=workspace_ref,
        actor_ref="actor:user",
        session_ref=session_ref,
        permissions=("read", "write", "feedback", "support", "admin"),
    )


def _anonymous_context() -> AccessContext:
    return AccessContext(
        tenant_ref="tenant:pilot",
        workspace_ref="workspace:pilot",
        actor_ref="actor:anonymous-pilot",
        session_ref="session:pilot",
        permissions=("read", "write", "feedback"),
        authenticated=False,
        anonymous_pilot=True,
    )


def _content(context: AccessContext, ref: str = "content:one") -> ContentReference:
    return ContentReference(
        tenant_ref=context.tenant_ref,
        workspace_ref=context.workspace_ref,
        content_ref=ref,
        artifact_ref=f"artifact:{ref.removeprefix('content:')}",
        content_digest=_digest(ref),
        media_type="text/plain",
        size_bytes=3,
        version=1,
        digest=_digest(f"identity:{ref}"),
    )


def _conversation(
    context: AccessContext,
    ref: str,
    *,
    version: int = 1,
) -> ConversationIdentity:
    return ConversationIdentity(
        tenant_ref=context.tenant_ref,
        workspace_ref=context.workspace_ref,
        conversation_ref=ref,
        session_ref=context.session_ref,
        state="active",
        version=version,
        digest=_digest(f"conversation:{ref}:{version}"),
    )


def test_models_reject_inline_material_and_scope_drift() -> None:
    context = _context()
    with pytest.raises(ValidationError, match="inline_ui_material_or_authority_forbidden"):
        ContentReference(
            tenant_ref=context.tenant_ref,
            workspace_ref=context.workspace_ref,
            content_ref="content:one",
            artifact_ref="artifact:one",
            content_digest=_digest("content"),
            media_type="text/plain",
            size_bytes=3,
            version=1,
            digest=_digest("identity"),
            text="raw content",
        )
    with pytest.raises(ValidationError):
        SavedQueryIdentity(
            tenant_ref=context.tenant_ref,
            workspace_ref=context.workspace_ref,
            saved_query_ref="query:one",
            owner_ref="user:one",
            query_ref="query-ref:one",
            visibility="public",
            version=1,
            digest=_digest("query"),
        )
    with pytest.raises(ValidationError, match="attachment_workspace_drift"):
        AttachmentIdentity(
            tenant_ref=context.tenant_ref,
            workspace_ref=context.workspace_ref,
            attachment_ref="attachment:one",
            content_ref=_content(
                _context(workspace_ref="workspace:other"),
                "content:other",
            ),
            attachment_digest=_digest("attachment"),
            media_type="text/plain",
            size_bytes=3,
            version=1,
            digest=_digest("attachment-identity"),
        )


def test_scope_is_explicit_and_cross_scope_reads_are_non_oracular() -> None:
    context = _context()
    other = _context(tenant_ref="tenant:other", workspace_ref="workspace:other")
    service = WebUiService(InMemoryWebUiRepository())
    entity = _conversation(context, "conversation:one")

    service.save_conversation(entity, context=context)
    assert service.get_entity("conversation", entity.conversation_ref, context=other) is None
    assert service.list_entities(
        "conversation",
        context=other,
        request=PageRequest(limit=10),
    ).items == ()
    with pytest.raises(WebUiAuthorizationError):
        service.save_conversation(entity, context=other)
    with pytest.raises(WebUiEntityNotFoundError):
        service.transition_retention(
            "conversation",
            entity.conversation_ref,
            context=other,
            expected_version=1,
            target="retained",
        )


def test_cas_and_pagination_are_bounded_and_deterministic() -> None:
    context = _context()
    service = WebUiService(InMemoryWebUiRepository())
    first = service.save_conversation(
        _conversation(context, "conversation:one"),
        context=context,
    )
    service.save_conversation(_conversation(context, "conversation:two"), context=context)
    updated = _conversation(context, "conversation:one", version=2)
    service.save_conversation(updated, context=context, expected_version=first.version)
    with pytest.raises(WebUiCasConflictError):
        service.save_conversation(
            _conversation(context, "conversation:one", version=3),
            context=context,
            expected_version=first.version,
        )

    with pytest.raises(ValidationError):
        PageRequest(limit=101)
    page = service.list_entities(
        "conversation",
        context=context,
        request=PageRequest(limit=1),
    )
    assert len(page.items) == 1
    assert page.next_cursor == "cursor:1"
    next_page = service.list_entities(
        "conversation",
        context=context,
        request=PageRequest(limit=1, cursor=page.next_cursor),
    )
    assert [item.conversation_ref for item in next_page.items] == ["conversation:two"]
    with pytest.raises(ValidationError):
        PageRequest(limit=1, cursor="offset:1")


def test_retention_requires_ordered_cas_and_honors_legal_hold() -> None:
    context = _context()
    service = WebUiService(InMemoryWebUiRepository())
    entity = _conversation(context, "conversation:retained")
    service.save_conversation(entity, context=context)

    pending = service.transition_retention(
        "conversation",
        entity.conversation_ref,
        context=context,
        expected_version=1,
        target="deletion_pending",
    )
    assert pending.version == 2
    deleted = service.transition_retention(
        "conversation",
        entity.conversation_ref,
        context=context,
        expected_version=pending.version,
        target="deleted",
    )
    assert deleted.state == "deleted"
    with pytest.raises(WebUiRetentionError):
        service.get_entity("conversation", entity.conversation_ref, context=context)

    held = _conversation(context, "conversation:held")
    service.save_conversation(held, context=context)
    hold = service.transition_retention(
        "conversation",
        held.conversation_ref,
        context=context,
        expected_version=1,
        target="legal_hold",
        legal_hold_ref="hold:case-1",
    )
    with pytest.raises(WebUiRetentionError):
        service.transition_retention(
            "conversation",
            held.conversation_ref,
            context=context,
            expected_version=hold.version,
            target="deleted",
        )
    released = service.transition_retention(
        "conversation",
        held.conversation_ref,
        context=context,
        expected_version=hold.version,
        target="active",
    )
    assert released.state == "active"


def test_ui_state_receipt_cannot_mutate_graphos_authority() -> None:
    context = _context()
    service = WebUiService(InMemoryWebUiRepository())
    preference = PreferenceIdentity(
        tenant_ref=context.tenant_ref,
        workspace_ref=context.workspace_ref,
        preference_ref="preference:one",
        user_ref="user:one",
        state_ref="state:preference-one",
        version=1,
        digest=_digest("preference"),
    )
    receipt = service.save_ui_state(preference, context=context)
    assert receipt.graphos_permission_mutations == 0
    assert receipt.provider_grants == 0
    assert receipt.public_shares == 0
    assert not hasattr(service, "grant_graph_permission")

    widget = WidgetIdentity(
        tenant_ref=context.tenant_ref,
        workspace_ref=context.workspace_ref,
        widget_ref="widget:one",
        dashboard_ref="dashboard:one",
        widget_type="table",
        state_ref="state:widget-one",
        version=1,
        digest=_digest("widget"),
    )
    dashboard = DashboardIdentity(
        tenant_ref=context.tenant_ref,
        workspace_ref=context.workspace_ref,
        dashboard_ref="dashboard:one",
        owner_ref="user:one",
        widgets=(widget,),
        version=1,
        digest=_digest("dashboard"),
    )
    assert service.save_ui_state(dashboard, context=context).version == 1


def test_anonymous_pilot_is_private_and_cannot_mint_authority() -> None:
    context = _anonymous_context()
    service = WebUiService(
        InMemoryWebUiRepository(),
        clock=lambda: 1_000,
    )
    pilot = service.open_anonymous_pilot(
        context=context,
        session_ref=context.session_ref,
        expires_at=1_060,
    )
    assert pilot.private_boundary is True
    assert pilot.session.private_boundary is True
    assert not hasattr(pilot, "token")
    assert not hasattr(pilot, "provider_grant")
    assert not hasattr(pilot, "public_share")

    with pytest.raises(WebUiPilotBoundaryError):
        service.save_ui_state(
            SavedQueryIdentity(
                tenant_ref=context.tenant_ref,
                workspace_ref=context.workspace_ref,
                saved_query_ref="query:pilot",
                owner_ref="user:anonymous-pilot",
                query_ref="query-ref:pilot",
                visibility="workspace",
                version=1,
                digest=_digest("pilot-query"),
            ),
            context=context,
        )
    with pytest.raises(ValidationError):
        AccessContext(
            tenant_ref=context.tenant_ref,
            workspace_ref=context.workspace_ref,
            actor_ref="actor:anonymous-pilot",
            session_ref=context.session_ref,
            permissions=("read", "admin"),
            authenticated=False,
            anonymous_pilot=True,
        )
