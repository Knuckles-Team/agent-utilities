"""Typed repository seams for the tenant-scoped Web UI authority."""

from __future__ import annotations

from typing import Protocol

from .models import (
    AccessContext,
    EntityKind,
    LifecycleState,
    Page,
    PageRequest,
    RetentionRecord,
    WebUiEntity,
)

__all__ = ["WebUiRepository"]


class WebUiRepository(Protocol):
    """Persistence authority; every operation carries its explicit scope."""

    def put(
        self,
        entity: WebUiEntity,
        *,
        context: AccessContext,
        expected_version: int | None,
    ) -> WebUiEntity:
        """Create or update one immutable identity under CAS."""

    def get(
        self,
        entity_kind: EntityKind,
        entity_ref: str,
        *,
        context: AccessContext,
    ) -> WebUiEntity | None:
        """Read one identity after scope authorization."""

    def page(
        self,
        entity_kind: EntityKind,
        *,
        context: AccessContext,
        request: PageRequest,
    ) -> Page:
        """Read a bounded, scope-filtered page."""

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
        """Apply one guarded retention/legal-hold transition."""
