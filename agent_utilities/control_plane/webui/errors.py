"""Fail-closed errors for the Web UI control-plane boundary."""

from __future__ import annotations

from typing import ClassVar

__all__ = [
    "WebUiAuthorizationError",
    "WebUiCasConflictError",
    "WebUiDomainError",
    "WebUiEntityNotFoundError",
    "WebUiPaginationError",
    "WebUiPilotBoundaryError",
    "WebUiRetentionError",
]


class WebUiDomainError(ValueError):
    """Base error whose stable code can be recorded without leaking data."""

    code: ClassVar[str] = "webui_domain_error"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.code)


class WebUiAuthorizationError(WebUiDomainError):
    """The supplied context cannot access the requested tenant/workspace."""

    code = "webui_authorization_denied"


class WebUiCasConflictError(WebUiDomainError):
    """The caller supplied a stale or invalid monotonic version."""

    code = "webui_cas_conflict"


class WebUiEntityNotFoundError(WebUiDomainError):
    """A retention transition targeted no entity in the caller's scope."""

    code = "webui_entity_not_found"


class WebUiPaginationError(WebUiDomainError):
    """A cursor or page request was not bounded and canonical."""

    code = "webui_pagination_invalid"


class WebUiRetentionError(WebUiDomainError):
    """A lifecycle transition would bypass retention or legal-hold policy."""

    code = "webui_retention_invalid"


class WebUiPilotBoundaryError(WebUiDomainError):
    """An anonymous pilot operation crossed its private boundary."""

    code = "webui_anonymous_pilot_boundary"
