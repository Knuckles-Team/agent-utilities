"""Tenant-scoped Web UI control-plane authority."""

from .errors import (
    WebUiAuthorizationError,
    WebUiCasConflictError,
    WebUiDomainError,
    WebUiEntityNotFoundError,
    WebUiPaginationError,
    WebUiPilotBoundaryError,
    WebUiRetentionError,
)
from .models import __all__ as _model_exports
from .models import *  # noqa: F403
from .protocols import WebUiRepository
from .repository import InMemoryWebUiRepository, entity_kind_for
from .service import WebUiService

__all__ = [
    *_model_exports,
    "InMemoryWebUiRepository",
    "WebUiAuthorizationError",
    "WebUiCasConflictError",
    "WebUiDomainError",
    "WebUiEntityNotFoundError",
    "WebUiPaginationError",
    "WebUiPilotBoundaryError",
    "WebUiRepository",
    "WebUiRetentionError",
    "WebUiService",
    "entity_kind_for",
]
